from __future__ import annotations
import json, math, re
from .timeline import shot_window, master_frame_count, frame_to_sample

def _parse_script(text):
    text=(text or "").strip()
    if not text:
        raise ValueError("Script is empty.")
    if text.startswith("{") or text.startswith("["):
        data=json.loads(text)
        if isinstance(data, dict):
            shots=[str(x).strip() for x in data.get("prompts",[]) if str(x).strip()]
        elif isinstance(data,list):
            shots=[str(x).strip() for x in data if str(x).strip()]
        else:
            shots=[]
    else:
        shots=[x.strip() for x in re.split(r"(?m)^---\s*$",text) if x.strip()]
    if not shots:
        raise ValueError("No shot prompts found.")
    return shots

def _sampler_names():
    try:
        import comfy.samplers
        return list(comfy.samplers.KSampler.SAMPLERS)
    except Exception:
        return ["res_multistep","euler","dpmpp_2m"]

def _scheduler_names():
    try:
        import comfy.samplers
        return list(comfy.samplers.KSampler.SCHEDULERS)
    except Exception:
        return ["simple","normal","beta"]

def _wave3(w):
    if w.ndim==2:
        w=w.unsqueeze(0)
    if w.ndim!=3:
        raise ValueError(f"Expected AUDIO [B,C,L], got {tuple(w.shape)}")
    return w[:1]

def _stereo(w):
    if w.shape[1]==1:
        return w.repeat(1,2,1)
    if w.shape[1]>2:
        return w[:,:2,:]
    return w

def _prepare_vae_audio(audio,audio_vae):
    from .semantic_ref import _resample_audio
    w=_stereo(_wave3(audio["waveform"]))
    sr=int(audio["sample_rate"])
    vr=int(getattr(audio_vae,"audio_sample_rate",32000))
    if sr!=vr:
        w=_resample_audio(w,sr,vr)
    return w.contiguous(),vr

def _slice_pad(w,start,length):
    import torch.nn.functional as F
    start,length=int(start),int(length)
    total=int(w.shape[-1])
    if length<=0:
        raise ValueError("Audio slice length must be > 0")
    if start<0:
        left=min(-start,length)
        body=w[...,:max(0,length-left)]
        return F.pad(body,(left,max(0,length-left-body.shape[-1])))
    if start>=total:
        return F.pad(w[...,:0],(0,length))
    out=w[...,start:min(total,start+length)]
    if out.shape[-1]<length:
        out=F.pad(out,(0,length-out.shape[-1]))
    return out

def _exact_master_output(audio,target_samples,policy):
    import torch.nn.functional as F
    w=_wave3(audio["waveform"])
    sr=int(audio["sample_rate"])
    have=int(w.shape[-1])
    if have<target_samples:
        if policy=="error":
            raise ValueError(
                f"Master audio is {have/sr:.3f}s but video needs {target_samples/sr:.3f}s.")
        w=F.pad(w,(0,target_samples-have))
    else:
        w=w[...,:target_samples]
    return {"waveform":w.contiguous(),"sample_rate":sr}

def _lock_audio_latent(av_latent,audio_vae,segment):
    import torch
    import torch.nn.functional as F
    import comfy.nested_tensor
    s=av_latent.get("samples")
    if s is None or not getattr(s,"is_nested",False):
        raise ValueError("Expected a joint MiniMax H3 AV latent.")
    parts=list(s.unbind())
    if len(parts)<2:
        raise ValueError("H3 latent has no audio stream.")
    video,template=parts[0],parts[1]
    za=audio_vae.encode(segment.movedim(1,-1))
    target_t=int(template.shape[-1])
    if za.shape[-1]>target_t:
        za=za[...,:target_t]
    elif za.shape[-1]<target_t:
        za=F.pad(za,(0,target_t-za.shape[-1]))
    out=dict(av_latent)
    out["samples"]=comfy.nested_tensor.NestedTensor((video,za))
    out["noise_mask"]=comfy.nested_tensor.NestedTensor(
        (torch.ones_like(video),torch.zeros_like(za)))
    return out

def _audio_locked_model(model):
    m=model.clone()
    tr=m.model_options["transformer_options"]=m.model_options.get(
        "transformer_options",{}).copy()
    tr["minimax_h3_lock_audio_clean"]=True
    return m

def _evict_te(clip,model):
    try:
        import comfy.model_management as mm
        te=getattr(clip.patcher,"load_device",None)
        di=getattr(model,"load_device",None)
        if te is not None and di is not None and str(te)!=str(di):
            return
        try:
            clip.patcher.model.to(mm.text_encoder_offload_device())
        except Exception:
            pass
        d=mm.get_torch_device()
        mm.free_memory(mm.get_total_memory(d)*0.9,d)
        mm.soft_empty_cache()
        print(f"[H3ExactMemory] TE evicted; {mm.get_free_memory(d)/(1024**3):.1f} GB free",flush=True)
    except Exception as e:
        print(f"[H3ExactMemory] TE eviction skipped: {e}",flush=True)

def _avbank_patch_active():
    try:
        import comfy.model_base as mb
        return bool(getattr(mb.MiniMaxH3.extra_conds,"_h3_avbank_merge",False))
    except Exception:
        return False

class H3ExactAudioSegment:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "master_audio":("AUDIO",),
            "shot_index":("INT",{"default":0,"min":0,"max":1024}),
            "frames_per_shot":("INT",{"default":243,"min":5,"max":4096}),
            "overlap_frames":("INT",{"default":1,"min":0,"max":240}),
            "fps":("FLOAT",{"default":24.0,"min":1.0,"max":240.0,"step":0.001}),
        }}
    RETURN_TYPES=("AUDIO","FLOAT","FLOAT","INT")
    RETURN_NAMES=("audio_segment","start_seconds","duration_seconds","trim_head_frames")
    FUNCTION="slice"
    CATEGORY="MiniMax H3/Exact Master Audio"
    def slice(self,master_audio,shot_index,frames_per_shot,overlap_frames,fps):
        w=shot_window(shot_index,frames_per_shot,overlap_frames)
        wav=_wave3(master_audio["waveform"])
        sr=int(master_audio["sample_rate"])
        start=frame_to_sample(w.generation_start_frame,sr,fps)
        length=frame_to_sample(w.generation_frames,sr,fps)
        seg=_slice_pad(wav,start,length)
        return ({"waveform":seg.contiguous(),"sample_rate":sr},
                w.generation_start_seconds(fps),
                w.generation_duration_seconds(fps),
                w.trim_head_frames)

class H3ExactMasterAudioMemorySampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":{
                "model":("MODEL",),"clip":("CLIP",),"video_vae":("VAE",),"audio_vae":("VAE",),
                "master_audio":("AUDIO",),
                "script":("STRING",{"multiline":True,"dynamicPrompts":False,
                    "default":"Shot 1 prompt.\n---\nShot 2 prompt.\n---\nShot 3 prompt."}),
                "shot_count":("INT",{"default":0,"min":0,"max":64}),
                "width":("INT",{"default":1344,"min":32,"max":4096,"step":16}),
                "height":("INT",{"default":768,"min":32,"max":4096,"step":16}),
                "frames_per_shot":("INT",{"default":243,"min":5,"max":1000,"step":17}),
                "seed":("INT",{"default":0,"min":0,"max":0xffffffffffffffff}),
                "steps":("INT",{"default":20,"min":1,"max":50}),
                "memory_frames":("INT",{"default":2,"min":0,"max":6}),
                "anchor_frames":("INT",{"default":1,"min":0,"max":2}),
                "seed_per_shot":("BOOLEAN",{"default":True}),
                "short_audio_policy":(["error","pad_silence"],{"default":"error"}),
            },
            "optional":{
                "start_image":("IMAGE",),
                "reference_images":("IMAGE",{
                    "tooltip":"Fixed REF2VA identity/product/style references. Requires current ComfyUI-H3-Multishot AV-bank patch."}),
                "reference_image_size":(["match","max"],{"default":"match"}),
                "sampler_name":(_sampler_names(),{"default":"res_multistep"}),
                "scheduler":(_scheduler_names(),{"default":"simple"}),
            }}
    RETURN_TYPES=("IMAGE","AUDIO","INT","STRING")
    RETURN_NAMES=("master_frames","exact_master_audio","shots_rendered","timeline_json")
    FUNCTION="run"
    CATEGORY="sampling/minimax"

    def run(self,model,clip,video_vae,audio_vae,master_audio,script,shot_count,
            width,height,frames_per_shot,seed,steps,memory_frames,anchor_frames,
            seed_per_shot=True,short_audio_policy="error",start_image=None,
            reference_images=None,reference_image_size="match",
            sampler_name="res_multistep",scheduler="simple"):
        import torch,node_helpers
        from comfy_extras import nodes_custom_sampler as ncs
        from comfy_extras import nodes_minimax_h3 as mmh3

        fps=24.0
        shots=_parse_script(script)
        n=int(shot_count) if int(shot_count)>0 else len(shots)
        shots=shots[:n]
        while len(shots)<n:
            shots.append(shots[-1])

        final_frames=master_frame_count(n,frames_per_shot,1)
        osr=int(master_audio["sample_rate"])
        exact_master=_exact_master_output(
            master_audio,frame_to_sample(final_frames,osr,fps),short_audio_policy)
        vae_master,vae_sr=_prepare_vae_audio(master_audio,audio_vae)

        smodel=_audio_locked_model(model)
        sigmas=ncs.BasicScheduler().get_sigmas(smodel,scheduler,steps,1.0)[0]
        sampler=ncs.KSamplerSelect().get_sampler(sampler_name)[0]

        ref_items,ref_blocks=[],[]
        if reference_images is not None:
            if not _avbank_patch_active():
                raise RuntimeError(
                    "Fixed REF2VA references + chaining keyframes require the current "
                    "ComfyUI-H3-Multishot refs+keyframes AV-bank patch.")
            for i in range(int(reference_images.shape[0])):
                img=reference_images[i:i+1]
                ih,iw=int(img.shape[1]),int(img.shape[2])
                if reference_image_size=="match":
                    scale=min(1.0,math.sqrt((width*height)/float(iw*ih)))
                else:
                    edge=int(getattr(mmh3,"REF_IMAGE_SHORT_EDGE",2048))
                    scale=min(1.0,edge/float(min(iw,ih)))
                multiple=int(getattr(mmh3,"CANVAS_MULTIPLE",32))
                tw=max(multiple,round(iw*scale/multiple)*multiple)
                th=max(multiple,round(ih*scale/multiple)*multiple)
                im=mmh3._resize(img,tw,th,"disabled")
                z=video_vae.encode(im)
                ref_items.append({"type":"image","data":im})
                ref_blocks.append({"kind":"image","latent_h":th//16,"latent_w":tw//16,"latent":z})
            print(f"[H3ExactMemory] encoded {len(ref_blocks)} fixed reference(s) once.",flush=True)

        anchor=start_image[:1].clone() if start_image is not None else None
        history=[]
        frame_parts=[]
        timeline=[]

        for si,prompt in enumerate(shots):
            w=shot_window(si,frames_per_shot,1)
            latent,frame_count=mmh3._empty_av_latent(width,height,frames_per_shot)

            mem=[]
            if anchor is not None and anchor_frames>0:
                mem.extend([anchor]*min(int(anchor_frames),2))
            if history:
                take=int(memory_frames) if int(memory_frames)>0 else 1
                mem.extend(history[-take:])
            mem=[mmh3._resize(x[:1],width,height,"disabled") for x in mem]

            cont=history[-1] if history else anchor
            chain=None
            keyframes=[]
            if cont is not None:
                chain=mmh3._resize(cont[:1],width,height,"disabled")
                keyframes=[{"resolved_frame_index":0,"latent":video_vae.encode(chain)}]

            if ref_blocks:
                items=list(ref_items)
                items.extend({"type":"image","data":x} for x in mem)
                if chain is not None:
                    items.append({"type":"image","data":chain})
                tokens=clip.tokenize(prompt,minimax_ref_items=items)
            else:
                ims=list(mem)
                if chain is not None:
                    ims.append(chain)
                tokens=clip.tokenize(prompt,images=ims)

            cond=clip.encode_from_tokens_scheduled(tokens)
            vals={}
            if keyframes:
                vals["minimax_keyframes"]=keyframes
                vals["minimax_frame_count"]=frame_count
            if ref_blocks:
                vals["minimax_refs"]=ref_blocks
            if vals:
                cond=node_helpers.conditioning_set_values(cond,vals)

            _evict_te(clip,smodel)

            astart=frame_to_sample(w.generation_start_frame,vae_sr,fps)
            alen=frame_to_sample(w.generation_frames,vae_sr,fps)
            aseg=_slice_pad(vae_master,astart,alen)
            locked=_lock_audio_latent(latent,audio_vae,aseg)

            guider=ncs.BasicGuider().get_guider(smodel,cond)[0]
            sseed=int(seed)+si if seed_per_shot else int(seed)
            noise=ncs.RandomNoise().get_noise(sseed)[0]
            out,_=ncs.SamplerCustomAdvanced().sample(noise,guider,sampler,sigmas,locked)

            video=list(out["samples"].unbind())[0]
            imgs=video_vae.decode(video)
            if imgs.ndim==5:
                imgs=imgs.reshape(-1,imgs.shape[-3],imgs.shape[-2],imgs.shape[-1])

            if anchor is None and anchor_frames>0:
                anchor=imgs[:1].detach().clone()
            history.append(imgs[-1:].detach().clone())
            if len(history)>max(8,int(memory_frames)+2):
                history.pop(0)

            if si>0:
                imgs=imgs[1:]
            frame_parts.append(imgs.cpu())
            timeline.append({
                "shot":si+1,
                "generation_start_frame":w.generation_start_frame,
                "generation_start_seconds":w.generation_start_seconds(fps),
                "trim_head_frames":w.trim_head_frames,
                "delivered_frames":int(imgs.shape[0]),
                "audio":"original_master"
            })

        master_frames=torch.cat(frame_parts,dim=0)
        if int(master_frames.shape[0])!=int(final_frames):
            raise RuntimeError(
                f"Timeline invariant failed: expected {final_frames} frames, got {master_frames.shape[0]}.")

        manifest={
            "version":1,"fps":fps,"shots":n,"frames_per_shot":int(frames_per_shot),
            "overlap_frames":1,"master_frames":int(master_frames.shape[0]),
            "master_seconds":master_frames.shape[0]/fps,
            "reference_images":int(reference_images.shape[0]) if reference_images is not None else 0,
            "memory_frames":int(memory_frames),"anchor_frames":int(anchor_frames),
            "timeline":timeline
        }
        print(f"[H3ExactMemory] done: {n} shots, {master_frames.shape[0]} frames; "
              "generated H3 audio was not decoded.",flush=True)
        return (master_frames,exact_master,n,json.dumps(manifest,indent=2))

NODE_CLASS_MAPPINGS={
    "H3ExactAudioSegment":H3ExactAudioSegment,
    "H3ExactMasterAudioMemorySampler":H3ExactMasterAudioMemorySampler,
}
NODE_DISPLAY_NAME_MAPPINGS={
    "H3ExactAudioSegment":"H3 Exact Master Audio Segment",
    "H3ExactMasterAudioMemorySampler":"H3 Exact Master Audio + Memory Sampler",
}
