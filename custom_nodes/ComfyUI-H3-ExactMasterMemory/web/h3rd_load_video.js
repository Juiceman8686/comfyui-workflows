// H3 Reference Director - video loader preview.
//
// ComfyUI already shows a preview of the chosen video on this node. This script changes how that
// preview behaves; it does not create the preview and the node works without this file.
//
//   - the preview plays in a loop, without sound
//   - while the mouse is over the video, its sound is on
//   - only the part that will really be loaded is played: skip_first_frames, skip_last_frames and
//     frame_load_cap move the start and the end of the loop
import { app } from "/scripts/app.js";

const NODE_TYPE = "H3RDLoadVideo";
const enhanced = new WeakSet();

// The part of the clip that is loaded, in seconds. Pure function, so it can be tested on its own.
export function trimRange(duration, s) {
    const fps = s.force_rate > 0 ? s.force_rate : 24;      // 0 = "keep the file's rate", unknown here: assume 24
    const whole = { start: 0, end: duration, empty: false };
    if (!(duration > 0)) return whole;
    const start = Math.max(0, s.skip_first) / fps;
    let end = duration - Math.max(0, s.skip_last) / fps;
    if (s.cap > 0) end = Math.min(end, start + (s.cap * Math.max(1, s.nth)) / fps);
    if (!(end - start >= 1 / fps)) return { start: 0, end: duration, empty: true };   // nothing left: show all
    return { start, end: Math.min(end, duration), empty: false };
}

function isOurs(node) {
    return node?.comfyClass === NODE_TYPE || node?.type === NODE_TYPE;
}

function value(node, name, fallback) {
    const w = node?.widgets?.find((x) => x.name === name);
    const v = Number(w?.value);
    return Number.isFinite(v) ? v : fallback;
}

export function settingsOf(node) {
    return {
        force_rate: value(node, "force_rate", 24),
        skip_first: value(node, "skip_first_frames", 0),
        skip_last: value(node, "skip_last_frames", 0),
        cap: value(node, "frame_load_cap", 0),
        nth: value(node, "select_every_nth", 1),
    };
}

function play(video, afterUnmute) {
    try {
        const p = video.play?.();
        if (p?.catch) {
            p.catch(() => {
                // the browser refused sound without a click: go back to silent playback
                if (afterUnmute) {
                    video.muted = true;
                    video.play?.()?.catch?.(() => {});
                }
            });
        }
    } catch (e) { /* not playable yet */ }
}

export function enhance(video, node) {
    if (!video || enhanced.has(video)) return false;
    enhanced.add(video);
    video.loop = true;
    video.muted = true;
    video.autoplay = true;
    video.playsInline = true;

    const keepInRange = () => {
        const d = video.duration;
        if (!Number.isFinite(d) || d <= 0) return;
        const r = trimRange(d, settingsOf(node));
        if (video.currentTime < r.start - 0.04 || video.currentTime >= r.end - 0.02) {
            try { video.currentTime = r.start; } catch (e) { /* not seekable yet */ }
        }
    };

    video.addEventListener("loadedmetadata", () => { keepInRange(); play(video); });
    video.addEventListener("timeupdate", keepInRange);
    video.addEventListener("ended", () => { keepInRange(); play(video); });
    video.addEventListener("mouseenter", () => { video.muted = false; play(video, true); });
    video.addEventListener("mouseleave", () => { video.muted = true; });

    // timeupdate fires only a few times a second; check every shown frame where the browser allows it
    if (typeof video.requestVideoFrameCallback === "function") {
        const tick = () => {
            keepInRange();
            if (video.isConnected !== false) video.requestVideoFrameCallback(tick);
        };
        video.requestVideoFrameCallback(tick);
    }
    if (video.readyState >= 1) keepInRange();
    play(video);
    return true;
}

// Every <video> element that belongs to this node, in the classic and in the new node renderer.
function videosOf(node) {
    const found = new Set();
    const add = (root) => {
        try {
            if (!root) return;
            if (root.tagName === "VIDEO") found.add(root);
            root.querySelectorAll?.("video")?.forEach((v) => found.add(v));
        } catch (e) { /* ignore */ }
    };
    add(node.videoContainer);
    for (const w of node.widgets ?? []) { add(w.element); add(w.inputEl); }
    if (typeof document !== "undefined" && node.id != null) {
        for (const sel of [`[data-node-id="${node.id}"] video`, `[node-id="${node.id}"] video`, `#node-${node.id} video`]) {
            try { document.querySelectorAll(sel).forEach((v) => found.add(v)); } catch (e) { /* ignore */ }
        }
    }
    return [...found];
}

export function scanNode(node) {
    if (!isOurs(node)) return 0;
    let n = 0;
    for (const v of videosOf(node)) if (enhance(v, node)) n++;
    return n;
}

function scanAll() {
    try {
        const g = app.graph;
        for (const node of g?._nodes ?? g?.nodes ?? []) scanNode(node);
    } catch (e) { /* ignore */ }
}

function soon(node) {
    for (const ms of [0, 150, 600, 1500, 3000]) setTimeout(() => { try { scanNode(node); } catch (e) { /* ignore */ } }, ms);
}

app.registerExtension({
    name: "H3RD.LoadVideo.Preview",
    setup() {
        // The preview element is created by ComfyUI some time after a file is chosen; look for it regularly.
        setInterval(scanAll, 1200);
    },
    nodeCreated(node) {
        if (!isOurs(node)) return;
        try {
            const w = node.widgets?.find((x) => x.name === "video");
            if (w && !w.__h3rdHooked) {
                w.__h3rdHooked = true;
                const original = w.callback;
                w.callback = function (...args) {
                    const r = original ? original.apply(this, args) : undefined;
                    soon(node);
                    return r;
                };
            }
            soon(node);
        } catch (e) { console.warn("[H3RD] load video preview", e); }
    },
    loadedGraphNode(node) {
        if (isOurs(node)) soon(node);
    },
});
