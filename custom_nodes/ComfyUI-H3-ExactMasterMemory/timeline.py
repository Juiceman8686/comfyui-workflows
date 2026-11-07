from dataclasses import dataclass

@dataclass(frozen=True)
class ShotWindow:
    shot_index: int
    generation_start_frame: int
    generation_frames: int
    trim_head_frames: int
    kept_start_frame: int
    kept_frames: int

    def generation_start_seconds(self, fps):
        return self.generation_start_frame / float(fps)

    def generation_duration_seconds(self, fps):
        return self.generation_frames / float(fps)

def shot_window(shot_index, frames_per_shot, overlap_frames=1):
    shot_index = int(shot_index)
    frames_per_shot = int(frames_per_shot)
    overlap_frames = int(overlap_frames)
    if shot_index < 0:
        raise ValueError("shot_index must be >= 0")
    if frames_per_shot < 2:
        raise ValueError("frames_per_shot must be >= 2")
    if overlap_frames < 0 or overlap_frames >= frames_per_shot:
        raise ValueError("overlap_frames must be in [0, frames_per_shot)")
    stride = frames_per_shot - overlap_frames
    generation_start = shot_index * stride
    trim = 0 if shot_index == 0 else overlap_frames
    kept_start = generation_start + trim
    kept_frames = frames_per_shot if shot_index == 0 else stride
    return ShotWindow(shot_index, generation_start, frames_per_shot, trim, kept_start, kept_frames)

def master_frame_count(shot_count, frames_per_shot, overlap_frames=1):
    shot_count = int(shot_count)
    if shot_count <= 0:
        return 0
    return int(frames_per_shot) + (shot_count - 1) * (int(frames_per_shot) - int(overlap_frames))

def frame_to_sample(frame, sample_rate, fps):
    return int(round(int(frame) / float(fps) * int(sample_rate)))
