"""Scene-level analysis of a clip: cuts, size, and the TRACK/GENERAL call.

``detect_scenes`` wraps scene_detection (imported on first use: it pulls
scenedetect and, with TransNetV2, torch). ``analyze_scenes_strategy`` decides
per scene whether to follow one face or show the whole frame, from the face
detector in vision.py. Moved out of main.py with the detectors so the layout
modules and the API can reach them without importing the pipeline.
"""
import cv2
import numpy as np

import frame_sampler
import vision


# NOTE: a "route text-heavy scenes to GENERAL" rule was tried here and removed
# on 26-jul-2026. The problem it targets is real — a screencast that happens to
# contain one face gets cropped to the face and its headlines come out cut
# mid-word — but edge density is the wrong signal for it. Measured: a
# constructed talking-head-beside-a-chart scored 0.012 while the SAME shot
# without the panels scored 0.029, because a flat panel of text has far fewer
# edges than ordinary scene detail. Canny measures visual busyness, not text.
# A real fix needs an actual text detector (MSER/EAST) validated against clips
# that contain the failure mode; this corpus has almost none.


def analyze_scenes_strategy(video_path, scenes):
    """
    Analyzes each scene to determine if it should be TRACK (Single person) or GENERAL (Group/Wide).
    Returns list of strategies corresponding to scenes.
    """
    from tqdm import tqdm

    cap = cv2.VideoCapture(video_path)
    strategies = []

    if not cap.isOpened():
        return ['TRACK'] * len(scenes)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Every scene's samples first, then one forward pass over the clip
    # (frame_sampler): same frames, same detector calls in the same order.
    plan = []
    for si, (start, end) in enumerate(scenes):
        s_f, e_f = start.get_frames(), end.get_frames()
        # Sample 5 frames spread across the scene, clamped inside it (the old
        # start+5/end-5 samples landed outside scenes shorter than ~10 frames).
        margin = min(2, max(0, (e_f - s_f - 1) // 2))
        frames_to_check = sorted(set(
            int(round(f)) for f in np.linspace(s_f + margin, e_f - 1 - margin, 5)
        ))
        plan.extend((si, f_idx) for f_idx in frames_to_check)

    counts_per_scene = [[] for _ in scenes]
    frames = frame_sampler.read_at_path(video_path, cap, [f_idx for _si, f_idx in plan])
    for (si, _f_idx), frame in tqdm(zip(plan, frames), total=len(plan),
                                    desc="   Analyzing Scenes"):
        if frame is None:
            continue

        # Near-black frames (fades, cut-to-black) carry no faces and used
        # to drag single-person scenes into GENERAL. Skip them.
        if frame.mean() < 16:
            continue

        # Detect faces
        candidates = vision.detect_face_candidates(frame)
        counts_per_scene[si].append(len(candidates))

    for face_counts in counts_per_scene:
        # Decision Logic
        if not face_counts:
            avg_faces = 0
        else:
            avg_faces = sum(face_counts) / len(face_counts)

        # Strategy:
        # 0 faces -> GENERAL (Landscape/B-roll)
        # 1 face -> TRACK
        # > 1.2 faces -> GENERAL (Group)

        if avg_faces > 1.2 or avg_faces < 0.5:
            strategies.append('GENERAL')
        else:
            strategies.append('TRACK')

    cap.release()

    # Hysteresis: a short scene whose two neighbors agree on the opposite
    # strategy is almost always a sampling miss (profile face, insert shot).
    # Each TRACK<->GENERAL flip is a full on-screen layout change, so flapping
    # is worse than an occasional wrong-but-stable choice.
    max_flip_frames = int(2.0 * fps)
    for i in range(1, len(strategies) - 1):
        dur = scenes[i][1].get_frames() - scenes[i][0].get_frames()
        if (dur < max_flip_frames
                and strategies[i - 1] == strategies[i + 1] != strategies[i]):
            strategies[i] = strategies[i - 1]

    return strategies

def detect_scenes(video_path):
    import scene_detection
    return scene_detection.detect_scenes(video_path)

def get_video_resolution(video_path):
    probe = cv2.VideoCapture(video_path)
    try:
        if not probe.isOpened():
            raise IOError(f"cannot open video: {video_path}")
        return (int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        probe.release()
