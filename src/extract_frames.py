"""Extract frames from a video file with OpenCV.

Sampling (e.g. every 10th frame) is used instead of processing every
frame: consecutive video frames are nearly identical, so sampling
keeps prediction fast without losing information.
"""

import argparse
from pathlib import Path

import cv2


def extract_frames(video_path, frame_interval=10, max_frames=None):
    """Read a video and return a list of sampled BGR frames.

    Args:
        video_path: Path to the video file (str or pathlib.Path).
        frame_interval: Keep 1 frame out of every ``frame_interval``
            frames. Must be >= 1.
        max_frames: Optional cap on how many frames to return.

    Returns:
        A list of frames, each a NumPy array in BGR format.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If OpenCV cannot open the video.
        ValueError: If ``frame_interval`` is less than 1.
    """
    if frame_interval < 1:
        raise ValueError("frame_interval must be >= 1")

    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open the video: {path}")

    frames = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_interval == 0:
                frames.append(frame)
                if max_frames is not None and len(frames) >= max_frames:
                    break
            frame_index += 1
    finally:
        capture.release()
    return frames


def main():
    parser = argparse.ArgumentParser(description="Save sampled frames from a video as JPG images.")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("output_dir", help="Folder to write frame images into")
    parser.add_argument("--frame-interval", type=int, default=10,
                        help="Keep 1 frame out of every N frames (default: 10)")
    args = parser.parse_args()

    frames = extract_frames(args.video_path, frame_interval=args.frame_interval)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(output_dir / f"frame_{i:05d}.jpg"), frame)
    print(f"Saved {len(frames)} frame(s) to {output_dir}")


if __name__ == "__main__":
    main()
