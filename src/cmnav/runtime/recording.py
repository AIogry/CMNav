"""Run artifacts with explicit observation/action alignment and depth units."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cmnav.runtime.artifacts import json_safe, write_json


class EpisodeRecorder:
    def __init__(self, directory: Path, *, save_every: int, fps: int, max_depth: float):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=False)
        (directory / "frames").mkdir()
        self.stream = (directory / "steps.jsonl").open("w", encoding="utf-8")
        self.save_every, self.fps, self.max_depth = save_every, fps, max_depth
        self.preview_paths: list[Path] = []

    def record(self, *, step: int, action: str | None, observation: dict[str, Any],
               metrics: dict[str, Any], episode_over: bool) -> None:
        if "rgb" not in observation or "depth" not in observation or "pointgoal" not in observation:
            raise ValueError("Expected rgb, depth and fixed pointgoal observations")
        frames: dict[str, str] = {}
        if step % self.save_every == 0 or episode_over:
            rgb = np.asarray(observation["rgb"])
            depth = np.asarray(observation["depth"], dtype=np.float32)
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[:, :, 0]
            if rgb.ndim != 3 or rgb.shape[2] not in (3, 4) or rgb.dtype != np.uint8:
                raise ValueError(f"Expected uint8 HxWx3/4 RGB; got {rgb.shape}, {rgb.dtype}")
            if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
                raise ValueError(f"RGB and depth dimensions differ: {rgb.shape}, {depth.shape}")
            rgb_path = self.directory / "frames" / f"{step:06d}.rgb.png"
            depth_path = self.directory / "frames" / f"{step:06d}.depth.npy"
            preview_path = self.directory / "frames" / f"{step:06d}.depth.png"
            Image.fromarray(rgb[:, :, :3]).save(rgb_path)
            np.save(depth_path, depth, allow_pickle=False)
            visible_depth = np.nan_to_num(depth, nan=0.0, posinf=self.max_depth, neginf=0.0)
            visible_depth = (np.clip(visible_depth / self.max_depth, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(visible_depth).save(preview_path)
            frames = {"rgb": str(rgb_path.relative_to(self.directory)),
                      "depth_metres": str(depth_path.relative_to(self.directory)),
                      "depth_preview": str(preview_path.relative_to(self.directory))}
            self.preview_paths.append(rgb_path)
        record = {"observation_step": step, "action_from_previous_observation": action,
                  "pointgoal": observation["pointgoal"], "frames": frames,
                  "evaluation_metrics": metrics, "episode_over": episode_over}
        self.stream.write(json.dumps(json_safe(record), allow_nan=False) + "\n")
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()

    def save_preview(self) -> None:
        # Keep previews bounded when an experiment contains thousands of steps.
        paths = self.preview_paths
        if len(paths) > 200:
            paths = [paths[int(index)] for index in np.linspace(0, len(paths) - 1, 200)]
        if not paths:
            return
        frames = []
        try:
            for path in paths:
                with Image.open(path) as frame:
                    preview = frame.convert("RGB")
                    preview.thumbnail((512, 512))
                    frames.append(preview)
            frames[0].save(self.directory / "preview.gif", save_all=True,
                           append_images=frames[1:], duration=max(1, round(1000 / self.fps)), loop=0)
        finally:
            for frame in frames:
                frame.close()

