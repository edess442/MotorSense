import random
import torch
import torchvision.transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms.autoaugment import RandAugment, _apply_op
from torchvision.transforms import RandomResizedCrop

class _VideoCompose:
    """
    Compose-style wrapper for video transforms.
    Applies the same transform consistently across frames.
    frames: List[PIL.Image] or List[Tensor]
    """
    def __init__(self, ops):
        self.ops = ops

    def __call__(self, frames):
        for op in self.ops:
            frames = op(frames)
        return frames


def _resolve_fill_for_frame(frame, fill):
    if torch.is_tensor(frame):
        channels, _, _ = F.get_dimensions(frame)
        if isinstance(fill, (int, float)):
            return [float(fill)] * channels
        if fill is not None:
            return [float(v) for v in fill]
    return fill


def build_video_transforms(cfg):
    ops = []

    for t in cfg:
        ttype = t["type"]

        # --------------------------------------------------
        # Resize (deterministic, per-frame)
        # --------------------------------------------------
        if ttype == "resize":
            size = t["size"]
            antialias = t.get("antialias", True)

            def _resize(frames, size=size):
                return [
                    F.resize(f, size, antialias=antialias)
                    for f in frames
                ]

            ops.append(_resize)

        # --------------------------------------------------
        # Center Crop (val)
        # --------------------------------------------------
        elif ttype == "center_crop":
            size = t["size"]

            def _center_crop(frames, size=size):
                return [F.center_crop(f, size) for f in frames]

            ops.append(_center_crop)

        # --------------------------------------------------
        # Random Resized Crop (train, sampled once)
        # --------------------------------------------------
        elif ttype == "random_resized_crop":
            size = t["size"]
            scale = tuple(t.get("scale", (0.8, 1.0)))
            ratio = tuple(t.get("ratio", (3.0 / 4.0, 4.0 / 3.0)))

            def _rrc(frames, size=size, scale=scale, ratio=ratio):
                i, j, h, w = RandomResizedCrop.get_params(
                    frames[0],
                    scale=scale,
                    ratio=ratio,
                )
                return [
                    F.resized_crop(f, i, j, h, w, size)
                    for f in frames
                ]

            ops.append(_rrc)

        # --------------------------------------------------
        # Color Jitter (train, sampled once per clip)
        # --------------------------------------------------
        elif ttype == "color_jitter":
            brightness = t.get("brightness", 0.4)
            contrast   = t.get("contrast",   0.4)
            saturation = t.get("saturation", 0.4)
            hue        = t.get("hue",        0.1)
            p          = t.get("p",          1.0)  # probability of applying at all

            cj = torchvision.transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )

            def _color_jitter(frames, cj=cj, p=p):
                if random.random() > p:
                    return frames
                # Sample the transform ONCE so all frames get identical jitter
                fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
                    cj.get_params(cj.brightness, cj.contrast, cj.saturation, cj.hue)
                out = []
                for f in frames:
                    for fn_id in fn_idx:
                        if fn_id == 0 and brightness_factor is not None:
                            f = F.adjust_brightness(f, brightness_factor)
                        elif fn_id == 1 and contrast_factor is not None:
                            f = F.adjust_contrast(f, contrast_factor)
                        elif fn_id == 2 and saturation_factor is not None:
                            f = F.adjust_saturation(f, saturation_factor)
                        elif fn_id == 3 and hue_factor is not None:
                            f = F.adjust_hue(f, hue_factor)
                    out.append(f)
                return out

            ops.append(_color_jitter)

        # --------------------------------------------------
        # RandAugment (train, sampled once)
        # --------------------------------------------------
        elif ttype == "rand_augment":
            num_ops = t.get("num_ops", 2)
            magnitude = t.get("magnitude", 9)
            magnitude_std = float(t.get("magnitude_std", 0.0))
            interpolation = t.get("interpolation", "bilinear")
            fill = t.get("fill", 128)

            interp = (
                InterpolationMode.BILINEAR
                if interpolation == "bilinear"
                else InterpolationMode.NEAREST
            )

            ra = RandAugment(
                num_ops=num_ops,
                magnitude=magnitude,
                interpolation=interp,
                fill=fill,
            )

            def _randaug(frames, ra=ra, fill=fill, magnitude_std=magnitude_std):
                if not frames:
                    return frames

                _, height, width = F.get_dimensions(frames[0])
                op_meta = ra._augmentation_space(ra.num_magnitude_bins, (height, width))
                op_names = list(op_meta.keys())

                if magnitude_std > 0.0:
                    sampled_magnitude = int(round(random.gauss(float(ra.magnitude), magnitude_std)))
                else:
                    sampled_magnitude = int(ra.magnitude)
                sampled_magnitude = max(0, min(int(ra.num_magnitude_bins) - 1, sampled_magnitude))

                sampled_ops = []
                for _ in range(int(ra.num_ops)):
                    op_index = int(torch.randint(len(op_names), (1,)).item())
                    op_name = op_names[op_index]
                    magnitudes, signed = op_meta[op_name]
                    op_magnitude = float(magnitudes[sampled_magnitude].item()) if getattr(magnitudes, "ndim", 0) > 0 else 0.0
                    if signed and bool(torch.randint(2, (1,)).item()):
                        op_magnitude *= -1.0
                    sampled_ops.append((op_name, op_magnitude))

                out = []
                for frame in frames:
                    frame_fill = _resolve_fill_for_frame(frame, fill)
                    transformed = frame
                    for op_name, op_magnitude in sampled_ops:
                        transformed = _apply_op(
                            transformed,
                            op_name,
                            op_magnitude,
                            interpolation=ra.interpolation,
                            fill=frame_fill,
                        )
                    out.append(transformed)
                return out

            ops.append(_randaug)

        # --------------------------------------------------
        # To Image (PIL → Tensor, video-aware)
        # --------------------------------------------------
        elif ttype == "to_image":
            def _to_image(frames):
                return [F.to_tensor(f) for f in frames]
            ops.append(_to_image)

        # --------------------------------------------------
        # To DType (Tensor dtype + scaling)
        # --------------------------------------------------
        elif ttype == "to_dtype":
            dtype = getattr(torch, t["dtype"])
            scale = t.get("scale", True)

            def _to_dtype(frames, dtype=dtype, scale=scale):
                return [
                    F.convert_image_dtype(f, dtype=dtype)
                    if scale else f.to(dtype)
                    for f in frames
                ]

            ops.append(_to_dtype)

        # --------------------------------------------------
        # Normalize (Tensor-only, per-frame)
        # --------------------------------------------------
        elif ttype == "normalize":
            mean = t["mean"]
            std = t["std"]

            def _norm(frames, mean=mean, std=std):
                return [
                    F.normalize(f, mean=mean, std=std)
                    for f in frames
                ]

            ops.append(_norm)

        else:
            raise ValueError(f"Unknown video transform: {ttype}")

    return _VideoCompose(ops)
