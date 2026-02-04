"""
This module receives an input image tensor and generates controlled masked
variants by altering specific regions (grid cells, corners, borders, center).

Each masked variant perturbs exactly one region at a time and returns
the modified image along with metadata describing the perturbation.

This module does not load images, run model inference, or compute scores.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torchvision.transforms.functional as TF


MaskFillMode = Literal["constant", "mean", "noise"]
MaskKind = Literal["occlude", "blur", "desaturate"]


@dataclass
class MaskResult:
    """
    Container for a single masked image and its metadata.
    """
    image: torch.Tensor
    region: str
    kind: MaskKind
    bbox: Tuple[int, int, int, int]
    params: Dict

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["image"] = None
        return d


class ImageMasker:
    """
    Produces masked variants of an image to probe model sensitivity
    to different spatial regions and visual cues.
    """

    def __init__(
        self,
        fill_mode: MaskFillMode = "constant",
        constant_value: float = 0.0,
        noise_std: float = 0.15,
        blur_kernel_size: int = 21,
        blur_sigma: float = 6.0,
        device: Optional[torch.device] = None,
    ):
        self.fill_mode = fill_mode
        self.constant_value = float(constant_value)
        self.noise_std = float(noise_std)
        self.blur_kernel_size = int(blur_kernel_size)
        self.blur_sigma = float(blur_sigma)
        self.device = device

    def generate_all(
        self,
        image: torch.Tensor,
        grid_size: int = 3,
        corner_fraction: float = 0.25,
        border_thickness: float = 0.18,
        center_fraction: float = 0.45,
        random_patches: int = 0,
        random_patch_fraction: float = 0.2,
        include_blur: bool = True,
        include_desaturate: bool = True,
    ) -> List[MaskResult]:
        self._validate_image(image)
        results: List[MaskResult] = []

        results += self.mask_grid(image, grid_size=grid_size, kind="occlude")
        results += self.mask_corners(image, fraction=corner_fraction, kind="occlude")
        results += self.mask_border(image, thickness=border_thickness, kind="occlude")
        results += self.mask_center(image, fraction=center_fraction, kind="occlude")

        if random_patches > 0:
            results += self.mask_random_patches(
                image,
                n=random_patches,
                patch_fraction=random_patch_fraction,
                kind="occlude",
            )

        if include_blur:
            results += self.mask_corners(image, fraction=corner_fraction, kind="blur")
            results += self.mask_border(image, thickness=border_thickness, kind="blur")
            results += self.mask_grid(image, grid_size=grid_size, kind="blur")

        if include_desaturate:
            results += self.mask_border(image, thickness=border_thickness, kind="desaturate")
            results += self.mask_grid(image, grid_size=grid_size, kind="desaturate")

        return results

    def mask_grid(self, image: torch.Tensor, grid_size: int = 3, kind: MaskKind = "occlude") -> List[MaskResult]:
        self._validate_image(image)
        _, H, W = image.shape
        cell_h = H // grid_size
        cell_w = W // grid_size

        results: List[MaskResult] = []
        for i in range(grid_size):
            for j in range(grid_size):
                y1 = i * cell_h
                y2 = H if i == grid_size - 1 else (i + 1) * cell_h
                x1 = j * cell_w
                x2 = W if j == grid_size - 1 else (j + 1) * cell_w
                results.append(
                    self._apply_region(
                        image,
                        (y1, y2, x1, x2),
                        region=f"grid_{i}_{j}",
                        kind=kind,
                        params={"grid_size": grid_size},
                    )
                )
        return results

    def mask_corners(self, image: torch.Tensor, fraction: float = 0.25, kind: MaskKind = "occlude") -> List[MaskResult]:
        self._validate_image(image)
        _, H, W = image.shape
        h = int(H * fraction)
        w = int(W * fraction)

        corners = {
            "top_left": (0, h, 0, w),
            "top_right": (0, h, W - w, W),
            "bottom_left": (H - h, H, 0, w),
            "bottom_right": (H - h, H, W - w, W),
        }

        return [
            self._apply_region(image, bbox, region=name, kind=kind, params={"fraction": fraction})
            for name, bbox in corners.items()
        ]

    def mask_border(self, image: torch.Tensor, thickness: float = 0.18, kind: MaskKind = "occlude") -> List[MaskResult]:
        self._validate_image(image)
        _, H, W = image.shape
        t_h = int(H * thickness)
        t_w = int(W * thickness)

        borders = {
            "border_top": (0, t_h, 0, W),
            "border_bottom": (H - t_h, H, 0, W),
            "border_left": (0, H, 0, t_w),
            "border_right": (0, H, W - t_w, W),
        }

        return [
            self._apply_region(image, bbox, region=name, kind=kind, params={"thickness": thickness})
            for name, bbox in borders.items()
        ]

    def mask_center(self, image: torch.Tensor, fraction: float = 0.45, kind: MaskKind = "occlude") -> List[MaskResult]:
        self._validate_image(image)
        _, H, W = image.shape
        h = int(H * fraction)
        w = int(W * fraction)

        y1 = (H - h) // 2
        y2 = y1 + h
        x1 = (W - w) // 2
        x2 = x1 + w

        return [
            self._apply_region(
                image,
                (y1, y2, x1, x2),
                region="center",
                kind=kind,
                params={"fraction": fraction},
            )
        ]

    def mask_random_patches(
        self,
        image: torch.Tensor,
        n: int = 10,
        patch_fraction: float = 0.2,
        kind: MaskKind = "occlude",
        seed: Optional[int] = 0,
    ) -> List[MaskResult]:
        self._validate_image(image)
        _, H, W = image.shape
        ph = int(H * patch_fraction)
        pw = int(W * patch_fraction)

        g = torch.Generator(device=image.device)
        if seed is not None:
            g.manual_seed(seed)

        results: List[MaskResult] = []
        for k in range(n):
            y1 = int(torch.randint(0, max(1, H - ph), (1,), generator=g).item())
            x1 = int(torch.randint(0, max(1, W - pw), (1,), generator=g).item())
            y2 = y1 + ph
            x2 = x1 + pw
            results.append(
                self._apply_region(
                    image,
                    (y1, y2, x1, x2),
                    region=f"rand_{k}",
                    kind=kind,
                    params={"patch_fraction": patch_fraction, "seed": seed},
                )
            )
        return results

    def _apply_region(
        self,
        image: torch.Tensor,
        bbox: Tuple[int, int, int, int],
        region: str,
        kind: MaskKind,
        params: Dict,
    ) -> MaskResult:
        y1, y2, x1, x2 = bbox
        masked = image.clone()

        if kind == "occlude":
            masked[:, y1:y2, x1:x2] = self._fill_region(image, bbox)

        elif kind == "blur":
            blurred = TF.gaussian_blur(
                image,
                kernel_size=[self._odd(self.blur_kernel_size)] * 2,
                sigma=[self.blur_sigma] * 2,
            )
            masked[:, y1:y2, x1:x2] = blurred[:, y1:y2, x1:x2]

        elif kind == "desaturate":
            if image.shape[0] == 3:
                region_tensor = masked[:, y1:y2, x1:x2]
                gray = (
                    0.299 * region_tensor[0]
                    + 0.587 * region_tensor[1]
                    + 0.114 * region_tensor[2]
                )
                masked[:, y1:y2, x1:x2] = gray.unsqueeze(0).repeat(3, 1, 1)

        return MaskResult(
            image=masked,
            region=region,
            kind=kind,
            bbox=(y1, y2, x1, x2),
            params=params,
        )

    def _fill_region(self, image: torch.Tensor, bbox: Tuple[int, int, int, int]) -> torch.Tensor:
        y1, y2, x1, x2 = bbox
        C = image.shape[0]
        h = y2 - y1
        w = x2 - x1

        if self.fill_mode == "constant":
            return torch.full((C, h, w), self.constant_value, device=image.device)

        mean = image.view(C, -1).mean(dim=1).view(C, 1, 1)

        if self.fill_mode == "mean":
            return mean.expand(C, h, w)

        if self.fill_mode == "noise":
            noise = torch.randn((C, h, w), device=image.device) * self.noise_std
            return (mean.expand(C, h, w) + noise).clamp(0.0, 1.0)

        raise ValueError("Invalid fill_mode")

    def _validate_image(self, image: torch.Tensor) -> None:
        if not isinstance(image, torch.Tensor):
            raise TypeError("Image must be a torch.Tensor")
        if image.ndim != 3:
            raise ValueError("Image must have shape (C, H, W)")
        if image.shape[0] not in (1, 3):
            raise ValueError("Image must have 1 or 3 channels")

    @staticmethod
    def _odd(k: int) -> int:
        return k if k % 2 == 1 else k + 1
