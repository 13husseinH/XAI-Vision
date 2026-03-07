import torch
import torchvision.transforms.functional as TF


class ImageMasker:

    def __init__(self,
                 fill_mode="constant",
                 constant_value=0.0,
                 noise_std=0.15,
                 blur_kernel_size=21,
                 blur_sigma=6.0):

        self.fill_mode = fill_mode
        self.constant_value = constant_value
        self.noise_std = noise_std
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma


    # =========================
    # MAIN FUNCTION
    # =========================

    def generate_all(self, image, grid_size=3):

        results = []

        results += self.mask_grid(image, grid_size)
        results += self.mask_corners(image)
        results += self.mask_border(image)
        results += self.mask_center(image)

        return results


    # =========================
    # GRID MASK
    # =========================

    def mask_grid(self, image, grid_size=3):

        C, H, W = image.shape
        cell_h = H // grid_size
        cell_w = W // grid_size

        results = []

        for i in range(grid_size):
            for j in range(grid_size):

                y1 = i * cell_h
                y2 = H if i == grid_size - 1 else (i + 1) * cell_h

                x1 = j * cell_w
                x2 = W if j == grid_size - 1 else (j + 1) * cell_w

                masked = image.clone()
                masked[:, y1:y2, x1:x2] = self.fill_region(image, y1, y2, x1, x2)

                results.append({
                    "image": masked,
                    "bbox": (y1, y2, x1, x2),
                })

        return results


    # =========================
    # CORNERS
    # =========================

    def mask_corners(self, image, fraction=0.25):

        C, H, W = image.shape
        h = int(H * fraction)
        w = int(W * fraction)

        corners = [
            (0, h, 0, w),
            (0, h, W - w, W),
            (H - h, H, 0, w),
            (H - h, H, W - w, W),
        ]

        results = []

        for (y1, y2, x1, x2) in corners:
            masked = image.clone()
            masked[:, y1:y2, x1:x2] = self.fill_region(image, y1, y2, x1, x2)
            results.append({
                "image": masked,
                "bbox": (y1, y2, x1, x2),
            })

        return results


    # =========================
    # BORDER
    # =========================

    def mask_border(self, image, thickness=0.18):

        C, H, W = image.shape
        t_h = int(H * thickness)
        t_w = int(W * thickness)

        borders = [
            (0, t_h, 0, W),
            (H - t_h, H, 0, W),
            (0, H, 0, t_w),
            (0, H, W - t_w, W),
        ]

        results = []

        for (y1, y2, x1, x2) in borders:
            masked = image.clone()
            masked[:, y1:y2, x1:x2] = self.fill_region(image, y1, y2, x1, x2)
            results.append({
                "image": masked,
                "bbox": (y1, y2, x1, x2),
            })

        return results


    # =========================
    # CENTER
    # =========================

    def mask_center(self, image, fraction=0.45):

        C, H, W = image.shape
        h = int(H * fraction)
        w = int(W * fraction)

        y1 = (H - h) // 2
        y2 = y1 + h
        x1 = (W - w) // 2
        x2 = x1 + w

        masked = image.clone()
        masked[:, y1:y2, x1:x2] = self.fill_region(image, y1, y2, x1, x2)

        return [{
            "image": masked,
            "bbox": (y1, y2, x1, x2),
        }]


    # =========================
    # FILL REGION
    # =========================

    def fill_region(self, image, y1, y2, x1, x2):

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
