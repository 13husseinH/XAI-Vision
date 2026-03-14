import torch


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


    def build_result(self, masked, y1, y2, x1, x2, mask_type, label):
        return {
            "image": masked,
            "bbox": (y1, y2, x1, x2),
            "mask_type": mask_type,
            "label": label,
        }


    # =========================
    # GRID MASK
    # =========================

    def mask_grid(self, image, grid_size=3):

        _c, h, w = image.shape
        cell_h = h // grid_size
        cell_w = w // grid_size

        results = []

        for i in range(grid_size):
            for j in range(grid_size):

                y1 = i * cell_h
                y2 = h if i == grid_size - 1 else (i + 1) * cell_h

                x1 = j * cell_w
                x2 = w if j == grid_size - 1 else (j + 1) * cell_w

                masked = image.clone()
                masked[:, y1:y2, x1:x2] = self.fill_region(image, y1, y2, x1, x2)

                results.append(self.build_result(masked, y1, y2, x1, x2, "grid", f"grid-{i}-{j}"))

        return results


    # =========================
    # CORNERS
    # =========================

    def mask_corners(self, image, fraction=0.25):

        _c, h_total, w_total = image.shape
        h = int(h_total * fraction)
        w = int(w_total * fraction)

        corners = [
            (0, h, 0, w, "corner-top-left"),
            (0, h, w_total - w, w_total, "corner-top-right"),
            (h_total - h, h_total, 0, w, "corner-bottom-left"),
            (h_total - h, h_total, w_total - w, w_total, "corner-bottom-right"),
        ]

        results = []

        for (y1, y2, x1, x2, label) in corners:
            masked = image.clone()
            masked[:, y1:y2, x1:x2] = self.fill_region(image, y1, y2, x1, x2)
            results.append(self.build_result(masked, y1, y2, x1, x2, "corner", label))

        return results


    # =========================
    # BORDER
    # =========================

    def mask_border(self, image, thickness=0.18):

        _c, h, w = image.shape
        t_h = int(h * thickness)
        t_w = int(w * thickness)

        borders = [
            (0, t_h, 0, w, "border-top"),
            (h - t_h, h, 0, w, "border-bottom"),
            (0, h, 0, t_w, "border-left"),
            (0, h, w - t_w, w, "border-right"),
        ]

        results = []

        for (y1, y2, x1, x2, label) in borders:
            masked = image.clone()
            masked[:, y1:y2, x1:x2] = self.fill_region(image, y1, y2, x1, x2)
            results.append(self.build_result(masked, y1, y2, x1, x2, "border", label))

        return results


    # =========================
    # CENTER
    # =========================

    def mask_center(self, image, fraction=0.45):

        _c, h_total, w_total = image.shape
        h = int(h_total * fraction)
        w = int(w_total * fraction)

        y1 = (h_total - h) // 2
        y2 = y1 + h
        x1 = (w_total - w) // 2
        x2 = x1 + w

        masked = image.clone()
        masked[:, y1:y2, x1:x2] = self.fill_region(image, y1, y2, x1, x2)

        return [self.build_result(masked, y1, y2, x1, x2, "center", "center-main")]


    # =========================
    # FILL REGION
    # =========================

    def fill_region(self, image, y1, y2, x1, x2):

        c = image.shape[0]
        h = y2 - y1
        w = x2 - x1

        if self.fill_mode == "constant":
            return torch.full((c, h, w), self.constant_value, device=image.device)

        mean = image.view(c, -1).mean(dim=1).view(c, 1, 1)

        if self.fill_mode == "mean":
            return mean.expand(c, h, w)

        if self.fill_mode == "noise":
            noise = torch.randn((c, h, w), device=image.device) * self.noise_std
            return (mean.expand(c, h, w) + noise).clamp(0.0, 1.0)

        raise ValueError("Invalid fill_mode")
