from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps


class ImportanceVisualizer:

    def _ranked_scores(self, scores):
        return sorted(scores, key=lambda item: item["importance"], reverse=True)

    def _positive_scores(self, scores):
        return [item for item in self._ranked_scores(scores) if item["importance"] > 0]

    def render_box_overlay(self, image, scores, topk=5):
        base = image.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        selected = self._positive_scores(scores)[:topk]
        if not selected:
            return base.convert("RGB")

        max_importance = max(item["importance"] for item in selected)
        scale = max(max_importance, 1e-8)

        for index, item in enumerate(selected, start=1):
            y1, y2, x1, x2 = item["bbox"]
            importance = item["importance"]
            strength = max(0.0, min(1.0, importance / scale))
            fill_alpha = int(45 + 95 * strength)
            outline_alpha = int(160 + 70 * strength)

            draw.rectangle(
                (x1, y1, x2, y2),
                fill=(220, 55, 35, fill_alpha),
                outline=(255, 255, 255, outline_alpha),
                width=3,
            )

            label = f"#{index} {importance:.3f}"
            draw.text((x1 + 6, y1 + 6), label, fill=(255, 255, 255, 255))

        return Image.alpha_composite(base, overlay).convert("RGB")

    def render_heatmap_overlay(self, image, scores):
        base = image.convert("RGBA")
        positive = self._positive_scores(scores)
        if not positive:
            return base.convert("RGB")

        width, height = base.size
        intensity = Image.new("L", (width, height), 0)

        max_importance = max(item["importance"] for item in positive)
        scale = max(max_importance, 1e-8)

        for item in positive:
            y1, y2, x1, x2 = item["bbox"]
            strength = max(0.0, min(1.0, item["importance"] / scale))
            layer = Image.new("L", (width, height), 0)
            layer_draw = ImageDraw.Draw(layer)
            layer_value = int(20 + 95 * strength)
            layer_draw.rectangle((x1, y1, x2, y2), fill=layer_value)
            layer = layer.filter(ImageFilter.GaussianBlur(radius=10))
            intensity = ImageChops.add(intensity, layer)

        intensity = intensity.filter(ImageFilter.GaussianBlur(radius=20))
        intensity = ImageOps.autocontrast(intensity)
        alpha = intensity.point(lambda value: int(value * 0.78))

        heat = Image.new("RGBA", (width, height), (230, 62, 32, 0))
        heat.putalpha(alpha)
        return Image.alpha_composite(base, heat).convert("RGB")
