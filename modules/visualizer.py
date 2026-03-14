from PIL import Image, ImageDraw


class ImportanceVisualizer:

    def render_overlay(self, image, scores, topk=5):
        base = image.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        ranked = sorted(scores, key=lambda item: item["importance"], reverse=True)
        selected = ranked[:topk]

        if not selected:
            return base.convert("RGB")

        max_importance = max(item["importance"] for item in selected)
        scale = max(max_importance, 1e-8)

        for index, item in enumerate(selected, start=1):
            y1, y2, x1, x2 = item["bbox"]
            importance = item["importance"]
            strength = max(0.0, min(1.0, importance / scale))
            fill_alpha = int(70 + 110 * strength)
            outline_alpha = int(150 + 80 * strength)

            draw.rectangle(
                (x1, y1, x2, y2),
                fill=(220, 55, 35, fill_alpha),
                outline=(255, 255, 255, outline_alpha),
                width=3,
            )

            label = f"#{index} {importance:.3f}"
            text_x = x1 + 6
            text_y = y1 + 6
            draw.text((text_x, text_y), label, fill=(255, 255, 255, 255))

        return Image.alpha_composite(base, overlay).convert("RGB")
