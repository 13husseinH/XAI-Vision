import torch


class ImageScorer:

    def score(self, model, original_image, masked_results):

        model.eval()

        with torch.no_grad():
            original_output = model(original_image.unsqueeze(0))
            original_prob = torch.softmax(original_output, dim=1)
            original_class = int(torch.argmax(original_prob, dim=1).item())
            original_conf = float(original_prob[0, original_class].item())

        scores = []

        for item in masked_results:

            img = item["image"]
            bbox = item["bbox"]
            mask_type = item.get("mask_type", "unknown")
            label = item.get("label", mask_type)

            with torch.no_grad():
                out = model(img.unsqueeze(0))
                prob = torch.softmax(out, dim=1)
                masked_conf = float(prob[0, original_class].item())

            importance = max(0.0, original_conf - masked_conf)

            scores.append({
                "bbox": bbox,
                "importance": importance,
                "original_confidence": original_conf,
                "masked_confidence": masked_conf,
                "mask_type": mask_type,
                "label": label,
                "target_class": original_class,
            })

        return scores
