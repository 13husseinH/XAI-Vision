import torch


class ImageScorer:

    def score(self, model, original_image, masked_results):

        model.eval()

        with torch.no_grad():
            original_output = model(original_image.unsqueeze(0))
            original_prob = torch.softmax(original_output, dim=1)
            original_class = int(torch.argmax(original_prob, dim=1).item())
            original_conf = original_prob[0, original_class].item()

        scores = []

        for item in masked_results:

            img = item["image"]
            bbox = item["bbox"]

            with torch.no_grad():
                out = model(img.unsqueeze(0))
                prob = torch.softmax(out, dim=1)
                conf = prob[0, original_class].item()

            importance = original_conf - conf

            scores.append({
                "bbox": bbox,
                "importance": importance,
            })

        return scores
