from pydantic import BaseModel


class BoundingBox(BaseModel):
    y1: int
    y2: int
    x1: int
    x2: int


class RegionScore(BaseModel):
    rank: int
    bbox: BoundingBox
    importance: float
    original_confidence: float
    masked_confidence: float
    mask_type: str
    label: str


class AnalyzeResponse(BaseModel):
    device: str
    class_index: int
    class_name: str
    confidence: float
    analysis_image: str
    heatmap_overlay_image: str
    box_overlay_image: str
    top_regions: list[RegionScore]
    all_regions: list[RegionScore]
