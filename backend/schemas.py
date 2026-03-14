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


class AnalyzeResponse(BaseModel):
    device: str
    class_index: int
    class_name: str
    confidence: float
    top_regions: list[RegionScore]
    analysis_image: str
    overlay_image: str
