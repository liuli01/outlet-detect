import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

CONFIG_DIR = Path("config/cameras")

@dataclass
class CameraConfig:
    camera_id: str
    outfall_point: tuple | None = None  # (x, y) 排口坐标；None/缺省=未标定，运行时用水面质心
    reference_frame: str = ""           # 标定参照帧图片路径
    hik_body: dict = field(default_factory=dict)  # 海康 API 请求体（含 cameraIndexCode）
    score_low: float = 0.4
    score_high: float = 0.7
    float_engine_id: str = ""       # 漂浮物引擎；空=关闭漂浮物检测
    float_confidence: float = 0.6
    ignore_classes: list = field(default_factory=list)  # 如 ["aeration"]

def _path(camera_id: str) -> Path:
    return CONFIG_DIR / f"{camera_id}.json"

def save_camera_config(cfg: CameraConfig) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = asdict(cfg)
    data["outfall_point"] = list(cfg.outfall_point) if cfg.outfall_point else None
    _path(cfg.camera_id).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def load_camera_config(camera_id: str) -> CameraConfig:
    data = json.loads(_path(camera_id).read_text(encoding="utf-8"))
    pt = data.get("outfall_point")
    data["outfall_point"] = tuple(pt) if pt else None  # 缺省/null → None（水面质心兜底）
    return CameraConfig(**data)

def list_camera_ids() -> list:
    if not CONFIG_DIR.exists():
        return []
    return [p.stem for p in CONFIG_DIR.glob("*.json")]
