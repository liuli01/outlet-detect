### Task 1: 项目脚手架与配置模块

**Files:**
- Create: `pyproject.toml`（由 `uv init` + `uv add` 生成）
- Create: `outlet_detect/__init__.py`
- Create: `outlet_detect/config.py`
- Create: `config/cameras/.gitkeep`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `CameraConfig` dataclass，字段：`camera_id: str`、`outfall_point: tuple[int, int]`、`reference_frame: str`、`hik_body: dict`、`score_low: float = 0.4`、`score_high: float = 0.7`、`vlm_base_url: str = ""`、`vlm_api_key: str = ""`、`vlm_model: str = ""`
- Produces: `load_camera_config(camera_id: str) -> CameraConfig`、`save_camera_config(cfg: CameraConfig) -> None`、`list_camera_ids() -> list[str]`

- [ ] **Step 1: 初始化 git 仓库与 uv 虚拟环境**

git 仓库已初始化（main 分支），跳过 `git init`。执行：

```bash
cd f:/Vaults/coding/outlet-detect
uv init --bare --python 3.12
uv add opencv-python numpy onnxruntime openai requests
uv add --dev pytest
```

（`uv init --bare` 只生成 `pyproject.toml`，不覆盖现有文件；依赖写进 pyproject 并自动创建 `.venv`）

`.gitignore` 内容：

```
__pycache__/
*.pyc
.venv/
data/
archive/
config/cameras/*.json
!config/cameras/.gitkeep
frames_preview/
videos/
```

- [ ] **Step 2: 写失败测试**

```python
# tests/test_config.py
import json
from pathlib import Path
from outlet_detect.config import CameraConfig, load_camera_config, save_camera_config

def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("outlet_detect.config.CONFIG_DIR", tmp_path)
    cfg = CameraConfig(
        camera_id="cam01",
        outfall_point=(640, 300),
        reference_frame="archive/cam01/ref.jpg",
        hik_body={"cameraIndexCode": "abc"},
    )
    save_camera_config(cfg)
    loaded = load_camera_config("cam01")
    assert loaded.camera_id == "cam01"
    assert loaded.outfall_point == (640, 300)
    assert loaded.score_low == 0.4
    assert loaded.score_high == 0.7

def test_list_camera_ids(tmp_path, monkeypatch):
    monkeypatch.setattr("outlet_detect.config.CONFIG_DIR", tmp_path)
    save_camera_config(CameraConfig(camera_id="a", outfall_point=(0, 0),
                                    reference_frame="", hik_body={}))
    save_camera_config(CameraConfig(camera_id="b", outfall_point=(0, 0),
                                    reference_frame="", hik_body={}))
    assert sorted(list_camera_ids()) == ["a", "b"]
```

- [ ] **Step 3: 运行测试确认失败**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'outlet_detect.config'`

- [ ] **Step 4: 实现配置模块**

```python
# outlet_detect/config.py
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

CONFIG_DIR = Path("config/cameras")

@dataclass
class CameraConfig:
    camera_id: str
    outfall_point: tuple  # (x, y) 排口在标定参照帧中的像素坐标
    reference_frame: str  # 标定参照帧图片路径
    hik_body: dict        # 海康 API 请求体（含 cameraIndexCode）
    score_low: float = 0.4
    score_high: float = 0.7
    vlm_base_url: str = ""
    vlm_api_key: str = ""
    vlm_model: str = ""

def _path(camera_id: str) -> Path:
    return CONFIG_DIR / f"{camera_id}.json"

def save_camera_config(cfg: CameraConfig) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = asdict(cfg)
    data["outfall_point"] = list(cfg.outfall_point)
    _path(cfg.camera_id).write_text(json.dumps(data, ensure_ascii=False, indent=2))

def load_camera_config(camera_id: str) -> CameraConfig:
    data = json.loads(_path(camera_id).read_text())
    data["outfall_point"] = tuple(data["outfall_point"])
    return CameraConfig(**data)

def list_camera_ids() -> list:
    if not CONFIG_DIR.exists():
        return []
    return [p.stem for p in CONFIG_DIR.glob("*.json")]
```

```python
# outlet_detect/__init__.py
```

（空文件）

- [ ] **Step 5: 运行测试确认通过并提交**

Run: `uv run pytest tests/test_config.py -v`
Expected: 2 passed

```bash
git add -A && git commit -m "chore: 项目脚手架与摄像头配置模块"
```
