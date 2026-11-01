"""FastAPI 测试台：上传照片或点选 images/ 样本库，测试漂浮物/排污检测。

启动：uv run python -m outlet_detect.webapp  →  http://127.0.0.1:8090
依赖 8005 推理服务（水面分割 + 漂浮物引擎）。
样本库（本地 images/ 目录，不入库）：
  images/排污/<组名>/   —— 5 连帧组，波动特征真实（排口点取 OUTFALL_POINTS 标注）
  images/无排污/*.jpg   —— 单张照片复制 5 帧（静止序列，波动=0）
  images/有漂浮物|无漂浮物/*.jpg —— 单张，测漂浮物
"""
import base64
import html
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import HTMLResponse, Response

from outlet_detect.float_detector import HttpFloatDetector, filter_to_water
from outlet_detect.reporter import annotate_discharge, annotate_float
from outlet_detect.rois import build_rois
from outlet_detect.rule_engine import compute_features, score, verdict
from outlet_detect.water_seg import HttpWaterSegmenter

DEFAULT_FLOAT_ENGINE = "floating/20261030_145216"
IMAGES_ROOT = Path("images")

# 排污 5 连帧组的人工标注排口点（与 validate_images.py 一致）
OUTFALL_POINTS = {
    "01521d22e3e199b46ac42aebcc68ab79": (730, 200),
    "255407597f6d29b497ec7028542a90b4": (860, 360),
    "bba6cc73a8fda9ed46db7555acc95304": (550, 150),
    "东城河阳光桥球机_1687161142095": (400, 500),
    "创新河春申路河道球机_1687161684788": (1780, 750),
    "创新河春申路河道球机_1687161745744": (200, 500),
    "普惠中心河五星路球机_1687161282421": (1280, 720),
    "芦花沟河天庆路球机_1687160641609": (1750, 620),
    "芦花沟河天庆路球机_1687160680573": (550, 620),
}

app = FastAPI(title="outlet-detect 测试台")

PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>检测测试台</title>
<style>
body{font-family:sans-serif;max-width:1100px;margin:2em auto;color:#222}
fieldset{margin:1em 0;padding:1em;border:1px solid #ccc;border-radius:6px}
img.out{max-width:100%;border:1px solid #999}
table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 10px}
.gallery{display:flex;flex-wrap:wrap;gap:10px}
.sample{text-align:center;font-size:12px}
.sample img{width:180px;border:1px solid #999;cursor:pointer}
.sample button{margin-top:2px}
h4{margin:.6em 0 .2em}
</style></head>
<body>
<h2>漂浮物 / 排污检测测试台</h2>
<form method="post" action="/test" enctype="multipart/form-data">
  <fieldset><legend>上传照片</legend>
    <input type="file" name="image" accept="image/*">
    <small>（与下方样本库二选一；都不选则报错）</small>
  </fieldset>
  <fieldset><legend>参数</legend>
    排口点 x:<input name="outfall_x" type="number" value="0" size="6">
    y:<input name="outfall_y" type="number" value="0" size="6">
    <small>（0,0 = 样本标注点，无标注则画面中心）</small><br>
    漂浮物引擎:<input name="float_engine" value="__ENGINE__" size="32">
    置信度:<input name="float_conf" value="0.3" size="5">
    <small>（引擎留空 = 关闭漂浮物检测）</small>
    <input type="hidden" name="sample_path" value="">
  </fieldset>
  <p><button type="submit" style="font-size:1.1em;padding:.4em 2em">检测上传照片</button></p>
</form>
__GALLERY__
<script>
function pick(p){var f=document.forms[0];f.sample_path.value=p;
  f.querySelector('input[type=file]').value='';f.submit();}
</script>
</body></html>"""


def make_segmenter():
    return HttpWaterSegmenter()


def make_detector(engine_id, confidence):
    return HttpFloatDetector(engine_id=engine_id, confidence=confidence)


def _imread(p: Path):
    """cv2.imread 不支持中文路径。"""
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def list_samples():
    """扫描样本库：[{label, path, thumb, group}]，path 为相对 images/ 的路径。"""
    out = []
    if not IMAGES_ROOT.is_dir():
        return out
    for group_dir in sorted((IMAGES_ROOT / "排污").glob("*")):
        if not group_dir.is_dir() or group_dir.name == "训练抽取":
            continue
        frames = sorted(group_dir.glob("*.jpg"))
        if frames:
            out.append({"label": f"排污·{group_dir.name[:20]}", "group": "排污（5连帧）",
                        "path": group_dir.relative_to(IMAGES_ROOT).as_posix(),
                        "thumb": frames[0].relative_to(IMAGES_ROOT).as_posix()})
    for sub in ("无排污", "有漂浮物", "无漂浮物"):
        for f in sorted((IMAGES_ROOT / sub).glob("*.jpg")):
            out.append({"label": f"{sub}·{f.stem[:16]}", "group": sub,
                        "path": f.relative_to(IMAGES_ROOT).as_posix(),
                        "thumb": f.relative_to(IMAGES_ROOT).as_posix()})
    return out


def _safe_sample_path(rel: str) -> Path:
    """限制样本路径在 images/ 内（防目录穿越）。"""
    p = (IMAGES_ROOT / rel).resolve()
    if not p.is_relative_to(IMAGES_ROOT.resolve()) or not p.exists():
        raise ValueError(f"非法样本路径: {rel}")
    return p


def load_sample_frames(rel: str):
    """返回 (frames, 默认排口点或None, 描述)。"""
    p = _safe_sample_path(rel)
    if p.is_dir():
        frames = [_imread(f) for f in sorted(p.glob("*.jpg"))[:5]]
        frames = [f for f in frames if f is not None]
        return frames, OUTFALL_POINTS.get(p.name), f"样本组 {p.name}（{len(frames)} 帧）"
    img = _imread(p)
    if img is None:
        raise ValueError(f"样本读不了: {rel}")
    return [img] * 5, None, f"单张样本 {p.name}（复制5帧，波动=0）"


@app.get("/", response_class=HTMLResponse)
def index():
    samples = list_samples()
    if samples:
        groups = {}
        for s in samples:
            groups.setdefault(s["group"], []).append(s)
        parts = []
        for g, items in groups.items():
            cards = "".join(
                f"<div class='sample'><img src='/sample_img?path="
                f"{html.escape(s['thumb'])}' onclick=\"pick('{html.escape(s['path'])}')\">"
                f"<br>{html.escape(s['label'])}"
                f"<br><button onclick=\"pick('{html.escape(s['path'])}')\">测这个</button></div>"
                for s in items)
            parts.append(f"<h4>{g}</h4><div class='gallery'>{cards}</div>")
        gallery = "<fieldset><legend>样本库（点击照片直接测）</legend>" + "".join(parts) + "</fieldset>"
    else:
        gallery = ("<fieldset><legend>样本库</legend>"
                   "<p>未找到 images/ 目录（本地测试数据，不入库）</p></fieldset>")
    return PAGE.replace("__ENGINE__", DEFAULT_FLOAT_ENGINE).replace("__GALLERY__", gallery)


@app.get("/sample_img")
def sample_img(path: str = Query(...)):
    try:
        p = _safe_sample_path(path)
    except ValueError:
        return Response(status_code=400)
    if not p.is_file():
        return Response(status_code=404)
    return Response(content=p.read_bytes(), media_type="image/jpeg")


def _b64img(img):
    return base64.b64encode(cv2.imencode(".jpg", img)[1]).decode()


@app.post("/test", response_class=HTMLResponse)
async def test(image: UploadFile | None = File(None), sample_path: str = Form(""),
               outfall_x: int = Form(0), outfall_y: int = Form(0),
               float_engine: str = Form(DEFAULT_FLOAT_ENGINE),
               float_conf: float = Form(0.3)):
    # 帧来源：上传文件 或 样本库
    src_desc = "上传照片（复制5帧，波动=0）"
    if sample_path.strip():
        try:
            frames, marked_point, src_desc = load_sample_frames(sample_path.strip())
        except ValueError as e:
            return HTMLResponse(str(e), status_code=400)
    else:
        data = await image.read() if image is not None else b""
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
        if img is None:
            return HTMLResponse("请上传照片或选择样本", status_code=400)
        frames, marked_point = [img] * 5, None
    if not frames:
        return HTMLResponse("样本帧为空", status_code=400)
    img = frames[0]
    h, w = img.shape[:2]

    # 1) 水面分割
    try:
        mask = make_segmenter().segment(img)
    except Exception as e:
        return HTMLResponse(f"水面分割服务异常（8005 是否在线？）："
                            f"{html.escape(str(e))}", status_code=502)
    water_pct = mask.sum() / 255 / mask.size * 100

    # 2) 漂浮物：只计水面范围内检出
    dets_all, dets, float_err = [], [], ""
    if float_engine.strip():
        try:
            dets_all = make_detector(float_engine.strip(), float_conf).detect(img)
            dets = filter_to_water(dets_all, mask)
        except Exception as e:
            float_err = str(e)

    # 3) 排污：优先 页面坐标 > 样本标注点 > 画面中心
    if outfall_x or outfall_y:
        point = (outfall_x, outfall_y)
    elif marked_point:
        point = marked_point
    else:
        point = (w // 2, h // 2)
    rois = build_rois(mask, point)
    feats = compute_features(frames, rois, None)
    s = score(feats)
    v = verdict(s)

    # 4) 标注图：水面绿色叠加 + ROI 三色圈 + 排口点 + 漂浮物红框
    ann = annotate_discharge(img, rois, s)
    ann = annotate_float(ann, dets_all)
    cv2.drawMarker(ann, point, (255, 0, 255), cv2.MARKER_CROSS, 30, 2)
    ann[mask > 0] = (ann[mask > 0] * 0.7 + np.array([0, 255, 0]) * 0.3).astype(np.uint8)

    kept = {id(d) for d in dets}
    rows = "".join(
        f"<tr><td>{'✔ 保留' if id(d) in kept else '✘ 岸上过滤'}</td>"
        f"<td>{html.escape(d.class_name)}</td><td>{d.confidence:.2f}</td>"
        f"<td>{tuple(int(x) for x in d.bbox)}</td></tr>" for d in dets_all)
    float_rows = rows or "<tr><td colspan=4>无检出</td></tr>"
    float_block = (f"<p style='color:red'>漂浮物检测异常：{html.escape(float_err)}</p>"
                   if float_err else
                   f"<table><tr><th>水面过滤</th><th>类别</th><th>置信度</th>"
                   f"<th>bbox</th></tr>{float_rows}</table>")
    verdict_cn = {"normal": "正常", "gray": "灰色（线上送 CNN/VLM 终判）",
                  "suspect": "疑似排污"}.get(v, v)
    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>检测结果</title>
<style>body{{font-family:sans-serif;max-width:960px;margin:2em auto}}
img.out{{max-width:100%;border:1px solid #999}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 10px}}</style></head>
<body>
<h2>检测结果</h2>
<p><a href="/">← 返回测试台</a></p>
<p>帧来源：{html.escape(src_desc)}；排口点 {point}</p>
<h3>水面分割</h3><p>水面覆盖率 {water_pct:.1f}%</p>
<h3>漂浮物（水面范围内 {len(dets)} / 总检出 {len(dets_all)}）</h3>
{float_block}
<h3>排污</h3>
<table>
<tr><th>综合评分</th><td>{s:.3f}</td></tr>
<tr><th>判定</th><td><b>{verdict_cn}</b></td></tr>
<tr><th>相对黑度</th><td>{feats['rel_darkness']:.2f}</td></tr>
<tr><th>波动能量比</th><td>{feats['flow_energy']:.2f}</td></tr>
<tr><th>颜色偏离</th><td>{feats['color_dev']:.2f}</td></tr>
</table>
<h3>标注图（红=排口区 黄=扩散区 绿圈=参考区 绿色蒙版=水面 红框=漂浮物 紫叉=排口点）</h3>
<img class="out" src="data:image/jpeg;base64,{_b64img(ann)}">
</body></html>"""


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8090)


if __name__ == "__main__":
    main()
