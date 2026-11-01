"""FastAPI 测试台：上传照片测试漂浮物/排污检测。

启动：uv run python -m outlet_detect.webapp  →  http://127.0.0.1:8090
依赖 8005 推理服务（水面分割 + 漂浮物引擎）。
排污说明：单张照片复制为 5 连帧（静止序列，波动特征=0），
只反映黑度/色偏——波动特征需在线连续帧才有意义。
"""
import base64
import html

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse

from outlet_detect.float_detector import HttpFloatDetector, filter_to_water
from outlet_detect.reporter import annotate_discharge, annotate_float
from outlet_detect.rois import build_rois
from outlet_detect.rule_engine import compute_features, score, verdict
from outlet_detect.water_seg import HttpWaterSegmenter

DEFAULT_FLOAT_ENGINE = "floating/20261030_145216"

app = FastAPI(title="outlet-detect 测试台")

PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>检测测试台</title>
<style>
body{font-family:sans-serif;max-width:960px;margin:2em auto;color:#222}
label{display:inline-block;min-width:7em}
fieldset{margin:1em 0;padding:1em;border:1px solid #ccc;border-radius:6px}
img{max-width:100%;border:1px solid #999}
table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 10px}
</style></head>
<body>
<h2>漂浮物 / 排污检测测试台</h2>
<form method="post" action="/test" enctype="multipart/form-data">
  <fieldset><legend>照片</legend>
    <input type="file" name="image" accept="image/*" required>
  </fieldset>
  <fieldset><legend>排污检测</legend>
    <label>排口点 x:</label><input name="outfall_x" type="number" value="0" size="6">
    <label>y:</label><input name="outfall_y" type="number" value="0" size="6">
    <small>（0,0 = 画面中心）</small>
  </fieldset>
  <fieldset><legend>漂浮物检测</legend>
    <label>引擎:</label><input name="float_engine" value="__ENGINE__" size="32">
    <label>置信度:</label><input name="float_conf" value="0.3" size="5">
    <small>（引擎留空 = 关闭漂浮物检测）</small>
  </fieldset>
  <p><button type="submit" style="font-size:1.1em;padding:.4em 2em">检测</button></p>
</form>
</body></html>"""


def make_segmenter():
    return HttpWaterSegmenter()


def make_detector(engine_id, confidence):
    return HttpFloatDetector(engine_id=engine_id, confidence=confidence)


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE.replace("__ENGINE__", DEFAULT_FLOAT_ENGINE)


def _b64img(img):
    return base64.b64encode(cv2.imencode(".jpg", img)[1]).decode()


@app.post("/test", response_class=HTMLResponse)
async def test(image: UploadFile = File(...), outfall_x: int = Form(0),
               outfall_y: int = Form(0),
               float_engine: str = Form(DEFAULT_FLOAT_ENGINE),
               float_conf: float = Form(0.3)):
    data = await image.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return HTMLResponse("图片解码失败，请上传 JPG/PNG", status_code=400)
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

    # 3) 排污：单张照片复制 5 帧（静止序列，波动=0）
    point = (outfall_x, outfall_y) if (outfall_x or outfall_y) else (w // 2, h // 2)
    rois = build_rois(mask, point)
    feats = compute_features([img] * 5, rois, None)
    s = score(feats)
    v = verdict(s)

    # 4) 标注图：水面绿色叠加 + ROI 三色圈 + 漂浮物红框
    ann = annotate_discharge(img, rois, s)
    ann = annotate_float(ann, dets_all)
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
img{{max-width:100%;border:1px solid #999}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 10px}}</style></head>
<body>
<h2>检测结果</h2>
<p><a href="/">← 再测一张</a></p>
<h3>水面分割</h3><p>水面覆盖率 {water_pct:.1f}%</p>
<h3>漂浮物（水面范围内 {len(dets)} / 总检出 {len(dets_all)}）</h3>
{float_block}
<h3>排污（静止序列，波动特征=0）</h3>
<table>
<tr><th>综合评分</th><td>{s:.3f}</td></tr>
<tr><th>判定</th><td><b>{verdict_cn}</b></td></tr>
<tr><th>相对黑度</th><td>{feats['rel_darkness']:.2f}</td></tr>
<tr><th>波动能量比</th><td>{feats['flow_energy']:.2f}</td></tr>
<tr><th>颜色偏离</th><td>{feats['color_dev']:.2f}</td></tr>
</table>
<h3>标注图（红=排口区 黄=扩散区 绿圈=参考区 绿色蒙版=水面 红框=漂浮物）</h3>
<img src="data:image/jpeg;base64,{_b64img(ann)}">
</body></html>"""


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8090)


if __name__ == "__main__":
    main()
