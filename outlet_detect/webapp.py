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
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>检测测试台</title>
<style>
:root{--bg:#f3f5f9;--card:#fff;--line:#e4e8f0;--txt:#1f2430;--sub:#6b7280;
--accent:#2563eb;--accent-d:#1d4ed8;--radius:12px;
--shadow:0 1px 3px rgba(16,24,40,.08),0 4px 14px rgba(16,24,40,.05)}
*{box-sizing:border-box}
body{font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif;
background:var(--bg);color:var(--txt);margin:0;padding:32px 16px 64px}
.wrap{max-width:1600px;margin:0 auto}
header.top{margin-bottom:22px}
h1{font-size:22px;margin:0 0 4px}
.tagline{color:var(--sub);font-size:13px}
.layout{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px;align-items:start}
.col-side{position:sticky;top:24px}
@media (max-width:1200px){.layout{grid-template-columns:1fr}.col-side{position:static}}
.card{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
box-shadow:var(--shadow);padding:20px 22px;margin-bottom:18px}
.card>h3{margin:0 0 14px;font-size:15px;display:flex;align-items:center;gap:8px}
.card>h3::before{content:"";width:4px;height:16px;border-radius:2px;background:var(--accent)}
.hint{color:var(--sub);font-size:12px}
.meta{color:var(--sub);font-size:13px;margin-bottom:14px}
.params{display:flex;flex-wrap:wrap;gap:16px 28px}
.field{display:flex;flex-direction:column;gap:5px}
.field label{font-size:12px;color:var(--sub)}
.field .row{display:flex;align-items:center;gap:6px}
input[type=number],input[type=text]{padding:7px 10px;border:1px solid var(--line);
border-radius:8px;font-size:14px;background:#fafbfd;outline:none;transition:border-color .15s}
input:focus{border-color:var(--accent);background:#fff}
.filebox{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.filebtn{display:inline-block;padding:8px 18px;border:1px dashed #b6c2d9;border-radius:8px;
color:var(--accent);cursor:pointer;font-size:14px;transition:all .15s;background:#f8faff}
.filebtn:hover{border-color:var(--accent);background:#eef4ff}
input[type=file]{display:none}
#fname{font-size:13px;color:var(--sub)}
.submit{margin-top:4px}
.submit button{background:var(--accent);color:#fff;border:none;border-radius:9px;
padding:11px 38px;font-size:15px;cursor:pointer;box-shadow:0 2px 6px rgba(37,99,235,.3);
transition:background .15s}
.submit button:hover{background:var(--accent-d)}
.submit button:disabled{background:#93b4f5;cursor:wait}
.gsec{margin:14px 0 6px;font-size:13px;font-weight:600;display:flex;align-items:center;gap:7px}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block}
.g-pw .dot,.badge.g-pw{background:#dc2626}.g-nw .dot,.badge.g-nw{background:#16a34a}
.g-fl .dot,.badge.g-fl{background:#d97706}.g-nofl .dot,.badge.g-nofl{background:#2563eb}
.gsec .cnt{color:var(--sub);font-weight:400}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px;margin-bottom:8px}
.sample{background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden;
cursor:pointer;transition:transform .15s,box-shadow .15s}
.sample:hover{transform:translateY(-3px);box-shadow:0 6px 16px rgba(16,24,40,.13)}
.thumb{position:relative;height:100px;background:#0b0f16}
.thumb img{width:100%;height:100%;object-fit:cover;display:block}
.badge{position:absolute;top:6px;left:6px;color:#fff;font-size:11px;
padding:2px 8px;border-radius:20px;opacity:.92}
.s-label{padding:8px 10px;font-size:12px;color:var(--txt);line-height:1.4;
white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
/* ===== 结果面板（JS 注入的片段也用这个样式） ===== */
.stats{display:flex;gap:10px;flex-wrap:wrap}
.stat{flex:1;min-width:100px;background:#f8fafd;border:1px solid var(--line);
border-radius:10px;padding:10px 14px}
.stat .k{font-size:12px;color:var(--sub);margin-bottom:4px}
.stat .v{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:4px 14px;border-radius:20px;font-size:13px;font-weight:600}
.pill.ok{background:#dcfce7;color:#15803d}
.pill.warn{background:#fef3c7;color:#b45309}
.pill.bad{background:#fee2e2;color:#b91c1c}
table{border-collapse:collapse;width:100%;font-size:13px}
td,th{border-bottom:1px solid var(--line);padding:8px 10px;text-align:left}
th{color:var(--sub);font-weight:600;font-size:12px;background:#f8fafd}
tr.keep td:first-child{color:#15803d}
tr.drop td:first-child{color:#b91c1c}
td.empty{color:var(--sub);text-align:center}
.errbox{color:#b91c1c;background:#fee2e2;border-radius:8px;padding:10px 14px;font-size:13px;margin:0}
img.out{max-width:100%;border-radius:8px;border:1px solid var(--line);display:block;
cursor:zoom-in}
#zoom{position:fixed;inset:0;background:rgba(8,12,20,.88);display:none;
align-items:center;justify-content:center;z-index:99;cursor:zoom-out;padding:24px}
#zoom.on{display:flex}
#zoom img{max-width:96vw;max-height:94vh;border-radius:8px;
box-shadow:0 8px 40px rgba(0,0,0,.5)}
.legend{display:flex;flex-wrap:wrap;gap:10px 14px;font-size:12px;color:var(--sub);margin-bottom:12px}
.lg{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;
vertical-align:-1px}
.swapbtn{background:rgba(255,255,255,.92);border:1px solid var(--line);border-radius:8px;
color:var(--accent);padding:5px 14px;font-size:12px;cursor:pointer;
box-shadow:0 1px 4px rgba(0,0,0,.25);transition:all .15s}
.swapbtn:hover{border-color:var(--accent);background:#fff}
.imgbox{position:relative}
.imgbox .swapbtn{position:absolute;top:10px;right:10px}
.spin{display:inline-block;width:14px;height:14px;border:2px solid #c7d6f5;
border-top-color:var(--accent);border-radius:50%;animation:rot .8s linear infinite;
vertical-align:-2px;margin-right:8px}
@keyframes rot{to{transform:rotate(360deg)}}
/* 结果面板紧凑化：少占高度，多留空间给标注图 */
#result .card{padding:12px 16px;margin-bottom:12px}
#result .card>h3{margin-bottom:8px}
#result .meta{margin-bottom:10px}
#result .stats{gap:8px}
#result .stat{padding:6px 10px;min-width:84px}
#result .stat .k{font-size:11px;margin-bottom:2px}
#result .stat .v{font-size:16px}
#result .pill{font-size:12px;padding:3px 10px}
#result td,#result th{padding:5px 8px}
#result .legend{margin-bottom:8px}
#result img.out{margin-top:2px}
</style></head>
<body><div class="wrap">
<header class="top">
  <h1>漂浮物 / 排污检测测试台</h1>
  <div class="tagline">上传照片或点选样本库，走与线上一致的分割 → 漂浮物 → 排污评分链路，结果在右侧实时显示</div>
</header>
<div class="layout">
<div class="col-main">
<form id="tform" enctype="multipart/form-data">
  <div class="card"><h3>上传照片</h3>
    <div class="filebox">
      <label class="filebtn">选择图片…<input type="file" name="image" accept="image/*"
        onchange="document.getElementById('fname').textContent=this.files[0]?this.files[0].name:''"></label>
      <span id="fname"></span>
      <span class="hint">与下方样本库二选一；单张照片复制 5 帧（波动=0）</span>
    </div>
  </div>
  <div class="card"><h3>参数</h3>
    <div class="params">
      <div class="field"><label>排口点（0,0 = 样本标注点 / 画面中心）</label>
        <div class="row">x <input name="outfall_x" type="number" value="0" style="width:90px">
        y <input name="outfall_y" type="number" value="0" style="width:90px"></div></div>
      <div class="field"><label>漂浮物引擎（留空 = 关闭）</label>
        <input name="float_engine" type="text" value="__ENGINE__" style="width:250px"></div>
      <div class="field"><label>漂浮物置信度</label>
        <input name="float_conf" type="text" value="0.3" style="width:80px"></div>
    </div>
    <input type="hidden" name="sample_path" value="">
    <p class="submit"><button type="submit" id="go">开始检测</button></p>
  </div>
</form>
__GALLERY__
</div>
<div class="col-side"><div id="result">
  <div class="card"><h3>检测结果</h3>
    <p class="hint">上传照片或点击左侧样本，结果将实时显示在这里，页面不跳转。</p>
  </div>
</div></div>
</div>
<script>
var RES=document.getElementById('result'),GO=document.getElementById('go');
async function submitForm(){
  var f=document.getElementById('tform');
  GO.disabled=true;
  RES.innerHTML='<div class="card"><h3>检测中…</h3><p class="hint"><span class="spin"></span>正在调用分割 / 检测服务，通常几秒钟</p></div>';
  if(window.innerWidth<=1200)RES.scrollIntoView({behavior:'smooth'});
  try{
    var r=await fetch('/test',{method:'POST',body:new FormData(f)});
    RES.innerHTML=await r.text();
  }catch(e){
    RES.innerHTML='<div class="card"><p class="errbox">请求失败：'+e+'</p></div>';
  }
  GO.disabled=false;
}
document.getElementById('tform').addEventListener('submit',function(e){e.preventDefault();submitForm();});
function pick(p){var f=document.getElementById('tform');f.sample_path.value=p;
  f.querySelector('input[type=file]').value='';
  document.getElementById('fname').textContent='';submitForm();}
/* 标注图点击放大：事件委托（结果图是后注入的；原图切换用片段内联onclick，不依赖本页JS） */
var Z=document.getElementById('zoom'),ZI=Z.querySelector('img');
document.addEventListener('click',function(e){
  if(e.target.matches('img.out')){ZI.src=e.target.src;Z.classList.add('on');}
});
Z.addEventListener('click',function(){Z.classList.remove('on');});
document.addEventListener('keydown',function(e){if(e.key==='Escape')Z.classList.remove('on');});
</script>
<div id="zoom"><img alt="放大查看"></div>
</div></body></html>"""


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


GROUP_CLASS = {"排污（5连帧）": "g-pw", "无排污": "g-nw",
               "有漂浮物": "g-fl", "无漂浮物": "g-nofl"}


@app.get("/", response_class=HTMLResponse)
def index():
    samples = list_samples()
    if samples:
        groups = {}
        for s in samples:
            groups.setdefault(s["group"], []).append(s)
        parts = []
        for g, items in groups.items():
            cls = GROUP_CLASS.get(g, "g-nofl")
            cards = "".join(
                f"<div class='sample' onclick=\"pick('{html.escape(s['path'])}')\">"
                f"<div class='thumb'><img loading='lazy' src='/sample_img?path="
                f"{html.escape(s['thumb'])}'><span class='badge {cls}'>{html.escape(g)}</span></div>"
                f"<div class='s-label'>{html.escape(s['label'])}</div></div>"
                for s in items)
            parts.append(f"<div class='gsec {cls}'><span class='dot'></span>{g}"
                         f"<span class='cnt'>（{len(items)}）</span></div>"
                         f"<div class='gallery'>{cards}</div>")
        gallery = ("<div class='card'><h3>样本库（点击照片直接测）</h3>"
                   + "".join(parts) + "</div>")
    else:
        gallery = ("<div class='card'><h3>样本库</h3>"
                   "<p class='hint'>未找到 images/ 目录（本地测试数据，不入库）</p></div>")
    return HTMLResponse(
        PAGE.replace("__ENGINE__", DEFAULT_FLOAT_ENGINE).replace("__GALLERY__", gallery),
        headers={"Cache-Control": "no-store"})


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


def _errbox(msg: str) -> str:
    """错误片段：注入右侧结果面板（样式在主页）。"""
    return f"<div class='card'><h3>检测失败</h3><p class='errbox'>{msg}</p></div>"


@app.post("/test", response_class=HTMLResponse)
async def test(image: UploadFile | None = File(None), sample_path: str = Form(""),
               outfall_x: int = Form(0), outfall_y: int = Form(0),
               float_engine: str = Form(""),  # 主页输入框已预填默认引擎；留空=关闭
               float_conf: float = Form(0.3)):
    # 帧来源：上传文件 或 样本库
    src_desc = "上传照片（复制5帧，波动=0）"
    if sample_path.strip():
        try:
            frames, marked_point, src_desc = load_sample_frames(sample_path.strip())
        except ValueError as e:
            return HTMLResponse(_errbox(html.escape(str(e))), status_code=400)
    else:
        data = await image.read() if image is not None else b""
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
        if img is None:
            return HTMLResponse(_errbox("请上传照片或选择样本"), status_code=400)
        frames, marked_point = [img] * 5, None
    if not frames:
        return HTMLResponse(_errbox("样本帧为空"), status_code=400)
    img = frames[0]
    h, w = img.shape[:2]

    # 1) 水面分割
    try:
        mask = make_segmenter().segment(img)
    except Exception as e:
        return HTMLResponse(_errbox(f"水面分割服务异常（8005 是否在线？）："
                                    f"{html.escape(str(e))}"), status_code=502)
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
        f"<tr class='{'keep' if id(d) in kept else 'drop'}'>"
        f"<td>{'✔ 保留' if id(d) in kept else '✘ 岸上过滤'}</td>"
        f"<td>{html.escape(d.class_name)}</td><td>{d.confidence:.2f}</td>"
        f"<td>{tuple(int(x) for x in d.bbox)}</td></tr>" for d in dets_all)
    float_rows = rows or "<tr><td colspan=4 class='empty'>无检出</td></tr>"
    float_block = (f"<p class='errbox'>漂浮物检测异常：{html.escape(float_err)}</p>"
                   if float_err else
                   f"<table><tr><th>水面过滤</th><th>类别</th><th>置信度</th>"
                   f"<th>bbox</th></tr>{float_rows}</table>")
    verdict_cn = {"normal": "正常", "gray": "灰色（线上送 CNN/VLM 终判）",
                  "suspect": "疑似排污"}.get(v, v)
    vcls = {"normal": "ok", "gray": "warn", "suspect": "bad"}.get(v, "ok")
    # 漂浮物判定：与线上一致，只计水面范围内检出（线上还需状态机连续确认才上报）
    if not float_engine.strip():
        float_cn, fcls = "未启用", "ok"
    elif float_err:
        float_cn, fcls = "检测异常", "warn"
    elif dets:
        float_cn = f"有漂浮物（最高 {max(d.confidence for d in dets):.2f}）"
        fcls = "bad"
    else:
        float_cn, fcls = "无漂浮物", "ok"
    # 结果为 HTML 片段：主页 fetch 后注入右侧 #result 面板，样式由主页提供
    return f"""<div class="card"><h3>检测结果</h3>
<div class="meta">帧来源：{html.escape(src_desc)} ｜ 排口点 {point}</div>
<div class="stats">
  <div class="stat"><div class="k">排污判定</div><div class="v"><span class="pill {vcls}">{verdict_cn}</span></div></div>
  <div class="stat"><div class="k">漂浮物判定</div><div class="v"><span class="pill {fcls}">{float_cn}</span></div></div>
  <div class="stat"><div class="k">综合评分</div><div class="v">{s:.3f}</div></div>
  <div class="stat"><div class="k">水面覆盖率</div><div class="v">{water_pct:.1f}%</div></div>
  <div class="stat"><div class="k">相对黑度</div><div class="v">{feats['rel_darkness']:.2f}</div></div>
  <div class="stat"><div class="k">波动能量比</div><div class="v">{feats['flow_energy']:.2f}</div></div>
  <div class="stat"><div class="k">颜色偏离</div><div class="v">{feats['color_dev']:.2f}</div></div>
</div></div>

<div class="card"><h3>漂浮物（水面范围内 {len(dets)} / 总检出 {len(dets_all)}）</h3>
{float_block}
</div>

<div class="card"><h3>标注图</h3>
<div class="legend">
  <span><span class="lg" style="background:#00c853;opacity:.5"></span>绿色蒙版 = 水面</span>
  <span><span class="lg" style="background:#f43f5e"></span>红 = 排口区</span>
  <span><span class="lg" style="background:#eab308"></span>黄 = 扩散区</span>
  <span><span class="lg" style="background:#22c55e"></span>绿圈 = 参考区</span>
  <span><span class="lg" style="background:#ef4444"></span>红框 = 漂浮物</span>
  <span><span class="lg" style="background:#d946ef"></span>紫叉 = 排口点</span>
</div>
<div class="imgbox">
<img class="out" title="点击放大" data-showing="ann"
  data-ann="data:image/jpeg;base64,{_b64img(ann)}"
  data-orig="data:image/jpeg;base64,{_b64img(img)}"
  src="data:image/jpeg;base64,{_b64img(ann)}">
<button class="swapbtn"
  onclick="var i=this.closest('.card').querySelector('img.out');var s=i.getAttribute('data-showing')==='orig';i.src=s?i.getAttribute('data-ann'):i.getAttribute('data-orig');i.setAttribute('data-showing',s?'ann':'orig');this.textContent=s?'查看原图':'切换回标注图';"
>查看原图</button>
</div>
<p class="hint" style="margin:8px 0 0">点击图片全屏查看，Esc 或点击空白处关闭</p>
</div>"""


def main():
    import os
    import uvicorn
    # 容器内必须绑 0.0.0.0 才能从宿主访问（compose 里已设 WEBAPP_HOST）
    uvicorn.run(app,
                host=os.environ.get("WEBAPP_HOST", "127.0.0.1"),
                port=int(os.environ.get("WEBAPP_PORT", "8090")))


if __name__ == "__main__":
    main()
