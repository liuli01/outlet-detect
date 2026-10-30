import numpy as np
import pytest

from outlet_detect.reporter import (MinioReporter, ReportEvent, ReporterError,
                                    annotate_float)


class FakeMinio:
    def __init__(self):
        self.uploads = []

    def fput_object(self, bucket, object_name, file_path, content_type=None):
        self.uploads.append((bucket, object_name, file_path))

    def presigned_get_object(self, bucket, object_name):
        return f"http://192.166.250.223:17077/{bucket}/{object_name}?sig=x"


def make_reporter(**kw):
    posts = []

    def fake_post(url, data):
        posts.append((url, data))

    return MinioReporter("192.166.250.223:17077", "ak", "sk",
                         "http://192.166.250.224:17082/api/insert_ai_result",
                         external_host="223.108.117.18",
                         client=FakeMinio(), post_fn=fake_post, **kw), posts


def test_report_uploads_and_posts(tmp_path):
    rep, posts = make_reporter()
    img = tmp_path / "a.jpg"
    import cv2
    cv2.imwrite(str(img), np.zeros((10, 10, 3), dtype=np.uint8))
    ev = ReportEvent(index_code="cam01", input_path=str(img), output_path=str(img),
                     result_a=0.88, result_b=0.72, result_c=0.72)
    url = rep.report(ev)
    # 原图+标注图都上传（input 为本地路径时）
    buckets = [u[0] for u in rep.client.uploads]
    assert buckets == ["input-image", "ai-image"]
    # 外网地址替换
    assert url.startswith("http://223.108.117.18/")
    # 上报字段
    data = posts[0][1]
    assert data["IndexCode"] == "cam01"
    assert data["ResultA"] == 0.88 and data["ResultB"] == 0.72 and data["ResultC"] == 0.72
    assert data["InputPicture"].startswith("http://223.108.117.18/")


def test_report_uses_url_input_directly():
    rep, posts = make_reporter()
    ev = ReportEvent(index_code="cam01", input_path="http://hik/pic.jpg",
                     output_path="http://hik/pic.jpg", result_b=0.9, result_c=0.9)
    # input 是 URL 时不上传原图，只上传 output（output 此处也是 URL，跳过上传）
    rep.report(ev)
    assert rep.client.uploads == []
    assert posts[0][1]["InputPicture"] == "http://hik/pic.jpg"


def test_result_values_clamped():
    rep, posts = make_reporter()
    ev = ReportEvent(index_code="c", input_path="http://x/1.jpg",
                     output_path="http://x/2.jpg", result_a=1.5, result_b=-0.2, result_c=0.5)
    rep.report(ev)
    assert posts[0][1]["ResultA"] == 1.0 and posts[0][1]["ResultB"] == 0.0


def test_upload_failure_raises_reporter_error(tmp_path):
    class BadMinio(FakeMinio):
        def fput_object(self, *a, **k):
            raise ConnectionError("minio down")
    posts = []
    rep = MinioReporter("h:9000", "ak", "sk", "http://r/api",
                        client=BadMinio(), post_fn=lambda u, d: posts.append(d))
    ev = ReportEvent(index_code="c", input_path=str(tmp_path / "x.jpg"),
                     output_path=str(tmp_path / "x.jpg"))
    (tmp_path / "x.jpg").write_bytes(b"x")
    with pytest.raises(ReporterError):
        rep.report(ev)


def test_annotate_float_draws_boxes():
    from outlet_detect.float_detector import FloatDetection
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = annotate_float(frame, [FloatDetection("bottle", 0.88, (10, 20, 50, 60))])
    assert out.shape == frame.shape
    assert out.sum() > 0  # 画了内容
