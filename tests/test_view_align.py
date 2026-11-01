import cv2
import numpy as np
from outlet_detect.view_align import ViewAligner

def make_scene(seed=0, size=(400, 600)):
    rng = np.random.RandomState(seed)
    img = np.full((*size, 3), 200, dtype=np.uint8)
    for _ in range(500):  # 随机方块作为特征点（模拟岸边参照物）
        x, y = rng.randint(20, size[1] - 40), rng.randint(20, size[0] - 40)
        cv2.rectangle(img, (x, y), (x + 15, y + 15),
                      int(rng.randint(0, 200)), -1)
    return img

def test_align_recovers_known_transform(tmp_path):
    ref = make_scene()
    ref_path = str(tmp_path / "ref.jpg")
    cv2.imwrite(ref_path, ref)
    point = (300, 200)
    # 模拟球机小幅转动：平移+缩放
    H_true = np.array([[1.05, 0.01, -30], [-0.01, 1.05, -20], [0, 0, 1]], dtype=np.float64)
    moved = cv2.warpPerspective(ref, H_true, (600, 400))
    aligner = ViewAligner(ref_path, point)
    result = aligner.align(moved)
    assert result.ok and result.in_frame
    mx, my = result.mapped_point
    # H_true 作用后的真实位置
    p = H_true @ np.array([300, 200, 1.0])
    ex, ey = p[0] / p[2], p[1] / p[2]
    assert abs(mx - ex) < 8 and abs(my - ey) < 8

def test_align_fails_on_unrelated_frame(tmp_path):
    ref = make_scene(seed=0)
    ref_path = str(tmp_path / "ref.jpg")
    cv2.imwrite(ref_path, ref)
    other = make_scene(seed=99)  # 完全不同的场景
    aligner = ViewAligner(ref_path, (300, 200))
    result = aligner.align(other)
    assert not result.ok


def test_align_without_outfall_point_only_checks_view(tmp_path):
    """未标定排口点（None）：视角校验照常，mapped_point=None 交给质心兜底。"""
    ref = make_scene()
    ref_path = str(tmp_path / "ref.jpg")
    cv2.imwrite(ref_path, ref)
    aligner = ViewAligner(ref_path, None)
    ok = aligner.align(ref)
    assert ok.ok and ok.in_frame and ok.mapped_point is None
    bad = aligner.align(make_scene(seed=99))
    assert not bad.ok
