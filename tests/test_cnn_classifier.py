# tests/test_cnn_classifier.py
import numpy as np
import pytest
from outlet_detect.cnn_classifier import CNNClassifier, crop_for_cnn
from outlet_detect.rois import ROIs

def test_disabled_when_no_model():
    c = CNNClassifier(None)
    assert not c.enabled
    with pytest.raises(RuntimeError):
        c.classify(np.zeros((224, 224, 3), dtype=np.uint8))

def test_crop_for_cnn():
    frame = np.full((300, 400, 3), 100, dtype=np.uint8)
    out = np.zeros((300, 400), dtype=np.uint8); out[100:150, 150:200] = 255
    plume = np.zeros((300, 400), dtype=np.uint8); plume[150:200, 150:250] = 255
    crop = crop_for_cnn(frame, ROIs(outfall=out, plume=plume,
                                    reference=np.zeros((300, 400), dtype=np.uint8)))
    assert crop.shape == (224, 224, 3)
