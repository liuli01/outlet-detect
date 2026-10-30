"""海康 Artemis 平台适配：AK/SK 签名抓图 + 摄像头列表（移植自 hik-sdk.py）。"""
import base64
import hashlib
import hmac
import json
import time
import uuid
from urllib import parse

import cv2
import numpy as np
import requests

requests.packages.urllib3.disable_warnings()  # Artemis 自签证书


class HikApi:
    """Artemis AK/SK 签名 POST 客户端。api_path 需含 /artemis 前缀。"""

    def __init__(self, base_url, api_path, app_key, app_secret):
        self.base_url = base_url
        self.api_path = api_path
        self.app_key = app_key
        self.app_secret = app_secret

    def post(self, body):
        nonce = str(uuid.uuid4())
        ts = str(int(round(time.time() * 1000)))
        sign_str = ("POST\n*/*\napplication/json" + "\nx-ca-key:" + self.app_key
                    + "\nx-ca-nonce:" + nonce + "\nx-ca-timestamp:" + ts + "\n"
                    + self.api_path)
        signature = base64.b64encode(
            hmac.new(self.app_secret.encode(), sign_str.encode(),
                     digestmod=hashlib.sha256).digest()).decode()
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/json",
            "x-ca-key": self.app_key,
            "x-ca-signature-headers": "x-ca-key,x-ca-nonce,x-ca-timestamp",
            "x-ca-signature": signature,
            "x-ca-timestamp": ts,
            "x-ca-nonce": nonce,
        }
        r = requests.post(self.base_url + self.api_path, data=json.dumps(body),
                          headers=headers, verify=False, timeout=30)
        return r.json()


def make_capture_fn(base_url, app_key, app_secret):
    """capture_fn(body) -> picUrl（manualCapture）；失败抛 RuntimeError。"""
    api = HikApi(base_url, "/artemis/api/video/v1/manualCapture", app_key, app_secret)

    def capture_fn(body):
        resp = api.post(body)
        if resp.get("msg") == "success":
            return resp["data"]["picUrl"]
        raise RuntimeError(f"manualCapture 失败: {resp}")

    return capture_fn


def make_download_fn(timeout=30):
    """download_fn(url) -> BGR ndarray；解码失败抛 RuntimeError。"""

    def download_fn(url):
        r = requests.get(url, verify=False, timeout=timeout)
        r.raise_for_status()
        img = cv2.imdecode(np.frombuffer(r.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"图片解码失败: {url[:100]}")
        return img

    return download_fn


def get_index_code_list(platform_url="http://192.166.250.224:17082/api/jyly/cameraindexcode"):
    """从业务平台获取摄像头 IndexCode 列表。"""
    r = requests.post(platform_url,
                      headers={"Content-Type": "application/x-www-form-urlencoded"},
                      data=parse.urlencode({}, True), timeout=30)
    return [i["indexCode"] for i in r.json()["data"]]
