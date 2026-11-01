"""
海康安防平台API文档https://open.hikvision.com/docs/docId?productId=5c67f1e2f05948198c909700&version=%2Ff8356830af1d40f3b1da7db12baa47af&curNodeId=88b95f4c-be46-4894-889c-92685aa59602
协议：仅支持 HTTPS 传输
url格式：https://{hostname}:{port}/artemis/{uri}
# AK\SK摘要认证
调用 API 时，如果API需要安全认证，首先需要获取API的授权，得到AppKey和AppSecret；
其次，拼接签名字符串，将计算后的签名放在请求的 Header 传入，网关会通过对称计算签名来验证请求者的身份。
"""
 #%%
import base64
import json
import time
import uuid
import hmac  # hex-based message authentication code 哈希消息认证码
import hashlib  # 提供了很多加密的算法
import requests
from urllib import parse
from datetime import datetime
import os
import shutil
import requests
import json

from minio.api import Minio
from roboflow_sdk import image_predict
import loguru



loguru.logger.add(
    "./app.log",
    rotation="5 MB",
    retention="90 days",
    compression="zip",
    # serialize=True, 	# serializes log messages to JSON.  Turn off for log files.
     enqueue=True
)
logger=loguru.logger



 
# 禁用requests SSL安全请求警告
from requests.packages.urllib3.exceptions import InsecureRequestWarning  # type: ignore
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)  # type: ignore


class Hik_API():
    def __init__(self,base_url,api_get_address_url,appKey,appSecret) -> None:
        self.base_url = base_url   # 可以正常访问的IP地址         
        self.api_get_address_url = api_get_address_url   # "/artemis/api/video/v1/cameras/previewURLs"，注意增加/artemis,
        self.appKey = appKey                             #  appKey 从环境变量 HIK_APP_KEY 读取
        self.appSecret = appSecret                       #  appSecret 从环境变量 HIK_APP_SECRET 读取               
 

    def requests(self,body):
 
        x_ca_nonce = str(uuid.uuid4())
        x_ca_timestamp = str(int(round(time.time()) * 1000))
        # sign_str 的拼接很关键，不然得不到正确的签名
        sign_str = "POST\n*/*\napplication/json" + "\nx-ca-key:" + self.appKey + "\nx-ca-nonce:" + \
                x_ca_nonce + "\nx-ca-timestamp:" + x_ca_timestamp + "\n" + \
                self.api_get_address_url

        # 签名转换
        temp = hmac.new(self.appSecret.encode(), sign_str.encode(), digestmod=hashlib.sha256)
        signature = base64.b64encode(temp.digest()).decode()
        # print("[INFO] 获取到的签名值为：", signature)
 
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/json",
            "x-ca-key": self.appKey,  # appKey，即 AK
            "x-ca-signature-headers": "x-ca-key,x-ca-nonce,x-ca-timestamp",
            "x-ca-signature": signature,  # 需要计算得到的签名，此处通过后台得到
            "x-ca-timestamp": x_ca_timestamp,  # 时间戳
            "x-ca-nonce": x_ca_nonce  # UUID，结合时间戳防重复
        }
 
        url = self.base_url + self.api_get_address_url
        results = requests.post(url, data=json.dumps(body), headers=headers,  verify=False)
        # print(results.json())
        # {'code': '0', 'msg': 'success', 'data': {'url': 'rtsp://ip:554/openUrl/Kp0VqXS'}}
        return results.json()
        
def get_picture_url(body):
    hik_api_preview= Hik_API (base_url = "https://223.71.143.153:14443",api_get_address_url="/artemis/api/video/v1/manualCapture",appKey=os.environ["HIK_APP_KEY"],appSecret=os.environ["HIK_APP_SECRET"])
    resp_json=hik_api_preview.requests(body)
    if resp_json["msg"]=="success":
        pic_url=resp_json["data"]["picUrl"]
        return pic_url
    else:
        print(f"{body} get picture url failed")
        return False

        
def get_video_url(body):
    hik_api_preview= Hik_API (base_url = "https://223.71.143.153:14443",api_get_address_url="/artemis/api/video/v1/cameras/playbackURLs",appKey=os.environ["HIK_APP_KEY"],appSecret=os.environ["HIK_APP_SECRET"])
    resp_json=hik_api_preview.requests(body)
    print(resp_json)
    if resp_json["msg"]=="success":
       video_url=resp_json["data"]["url"]
       return video_url
    else:
        print(f"{body} get_video_url failed")
        return False

# 获取摄像头ID
#%% 获取摄像机IndexCode列表
def get_index_code_list():
    code_list=[]
    requestUrl = 'http://192.166.250.224:17082/api/jyly/cameraindexcode'
    headers = {
    'Content-Type': 'application/x-www-form-urlencoded'
    }
    formData = {
    
    }
    data = parse.urlencode(formData, True)
    response = requests.post(requestUrl, headers = headers, data = data)
    code_dict=response.json()["data"]
    for i in code_dict:
        code_list.append(i.get("indexCode"))
    return code_list

#%% 存入模型识别结果
def save_model_result(index_code,input_picture,output_picture,resulta,resultb,resultc):
    requestUrl = 'http://192.166.250.224:17082/api/insert_ai_result'
    headers = {
    'Content-Type': 'application/x-www-form-urlencoded'
    }
    formData = {
    "IndexCode": index_code,
    "InputPicture": input_picture,
    "OutputPicture": output_picture,
    "ResultA": resulta,
    "ResultB": resultb,
    "ResultC": resultc
    }
    data = parse.urlencode(formData, True)
    requests.post(requestUrl, headers = headers, data = data)



#%% minio上传

client = Minio(endpoint="192.166.250.223:17077", 
               access_key="minio", 
               secret_key=os.environ["MINIO_SECRET_KEY"],
               secure=False)
def fput_object(bucket_name,object_name,file_path):
    client.fput_object(bucket_name, object_name, file_path,content_type="image/jpeg")

def fget_object_url(object_name):
    """获取训练集文件地址,桶策略需要对外公开,传入"objetc_name"含文件路径
    """
    url=client.presigned_get_object(bucket_name='ai-image',object_name=object_name)
    address, params = url.split("?") 
    address_url=address.replace("192.166.250.223","223.108.117.18")
    return address_url

def float_detection(index_code,ignore_list=[],confidence=0.9,color_confidence=0.9):
        # 输入文件名称
        input_name="./input.jpg"
        # 获取当前时间  
        now = datetime.now()  
        # 格式化日期和时间  
        formatted_date = now.strftime("%Y-%m-%d-%H-%M")
        object_name = f"{index_code}-{formatted_date}.jpg"

        body = {
        "cameraIndexCode": index_code,
        "recordLocation": "1",
        "protocol": "hls",
        "transmode": 1,
        "uuid": ""
        }
        input_picture_url=get_picture_url(body=body)

        if input_picture_url:
            response = requests.get(input_picture_url, verify=False)
            with open(input_name,mode = "wb") as f:
                f.write(response.content)
            # logger.info(f"识别图片:{input_picture_url}")
            float_result=image_predict(model_id="floating-o49zc/3",input_image=input_name,output_image=object_name,ignore_list=ignore_list,confidence=confidence)
            if float_result:
                fput_object(bucket_name='ai-image',object_name=object_name,file_path=f"./output/{object_name}")
                fput_object(bucket_name='input-image',object_name=object_name,file_path=input_name)
                output_picture_url=fget_object_url(object_name)
                save_model_result(index_code,input_picture_url,output_picture_url,resulta=confidence,resultb=0,resultc=0)
                logger.info(f"识别图片:{output_picture_url},模型识别结果: {float_result}")
           
            # 误报多次，暂时屏蔽，自定义准确性0.95报警
            color_result=image_predict(model_id="river_color/4",input_image=input_name,output_image=object_name,ignore_list=ignore_list,confidence=color_confidence)
            if color_result:
                fput_object(bucket_name='ai-image',object_name=object_name,file_path=f"./output/{object_name}")
                fput_object(bucket_name='input-image',object_name=object_name,file_path=input_name)
                output_picture_url=fget_object_url(object_name)
                save_model_result(index_code,input_picture_url,output_picture_url,resulta=0,resultb=color_confidence,resultc=color_confidence)
                logger.info(f"识别图片:{output_picture_url},模型识别结果: {color_result}")
        
try:
    os.makedirs("./output",exist_ok=True)
    index_code_list=get_index_code_list()
    for i in range(len(index_code_list)):
        index_code=index_code_list[i]
        float_detection(index_code,ignore_list=["aeration"],confidence=0.6,color_confidence=0.95)
    shutil.rmtree("./output",ignore_errors=True)
    logger.info("20250305程序执行完成")
except Exception as e:
    logger.error(f"{i} 出错：{e}")

