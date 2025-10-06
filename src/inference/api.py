"""
二维码检测推理API
"""

import os
import io
import base64
import time
from typing import Dict, Optional
import numpy as np
from PIL import Image
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from src.inference.qrcode_detector import QRCodeDetector


# 初始化FastAPI应用
app = FastAPI(title="二维码/小程序码检测API",
              description="基于YOLOv8的二维码和小程序码检测服务",
              version="1.0.0")

# 全局变量存储检测器
detector: Optional[QRCodeDetector] = None


@app.on_event("startup")
async def load_model():
    """应用启动时加载模型"""
    global detector
    model_path = os.environ.get("MODEL_PATH", "models/yolov8/yolov8m_qr_detector/weights/best.pt")
    
    if os.path.exists(model_path):
        detector = QRCodeDetector(model_path)
        print(f"模型加载成功: {model_path}")
    else:
        print(f"模型文件不存在: {model_path}")
        # 在实际部署中，你可能想要抛出异常或使用默认模型


@app.get("/")
async def root():
    """根路径"""
    return {"message": "二维码/小程序码检测API服务已启动"}


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "model_loaded": detector is not None}


@app.post("/detect")
async def detect_qr_codes(file: UploadFile = File(...), 
                         min_confidence: float = 0.5) -> Dict:
    """检测上传图像中的二维码/小程序码"""
    if detector is None:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    try:
        # 读取上传的文件
        contents = await file.read()
        
        # 转换为numpy数组
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="无法解码图像文件")
        
        # 转换为RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 执行检测
        result = detector.detect_from_array(image, min_confidence)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测过程中发生错误: {str(e)}")


@app.post("/detect_base64")
async def detect_qr_codes_base64(image_data: Dict[str, str], 
                               min_confidence: float = 0.5) -> Dict:
    """检测Base64编码图像中的二维码/小程序码"""
    if detector is None:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    try:
        # 解码Base64图像数据
        image_base64 = image_data.get("image", "")
        if not image_base64:
            raise HTTPException(status_code=400, detail="未提供图像数据")
        
        # 移除可能的数据URI前缀
        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",")[1]
        
        # 解码Base64
        image_bytes = base64.b64decode(image_base64)
        
        # 转换为numpy数组
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="无法解码Base64图像数据")
        
        # 转换为RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 执行检测
        result = detector.detect_from_array(image, min_confidence)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测过程中发生错误: {str(e)}")


if __name__ == "__main__":
    # 启动服务器
    uvicorn.run("src.inference.api:app", 
                host="0.0.0.0", 
                port=8000, 
                reload=True)