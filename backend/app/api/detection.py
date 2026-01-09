"""
目标检测 API
"""
import os
import uuid
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse

from app.config import settings, BASE_DIR, DATASET_DIR
from app.models import DetectionRequest, DetectionResult, get_class_info
from app.services.detector import detector

router = APIRouter()


@router.post("/detect", response_model=DetectionResult)
async def detect_image(
    file: UploadFile = File(..., description="要检测的图像文件"),
    conf_threshold: float = Form(0.25, description="置信度阈值"),
    iou_threshold: float = Form(0.45, description="IOU阈值"),
    img_size: int = Form(640, description="推理图像尺寸"),
    weights: str = Form("yolov5s.pt", description="模型权重"),
    classes: Optional[str] = Form(None, description="类别ID列表，逗号分隔")
):
    """
    对上传的图像进行目标检测
    """
    # 验证文件类型
    if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file.content_type}"
        )
    
    # 保存上传文件
    file_id = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix
    save_path = Path(settings.UPLOAD_DIR) / "images" / f"{file_id}{file_ext}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 解析类别列表
        class_list = None
        if classes:
            class_list = [int(c.strip()) for c in classes.split(",")]
        
        # 执行检测
        result = detector.detect(
            image_path=str(save_path),
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            img_size=img_size,
            weights=weights,
            classes=class_list
        )
        
        # 更新图像路径为相对路径（用于前端访问）
        result.image_path = f"/uploads/images/{file_id}{file_ext}"
        
        return result
        
    except Exception as e:
        # 清理文件
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect/batch")
async def detect_images_batch(
    files: List[UploadFile] = File(..., description="要检测的图像文件列表"),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    img_size: int = Form(640),
    weights: str = Form("yolov5s.pt")
):
    """
    批量目标检测
    """
    results = []
    
    for file in files:
        if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
            continue
        
        file_id = str(uuid.uuid4())
        file_ext = Path(file.filename).suffix
        save_path = Path(settings.UPLOAD_DIR) / "images" / f"{file_id}{file_ext}"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(save_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            result = detector.detect(
                image_path=str(save_path),
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                img_size=img_size,
                weights=weights
            )
            
            result.image_path = f"/uploads/images/{file_id}{file_ext}"
            results.append(result)
            
        except Exception as e:
            results.append({
                "error": str(e),
                "filename": file.filename
            })
    
    return {"results": results, "total": len(results)}


@router.post("/detect/path")
async def detect_from_path(
    image_path: str = Form(..., description="图像相对路径，如 /uploads/augmented/xxx.jpg"),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    img_size: int = Form(640),
    weights: str = Form("yolov5s.pt"),
    classes: Optional[str] = Form(None, description="类别ID列表，逗号分隔")
):
    """
    从本地路径检测图像（用于预处理后的增强图片）
    """
    # 将相对路径转换为绝对路径
    # 支持的路径格式: /uploads/..., /datasets/..., /custom_datasets/...
    if image_path.startswith('/uploads/'):
        abs_path = BASE_DIR / image_path.lstrip('/')
    elif image_path.startswith('/datasets/'):
        abs_path = DATASET_DIR / image_path.replace('/datasets/', '')
    elif image_path.startswith('/custom_datasets/'):
        abs_path = Path(settings.CUSTOM_DATASET_DIR) / image_path.replace('/custom_datasets/', '')
    else:
        raise HTTPException(status_code=400, detail=f"不支持的路径格式: {image_path}")
    
    if not abs_path.exists():
        raise HTTPException(status_code=404, detail=f"图像文件不存在: {image_path}")
    
    try:
        # 解析类别列表
        class_list = None
        if classes:
            class_list = [int(c.strip()) for c in classes.split(",")]
        
        # 复制图片到 images 目录并生成新的 ID
        file_id = str(uuid.uuid4())
        file_ext = abs_path.suffix
        new_save_path = Path(settings.UPLOAD_DIR) / "images" / f"{file_id}{file_ext}"
        new_save_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(abs_path, new_save_path)
        
        # 执行检测
        result = detector.detect(
            image_path=str(new_save_path),
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            img_size=img_size,
            weights=weights,
            classes=class_list
        )
        
        result.image_path = f"/uploads/images/{file_id}{file_ext}"
        result.image_id = file_id
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect/url")
async def detect_from_url(
    image_url: str = Form(..., description="图像URL"),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    img_size: int = Form(640),
    weights: str = Form("yolov5s.pt")
):
    """
    从URL检测图像
    """
    import urllib.request
    
    file_id = str(uuid.uuid4())
    # 从URL获取扩展名
    url_path = Path(image_url.split("?")[0])
    file_ext = url_path.suffix or ".jpg"
    save_path = Path(settings.UPLOAD_DIR) / "images" / f"{file_id}{file_ext}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 下载图像
        urllib.request.urlretrieve(image_url, str(save_path))
        
        result = detector.detect(
            image_path=str(save_path),
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            img_size=img_size,
            weights=weights
        )
        
        result.image_path = f"/uploads/images/{file_id}{file_ext}"
        return result
        
    except Exception as e:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/weights")
async def get_available_weights():
    """获取可用的模型权重列表"""
    weights = detector.get_available_weights()
    return {"weights": weights}


@router.get("/classes")
async def get_classes(dataset: str = Query("coco", description="数据集类型: coco, voc, custom, all")):
    """获取支持的目标类别列表"""
    return {"classes": get_class_info(dataset)}


@router.post("/classes/custom")
async def add_custom_class_api(name: str = Query(..., description="类别名称"), 
                                color: str = Query(None, description="颜色(hex格式)")):
    """添加自定义类别"""
    try:
        from app.models import add_custom_class
        new_class = add_custom_class(name, color)
        return {"success": True, "class": new_class}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/classes/custom/{class_id}")
async def delete_custom_class_api(class_id: int):
    """删除自定义类别"""
    try:
        from app.models import delete_custom_class
        delete_custom_class(class_id)
        return {"success": True, "message": f"类别 {class_id} 已删除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/classes/custom/{class_id}")
async def update_custom_class_api(class_id: int,
                                   name: str = Query(None, description="新类别名称"),
                                   color: str = Query(None, description="新颜色(hex格式)")):
    """更新自定义类别"""
    try:
        from app.models import update_custom_class
        updated_class = update_custom_class(class_id, name, color)
        return {"success": True, "class": updated_class}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/info")
async def get_model_info(weights: str = Query("yolov5s.pt")):
    """获取模型信息"""
    try:
        info = detector.get_model_info(weights)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
