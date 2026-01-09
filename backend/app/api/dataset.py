"""
数据集管理 API
"""
import os
import sys
import yaml
import shutil
import threading
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from app.config import settings, YOLOV5_DIR, DATASET_DIR
from app.models import DatasetInfo

router = APIRouter()

# 下载任务状态存储
download_tasks: Dict[str, Dict[str, Any]] = {}


def check_dataset_exists(dataset_dir: Path, dataset_type: str = "coco") -> dict:
    """检查数据集是否存在并统计图像数量"""
    result = {
        "exists": False,
        "images_exist": False,
        "labels_exist": False,
        "train_images": 0,
        "val_images": 0,
        "test_images": 0
    }
    
    if not dataset_dir.exists():
        return result
    
    result["exists"] = True
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    
    if images_dir.exists():
        result["images_exist"] = True
        # 根据数据集类型统计
        if dataset_type == "coco":
            for split in ["train2017", "val2017", "test2017"]:
                split_dir = images_dir / split
                if split_dir.exists():
                    count = sum(1 for f in split_dir.glob("*.jpg")) + sum(1 for f in split_dir.glob("*.png"))
                    if "train" in split:
                        result["train_images"] = count
                    elif "val" in split:
                        result["val_images"] = count
                    elif "test" in split:
                        result["test_images"] = count
        elif dataset_type == "voc":
            for subdir in images_dir.iterdir():
                if subdir.is_dir():
                    count = sum(1 for f in subdir.glob("*.jpg")) + sum(1 for f in subdir.glob("*.png"))
                    if "train" in subdir.name:
                        result["train_images"] += count
                    elif "val" in subdir.name:
                        result["val_images"] += count
                    elif "test" in subdir.name:
                        result["test_images"] += count
        else:
            # 通用处理
            for split in ["train", "val", "test"]:
                split_dir = images_dir / split
                if split_dir.exists():
                    count = sum(1 for f in split_dir.glob("*.jpg")) + sum(1 for f in split_dir.glob("*.png"))
                    result[f"{split}_images"] = count
    
    if labels_dir.exists():
        result["labels_exist"] = sum(1 for _ in labels_dir.rglob("*.txt")) > 0
    
    return result


@router.get("/list")
async def list_datasets():
    """
    列出所有可用数据集，包括数据集是否存在的状态
    """
    datasets = []
    
    # 检查 COCO 数据集
    coco_yaml = DATASET_DIR / "coco" / "coco.yaml"
    if coco_yaml.exists():
        with open(coco_yaml, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
            # 检查数据集实际存在状态
            coco_dir = DATASET_DIR / "coco"
            status = check_dataset_exists(coco_dir, "coco")
            
            names = config.get("names", {})
            if isinstance(names, dict):
                class_list = list(names.values())
            else:
                class_list = names if isinstance(names, list) else []
            
            datasets.append({
                "name": "COCO",
                "path": str(coco_yaml),
                "type": "coco",
                "classes": class_list,
                "num_classes": len(class_list),
                "status": status,
                "downloadable": True
            })
    
    # 检查 VOC 数据集
    voc_yaml = DATASET_DIR / "VOC" / "VOC.yaml"
    if voc_yaml.exists():
        with open(voc_yaml, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
            # 检查数据集实际存在状态
            voc_dir = DATASET_DIR / "VOC"
            status = check_dataset_exists(voc_dir, "voc")
            
            names = config.get("names", {})
            if isinstance(names, dict):
                class_list = list(names.values())
            else:
                class_list = names if isinstance(names, list) else []
            
            datasets.append({
                "name": "Pascal VOC",
                "path": str(voc_yaml),
                "type": "voc",
                "classes": class_list,
                "num_classes": len(class_list),
                "status": status,
                "downloadable": True
            })
    
    # 检查自定义数据集
    custom_dir = Path(settings.CUSTOM_DATASET_DIR)
    if custom_dir.exists():
        for yaml_file in custom_dir.glob("**/*.yaml"):
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    
                    dataset_dir = yaml_file.parent
                    status = check_dataset_exists(dataset_dir, "custom")
                    
                    names = config.get("names", {})
                    if isinstance(names, dict):
                        class_list = list(names.values())
                    else:
                        class_list = names if isinstance(names, list) else []
                    
                    datasets.append({
                        "name": yaml_file.stem,
                        "path": str(yaml_file),
                        "type": "custom",
                        "classes": class_list,
                        "num_classes": len(class_list),
                        "status": status,
                        "downloadable": False
                    })
            except:
                pass
    
    return {"datasets": datasets}


def run_dataset_download(task_id: str, dataset_name: str, yaml_path: Path):
    """后台运行数据集下载任务"""
    global download_tasks
    
    try:
        download_tasks[task_id]["status"] = "running"
        download_tasks[task_id]["message"] = f"正在下载 {dataset_name} 数据集..."
        
        # 读取yaml配置获取下载脚本
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        download_script = config.get('download', '')
        if not download_script:
            download_tasks[task_id]["status"] = "failed"
            download_tasks[task_id]["message"] = "数据集配置文件中没有下载脚本"
            return
        
        # 创建临时下载脚本
        temp_script = YOLOV5_DIR / f"temp_download_{task_id}.py"
        
        # 构建下载脚本内容
        script_content = f'''
import sys
sys.path.insert(0, r"{YOLOV5_DIR}")

yaml = {repr(config)}

{download_script}
'''
        
        with open(temp_script, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # 执行下载脚本
        process = subprocess.Popen(
            [sys.executable, str(temp_script)],
            cwd=str(YOLOV5_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        download_tasks[task_id]["process"] = process
        
        output_lines = []
        for line in process.stdout:
            output_lines.append(line.strip())
            download_tasks[task_id]["output"] = output_lines[-50:]
            
            # 解析下载进度
            if "%" in line:
                try:
                    # 尝试从输出中提取百分比
                    import re
                    match = re.search(r'(\d+(?:\.\d+)?)\s*%', line)
                    if match:
                        download_tasks[task_id]["progress"] = float(match.group(1))
                except:
                    pass
        
        process.wait()
        
        # 清理临时脚本
        if temp_script.exists():
            temp_script.unlink()
        
        if process.returncode == 0:
            download_tasks[task_id]["status"] = "completed"
            download_tasks[task_id]["message"] = f"{dataset_name} 数据集下载完成"
            download_tasks[task_id]["progress"] = 100
        else:
            download_tasks[task_id]["status"] = "failed"
            download_tasks[task_id]["message"] = f"下载失败，返回码: {process.returncode}"
            
    except Exception as e:
        download_tasks[task_id]["status"] = "failed"
        download_tasks[task_id]["message"] = str(e)


@router.post("/download/{dataset_name}")
async def download_dataset(dataset_name: str, background_tasks: BackgroundTasks):
    """
    下载数据集到 datasets 目录
    支持 COCO 和 VOC 等标准数据集
    """
    import uuid
    
    # 确定数据集配置文件
    dataset_name_lower = dataset_name.lower().replace(" ", "")
    
    if dataset_name_lower == "coco":
        yaml_path = DATASET_DIR / "coco" / "coco.yaml"
        if not yaml_path.exists():
            # 尝试从yolov5/data复制
            source_yaml = YOLOV5_DIR / "data" / "coco.yaml"
            if source_yaml.exists():
                (DATASET_DIR / "coco").mkdir(parents=True, exist_ok=True)
                shutil.copy(source_yaml, yaml_path)
                # 更新path为datasets目录
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                content = content.replace('../datasets/coco', str(DATASET_DIR / 'coco').replace('\\', '/'))
                with open(yaml_path, 'w', encoding='utf-8') as f:
                    f.write(content)
    elif dataset_name_lower in ["voc", "pascalvoc"]:
        yaml_path = DATASET_DIR / "VOC" / "VOC.yaml"
        if not yaml_path.exists():
            source_yaml = YOLOV5_DIR / "data" / "VOC.yaml"
            if source_yaml.exists():
                (DATASET_DIR / "VOC").mkdir(parents=True, exist_ok=True)
                shutil.copy(source_yaml, yaml_path)
                # 更新path为datasets目录
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                content = content.replace('../datasets/VOC', str(DATASET_DIR / 'VOC').replace('\\', '/'))
                with open(yaml_path, 'w', encoding='utf-8') as f:
                    f.write(content)
    else:
        raise HTTPException(status_code=400, detail=f"不支持下载的数据集: {dataset_name}")
    
    if not yaml_path.exists():
        raise HTTPException(status_code=404, detail="数据集配置文件不存在")
    
    # 创建下载任务
    task_id = str(uuid.uuid4())
    download_tasks[task_id] = {
        "task_id": task_id,
        "dataset_name": dataset_name,
        "status": "pending",
        "progress": 0,
        "message": "准备下载...",
        "output": []
    }
    
    # 后台启动下载
    thread = threading.Thread(target=run_dataset_download, args=(task_id, dataset_name, yaml_path))
    thread.start()
    
    return {
        "task_id": task_id,
        "message": f"数据集 {dataset_name} 下载任务已创建",
        "status": "pending"
    }


@router.get("/download/status/{task_id}")
async def get_download_status(task_id: str):
    """获取数据集下载状态"""
    if task_id not in download_tasks:
        raise HTTPException(status_code=404, detail="下载任务不存在")
    
    task = download_tasks[task_id]
    return {
        "task_id": task["task_id"],
        "dataset_name": task["dataset_name"],
        "status": task["status"],
        "progress": task.get("progress", 0),
        "message": task.get("message", ""),
        "output": task.get("output", [])[-20:]  # 只返回最后20行
    }


@router.get("/check/{dataset_name}")
async def check_dataset_status(dataset_name: str):
    """
    检查数据集是否存在及其状态
    """
    dataset_name_lower = dataset_name.lower().replace(" ", "")
    
    if dataset_name_lower == "coco":
        dataset_dir = DATASET_DIR / "coco"
        yaml_path = dataset_dir / "coco.yaml"
        dataset_type = "coco"
    elif dataset_name_lower in ["voc", "pascalvoc"]:
        dataset_dir = DATASET_DIR / "VOC"
        yaml_path = dataset_dir / "VOC.yaml"
        dataset_type = "voc"
    else:
        # 检查自定义数据集
        dataset_dir = Path(settings.CUSTOM_DATASET_DIR) / dataset_name
        yaml_path = dataset_dir / f"{dataset_name}.yaml"
        dataset_type = "custom"
    
    status = check_dataset_exists(dataset_dir, dataset_type)
    status["yaml_exists"] = yaml_path.exists()
    status["dataset_dir"] = str(dataset_dir)
    
    return status


@router.get("/{dataset_name}")
async def get_dataset_info(dataset_name: str):
    """
    获取数据集详细信息
    """
    # 查找数据集配置文件
    yaml_path = None
    dataset_base_dir = None
    
    # 检查预定义数据集
    dataset_name_lower = dataset_name.lower().replace(" ", "")
    if dataset_name_lower == "coco":
        yaml_path = DATASET_DIR / "coco" / "coco.yaml"
        dataset_base_dir = DATASET_DIR / "coco"
    elif dataset_name_lower in ["voc", "pascalvoc"]:
        yaml_path = DATASET_DIR / "VOC" / "VOC.yaml"
        dataset_base_dir = DATASET_DIR / "VOC"
    else:
        # 检查自定义数据集
        custom_path = Path(settings.CUSTOM_DATASET_DIR) / f"{dataset_name}.yaml"
        if custom_path.exists():
            yaml_path = custom_path
            dataset_base_dir = custom_path.parent
    
    if not yaml_path or not yaml_path.exists():
        raise HTTPException(status_code=404, detail="数据集不存在")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 统计图像数量
    train_count = val_count = test_count = 0
    
    train_paths = config.get("train", "")
    val_paths = config.get("val", "")
    test_paths = config.get("test", "")
    
    # 辅助函数：统计目录中的图像数量
    def count_images_in_paths(paths, base_dir):
        """统计路径（可以是单个路径或路径列表）中的图像数量"""
        if not paths:
            return 0
        
        # 确保是列表
        if isinstance(paths, str):
            paths = [paths]
        
        total = 0
        for path in paths:
            full_path = Path(path)
            if not full_path.is_absolute():
                full_path = base_dir / path
            
            if full_path.exists() and full_path.is_dir():
                total += len(list(full_path.glob("*.jpg")))
                total += len(list(full_path.glob("*.jpeg")))
                total += len(list(full_path.glob("*.png")))
        
        return total
    
    train_count = count_images_in_paths(train_paths, dataset_base_dir)
    val_count = count_images_in_paths(val_paths, dataset_base_dir)
    test_count = count_images_in_paths(test_paths, dataset_base_dir)
    
    return {
        "name": dataset_name,
        "path": str(yaml_path),
        "classes": config.get("names", []),
        "num_classes": config.get("nc", len(config.get("names", [])) if isinstance(config.get("names"), (list, dict)) else 0),
        "split": {
            "train": train_count,
            "val": val_count,
            "test": test_count
        },
        "config": config
    }


@router.post("/create")
async def create_custom_dataset(
    name: str = Form(..., description="数据集名称"),
    classes: str = Form(..., description="类别列表，逗号分隔"),
    description: Optional[str] = Form(None, description="数据集描述")
):
    """
    创建自定义数据集
    """
    # 创建数据集目录结构
    dataset_dir = Path(settings.CUSTOM_DATASET_DIR) / name
    
    if dataset_dir.exists():
        raise HTTPException(status_code=400, detail="数据集已存在")
    
    try:
        # 创建目录结构
        (dataset_dir / "images" / "train").mkdir(parents=True)
        (dataset_dir / "images" / "val").mkdir(parents=True)
        (dataset_dir / "images" / "test").mkdir(parents=True)
        (dataset_dir / "labels" / "train").mkdir(parents=True)
        (dataset_dir / "labels" / "val").mkdir(parents=True)
        (dataset_dir / "labels" / "test").mkdir(parents=True)
        
        # 解析类别
        class_list = [c.strip() for c in classes.split(",")]
        
        # 创建配置文件
        config = {
            "path": str(dataset_dir),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(class_list),
            "names": class_list
        }
        
        if description:
            config["description"] = description
        
        yaml_path = dataset_dir / f"{name}.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        
        return {
            "success": True,
            "message": "数据集创建成功",
            "dataset": {
                "name": name,
                "path": str(dataset_dir),
                "yaml_path": str(yaml_path),
                "classes": class_list
            }
        }
        
    except Exception as e:
        # 清理
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{dataset_name}/upload")
async def upload_dataset_images(
    dataset_name: str,
    files: List[UploadFile] = File(..., description="图像文件"),
    split: str = Form("train", description="数据集划分 (train/val/test)")
):
    """
    上传图像到数据集
    """
    if split not in ["train", "val", "test"]:
        raise HTTPException(status_code=400, detail="无效的数据集划分")
    
    # 查找数据集目录
    dataset_dir = Path(settings.CUSTOM_DATASET_DIR) / dataset_name
    
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail="数据集不存在")
    
    images_dir = dataset_dir / "images" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    
    uploaded = []
    failed = []
    
    for file in files:
        try:
            if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
                failed.append({"filename": file.filename, "error": "不支持的文件类型"})
                continue
            
            save_path = images_dir / file.filename
            content = await file.read()
            
            with open(save_path, "wb") as f:
                f.write(content)
            
            uploaded.append(file.filename)
            
        except Exception as e:
            failed.append({"filename": file.filename, "error": str(e)})
    
    return {
        "success": True,
        "uploaded": len(uploaded),
        "failed": len(failed),
        "uploaded_files": uploaded,
        "failed_files": failed
    }


@router.delete("/{dataset_name}")
async def delete_dataset(dataset_name: str):
    """
    删除自定义数据集
    """
    dataset_dir = Path(settings.CUSTOM_DATASET_DIR) / dataset_name
    
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail="数据集不存在")
    
    try:
        shutil.rmtree(dataset_dir)
        return {"success": True, "message": "数据集已删除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_name}/images")
async def list_dataset_images(
    dataset_name: str,
    split: str = "train",
    page: int = 1,
    page_size: int = 20
):
    """
    列出数据集中的图像
    """
    # 查找数据集目录
    dataset_dir = Path(settings.CUSTOM_DATASET_DIR) / dataset_name
    is_voc = False
    is_coco = False
    
    if not dataset_dir.exists():
        # 检查预定义数据集
        dataset_name_lower = dataset_name.lower().replace(" ", "")
        if dataset_name_lower == "coco":
            dataset_dir = DATASET_DIR / "coco"
            is_coco = True
        elif dataset_name_lower in ["voc", "pascalvoc"]:
            dataset_dir = DATASET_DIR / "VOC"
            is_voc = True
    
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail=f"数据集目录不存在: {dataset_dir}")
    
    # 查找图像目录
    images = []
    images_dir = dataset_dir / "images"
    
    if not images_dir.exists():
        raise HTTPException(status_code=404, detail=f"图像目录不存在: {images_dir}")
    
    if is_voc:
        # VOC 数据集特殊处理: train2007, train2012, val2007, val2012 等
        # 根据 split 匹配对应目录（train 匹配 train2007, train2012; val 匹配 val2007, val2012; test 匹配 test2007）
        for subdir in images_dir.iterdir():
            if subdir.is_dir() and subdir.name.startswith(split):
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                    images.extend(subdir.glob(ext))
    elif is_coco:
        # COCO 数据集: train2017, val2017, test2017
        split_dir = images_dir / f"{split}2017"
        if split_dir.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                images.extend(split_dir.glob(ext))
    else:
        # 自定义数据集
        split_dir = images_dir / split
        if split_dir.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                images.extend(split_dir.glob(ext))
        elif images_dir.exists():
            # 直接在 images 目录下查找
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                images.extend(images_dir.glob(ext))
    
    # 排序
    images = sorted(images, key=lambda x: x.name)
    
    # 分页
    total = len(images)
    start = (page - 1) * page_size
    end = start + page_size
    paginated = images[start:end]
    
    # 转换路径为URL
    def get_image_url(img_path: Path) -> str:
        # 获取相对于数据集目录的路径
        try:
            rel_path = img_path.relative_to(DATASET_DIR)
            return f"/datasets/{rel_path.as_posix()}"
        except ValueError:
            # 自定义数据集
            try:
                rel_path = img_path.relative_to(Path(settings.CUSTOM_DATASET_DIR))
                return f"/custom_datasets/{rel_path.as_posix()}"
            except ValueError:
                return str(img_path)
    
    return {
        "images": [{"name": img.name, "path": get_image_url(img)} for img in paginated],
        "total": total,
        "page": page,
        "page_size": page_size
    }
