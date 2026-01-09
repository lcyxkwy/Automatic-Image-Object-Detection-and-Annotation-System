"""
模型训练 API
"""
import os
import sys
import uuid
import json
import yaml
import subprocess
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from app.config import settings, YOLOV5_DIR, DATASET_DIR
from app.models import TrainingConfig, TrainingStatus, EvaluationResult

router = APIRouter()

# 训练任务状态存储
training_tasks: Dict[str, Dict[str, Any]] = {}


def run_training(task_id: str, config: TrainingConfig):
    """后台运行训练任务"""
    global training_tasks
    
    try:
        training_tasks[task_id]["status"] = "running"
        training_tasks[task_id]["message"] = "训练进行中..."
        
        # 构建训练命令
        train_script = YOLOV5_DIR / "train.py"
        
        # 确定权重路径 - 优先使用本地权重文件
        weights_path = config.weights
        if not Path(weights_path).is_absolute():
            # 首先检查 yolov5 根目录
            local_weights = YOLOV5_DIR / config.weights
            if local_weights.exists():
                weights_path = str(local_weights)
            else:
                # 检查 weights 目录
                weights_dir_path = YOLOV5_DIR / "weights" / config.weights
                if weights_dir_path.exists():
                    weights_path = str(weights_dir_path)
                else:
                    # 如果权重文件不存在，记录警告但继续（YOLOv5会尝试下载）
                    training_tasks[task_id]["message"] = f"警告: 本地未找到权重文件 {config.weights}，将尝试下载..."
        
        # 验证权重文件存在
        if not Path(weights_path).exists():
            training_tasks[task_id]["status"] = "failed"
            training_tasks[task_id]["message"] = f"权重文件不存在: {weights_path}，请先下载权重文件到 yolov5/weights 目录"
            return
        
        cmd = [
            sys.executable, str(train_script),
            "--weights", weights_path,
            "--data", config.data_yaml,
            "--epochs", str(config.epochs),
            "--batch-size", str(config.batch_size),
            "--img", str(config.img_size),
            "--project", config.project,
            "--name", config.name,
            "--workers", str(config.workers),
            "--patience", str(config.patience),
            "--optimizer", config.optimizer,
        ]
        
        if config.device:
            cmd.extend(["--device", config.device])
        
        # 执行训练
        process = subprocess.Popen(
            cmd,
            cwd=str(YOLOV5_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        training_tasks[task_id]["process"] = process
        
        # 读取输出并更新进度
        output_lines = []
        for line in process.stdout:
            output_lines.append(line.strip())
            training_tasks[task_id]["output"] = output_lines[-100:]  # 保留最后100行
            
            # 解析进度
            if "Epoch" in line:
                try:
                    # 解析类似 "Epoch 1/100" 的格式
                    parts = line.split("Epoch")[1].strip().split("/")
                    if len(parts) >= 2:
                        current = int(parts[0].split()[0])
                        total = int(parts[1].split()[0])
                        training_tasks[task_id]["current_epoch"] = current
                        training_tasks[task_id]["total_epochs"] = total
                        training_tasks[task_id]["progress"] = (current / total) * 100
                except:
                    pass
        
        process.wait()
        
        if process.returncode == 0:
            training_tasks[task_id]["status"] = "completed"
            training_tasks[task_id]["message"] = "训练完成"
            training_tasks[task_id]["progress"] = 100
            
            # 查找最新的训练结果
            results_dir = YOLOV5_DIR / config.project / config.name
            if results_dir.exists():
                training_tasks[task_id]["results_dir"] = str(results_dir)
        else:
            training_tasks[task_id]["status"] = "failed"
            training_tasks[task_id]["message"] = f"训练失败，返回码: {process.returncode}"
            
    except Exception as e:
        training_tasks[task_id]["status"] = "failed"
        training_tasks[task_id]["message"] = str(e)


@router.post("/start")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """
    启动模型训练
    """
    # 验证数据集配置文件
    if not Path(config.data_yaml).exists():
        # 检查相对于 yolov5 目录
        data_yaml_path = YOLOV5_DIR / config.data_yaml
        if not data_yaml_path.exists():
            data_yaml_path = YOLOV5_DIR / "data" / config.data_yaml
            if not data_yaml_path.exists():
                raise HTTPException(
                    status_code=400, 
                    detail=f"数据集配置文件不存在: {config.data_yaml}"
                )
        config.data_yaml = str(data_yaml_path)
    
    # 创建训练任务
    task_id = str(uuid.uuid4())
    training_tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0,
        "current_epoch": 0,
        "total_epochs": config.epochs,
        "config": config.dict(),
        "created_at": datetime.now().isoformat(),
        "message": "准备开始训练...",
        "output": []
    }
    
    # 后台启动训练
    thread = threading.Thread(target=run_training, args=(task_id, config))
    thread.start()
    
    return {
        "task_id": task_id,
        "message": "训练任务已创建",
        "status": "pending"
    }


@router.get("/status/{task_id}")
async def get_training_status(task_id: str):
    """
    获取训练状态
    """
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    
    task = training_tasks[task_id]
    return TrainingStatus(
        task_id=task["task_id"],
        status=task["status"],
        progress=task.get("progress", 0),
        current_epoch=task.get("current_epoch", 0),
        total_epochs=task.get("total_epochs", 0),
        message=task.get("message"),
        metrics=task.get("metrics")
    )


@router.get("/output/{task_id}")
async def get_training_output(task_id: str):
    """
    获取训练输出日志
    """
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    
    return {
        "task_id": task_id,
        "output": training_tasks[task_id].get("output", [])
    }


@router.post("/stop/{task_id}")
async def stop_training(task_id: str):
    """
    停止训练任务
    """
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    
    task = training_tasks[task_id]
    
    if task.get("process"):
        task["process"].terminate()
        task["status"] = "stopped"
        task["message"] = "训练已手动停止"
        return {"success": True, "message": "训练已停止"}
    
    return {"success": False, "message": "无法停止训练"}


@router.get("/list")
async def list_training_tasks():
    """
    列出所有训练任务
    """
    tasks = []
    for task_id, task in training_tasks.items():
        tasks.append({
            "task_id": task_id,
            "status": task["status"],
            "progress": task.get("progress", 0),
            "created_at": task.get("created_at"),
            "message": task.get("message")
        })
    
    return {"tasks": tasks, "total": len(tasks)}


@router.get("/results/{task_id}")
async def get_training_results(task_id: str):
    """
    获取训练结果
    """
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    
    task = training_tasks[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="训练尚未完成")
    
    results_dir = task.get("results_dir")
    if not results_dir or not Path(results_dir).exists():
        raise HTTPException(status_code=404, detail="训练结果目录不存在")
    
    results_path = Path(results_dir)
    
    # 读取训练结果
    result = {
        "task_id": task_id,
        "results_dir": results_dir,
        "weights": {}
    }
    
    # 检查权重文件
    for weight_file in ["best.pt", "last.pt"]:
        weight_path = results_path / "weights" / weight_file
        if weight_path.exists():
            result["weights"][weight_file] = str(weight_path)
    
    # 检查结果图表
    result["plots"] = []
    for plot_file in results_path.glob("*.png"):
        result["plots"].append(str(plot_file))
    
    # 读取 results.csv
    results_csv = results_path / "results.csv"
    if results_csv.exists():
        import pandas as pd
        df = pd.read_csv(results_csv)
        result["metrics_history"] = df.to_dict(orient="records")
    
    return result


@router.get("/datasets")
async def list_available_datasets():
    """
    列出可用的数据集配置
    使用 datasets 目录下的 coco 和 VOC 数据集，以及自定义数据集
    """
    datasets = []
    seen_names = set()
    
    # 允许的内置数据集列表
    allowed_datasets = ['coco', 'voc']
    
    def get_dataset_info(yaml_file: Path, source: str) -> Optional[dict]:
        """读取数据集配置并检查数据是否存在"""
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 获取数据集路径
            dataset_path = config.get('path', '')
            if not Path(dataset_path).is_absolute():
                # 相对路径，相对于yaml文件所在目录
                dataset_path = str(yaml_file.parent / dataset_path) if dataset_path else str(yaml_file.parent)
            
            dataset_path = Path(dataset_path)
            
            # 检查数据集是否存在
            images_exist = False
            train_count = val_count = 0
            
            if dataset_path.exists():
                images_dir = dataset_path / 'images'
                if images_dir.exists():
                    # 统计图像数量
                    for subdir in images_dir.iterdir():
                        if subdir.is_dir():
                            count = sum(1 for _ in subdir.glob('*.jpg')) + sum(1 for _ in subdir.glob('*.png'))
                            if 'train' in subdir.name.lower():
                                train_count += count
                            elif 'val' in subdir.name.lower():
                                val_count += count
                    images_exist = train_count > 0 or val_count > 0
            
            # 获取类别信息
            names = config.get('names', {})
            if isinstance(names, dict):
                class_names = list(names.values())
            else:
                class_names = names if isinstance(names, list) else []
            
            return {
                "name": yaml_file.stem,
                "path": str(yaml_file),
                "relative_path": yaml_file.name,
                "source": source,
                "dataset_dir": str(dataset_path),
                "exists": images_exist,
                "train_images": train_count,
                "val_images": val_count,
                "num_classes": len(class_names),
                "classes": class_names[:10] if len(class_names) > 10 else class_names,  # 只返回前10个类别
                "has_download": 'download' in config
            }
        except Exception as e:
            return None
    
    # 1. 检查 datasets 目录下的 coco 和 VOC 数据集
    if DATASET_DIR.exists():
        for subdir in DATASET_DIR.iterdir():
            if subdir.is_dir() and subdir.name.lower() in allowed_datasets:
                for yaml_file in subdir.glob("*.yaml"):
                    info = get_dataset_info(yaml_file, 'datasets')
                    if info:
                        datasets.append(info)
                        seen_names.add(yaml_file.stem.lower())
    
    # 2. 检查自定义数据集目录
    custom_dir = Path(settings.CUSTOM_DATASET_DIR)
    if custom_dir.exists():
        for yaml_file in custom_dir.glob("**/*.yaml"):
            if yaml_file.stem.lower() not in seen_names:
                info = get_dataset_info(yaml_file, 'custom')
                if info:
                    datasets.append(info)
                    seen_names.add(yaml_file.stem.lower())
    
    # 按来源和名称排序（内置数据集在前，自定义数据集在后）
    def sort_key(d):
        source_order = {'datasets': 0, 'custom': 1}
        return (source_order.get(d['source'], 2), d['name'].lower())
    
    datasets.sort(key=sort_key)
    
    return {"datasets": datasets}


@router.get("/weights")
async def list_available_weights():
    """
    列出可用的预训练权重文件
    """
    weights = []
    
    # 检查 yolov5 根目录
    for pt_file in YOLOV5_DIR.glob("*.pt"):
        weights.append({
            "name": pt_file.stem,
            "path": str(pt_file),
            "size_mb": round(pt_file.stat().st_size / (1024 * 1024), 2),
            "exists": True
        })
    
    # 检查 weights 目录
    weights_dir = YOLOV5_DIR / "weights"
    if weights_dir.exists():
        for pt_file in weights_dir.glob("*.pt"):
            # 避免重复
            if not any(w['name'] == pt_file.stem for w in weights):
                weights.append({
                    "name": pt_file.stem,
                    "path": str(pt_file),
                    "size_mb": round(pt_file.stat().st_size / (1024 * 1024), 2),
                    "exists": True
                })
    
    # 添加标准权重列表（即使不存在）
    standard_weights = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
    for name in standard_weights:
        if not any(w['name'] == name for w in weights):
            weights.append({
                "name": name,
                "path": f"{name}.pt",
                "size_mb": 0,
                "exists": False
            })
    
    return {"weights": weights}


@router.get("/hyperparameters")
async def list_hyperparameters():
    """
    列出可用的超参数配置
    """
    hyps = []
    
    hyps_dir = YOLOV5_DIR / "data" / "hyps"
    if hyps_dir.exists():
        for yaml_file in hyps_dir.glob("*.yaml"):
            hyps.append({
                "name": yaml_file.stem,
                "path": str(yaml_file)
            })
    
    return {"hyperparameters": hyps}
