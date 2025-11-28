# config.py
import os
from itertools import cycle
from typing import Dict, Any, List

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
# CoT验证置信度阈值配置
COT_CONFIDENCE_THRESHOLD = 0.5  # 置信度阈值，高于此值才认为是有效纠正
LOW_CONFIDENCE_THRESHOLD = 0.4  # 低置信度阈值，低于此值需要特别关注
# 导入各个子配置
from config_models import MODEL_CONFIGS, TEACHER_MODELS, STUDENT_MODELS, DEFAULT_TEACHER_MODEL, DEFAULT_STUDENT_MODEL
from config_api import THIRD_PARTY_CONFIGS, API_KEY_CYCLER, get_api_config
from config_runtime import (
    DEFAULT_REWRITE_RATIO, MAX_WORKERS, API_RETRY_LIMIT, REQUEST_TIMEOUT,
    BATCH_SIZE, MAX_CONCURRENT_REQUESTS, EVALUATION_CONFIG, DEBUG_MODE,
    LOG_LEVEL, ENABLE_PROGRESS_BAR, ENABLE_API_CACHE, CACHE_EXPIRE_TIME,
    ENABLE_FALLBACK_MODELS, MAX_FALLBACK_ATTEMPTS, ENABLE_PERFORMANCE_MONITORING,
    PERFORMANCE_LOG_FILE
)

# 合并所有模型配置
ALL_MODELS = {**TEACHER_MODELS, **STUDENT_MODELS}

def get_model_config(model_name: str) -> Dict[str, Any]:
    """获取模型特定配置"""
    return MODEL_CONFIGS.get(model_name, {
        "temperature": 0.7,
        "max_tokens": 2000
    })

def get_model_deployment_config(model_name: str) -> Dict[str, Any]:
    """获取模型部署配置"""
    if model_name not in ALL_MODELS:
        raise ValueError(f"未知模型: {model_name}")
    
    model_config = ALL_MODELS[model_name].copy()
    
    # 如果是第三方API模型，添加API配置
    if model_config["type"] == "third_party":
        api_config = get_api_config()
        model_config.update(api_config)
    
    return model_config

def get_orchestrator_config() -> Dict[str, Any]:
    """获取协调器配置"""
    return {
        "orchestrator": get_model_deployment_config(DEFAULT_TEACHER_MODEL),
        "teacher_model": get_model_deployment_config(DEFAULT_TEACHER_MODEL),
        "student_model": get_model_deployment_config(DEFAULT_STUDENT_MODEL)
    }