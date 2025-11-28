# config_models.py
from typing import Dict, Any

# 本地模型配置 (vLLM部署)
LOCAL_MODELS = {
    "qwen-7b": {
        "type": "local",
        "model": "qwen-7b",
        "base_url": "http://localhost:8006/v1",  # vLLM服务器地址
        "api_key": "EMPTY"  # vLLM通常不需要API密钥
    },
    "qwen-4b": {
        "type": "local", 
        "model": "qwen-4b",
        "base_url": "http://localhost:8005/v1",
        "api_key": "EMPTY"
    },
    "qwen-0.5b": {
        "type": "local",
        "model": "qwen-0.5b",
        "base_url": "http://localhost:8003/v1",
        "api_key": "EMPTY"
    },
    "qwen-8b": {
        "type": "local",
        "model": "qwen-8b",
        "base_url": "http://localhost:8003/v1",
        "api_key": "EMPTY"
    },
    "llama3-8b": {
        "type": "local",
        "model": "llama3-8b",
        "base_url": "http://localhost:8002/v1",
        "api_key": "EMPTY"
    },
    "gemma-4bt": {
        "type": "local",
        "model": "gemma-4b",
        "base_url": "http://localhost:8004/v1",
        "api_key": "EMPTY"
    },
    "deepseek-R1-distill": {
        "type": "local",
        "model": "deepseek-R1-distill",
        "base_url": "http://localhost:8003/v1",
        "api_key": "EMPTY"
    },

}

# 第三方API模型配置
THIRD_PARTY_MODELS = {
    "gpt-4o-mini": {
        "type": "third_party",
        "model": "gpt-4o-mini"
    },
    "gpt-4o": {
        "type": "third_party",
        "model": "gpt-4o"
    },
    "deepseek-chat": {
        "type": "third_party", 
        "model": "deepseek-chat"
    },
    "qwen-plus": {
        "type": "third_party",
        "model": "qwen-plus"
    },
    "gpt-3.5-turbo": {
        "type": "third_party",
        "model": "gpt-3.5-turbo"
    },
    "grok-2": {
        "type": "third_party",
        "model": "grok-2"
    },
    "gemini-1.5-flash": {
        "type": "third_party",
        "model": "gemini-1.5-flash"
    }
}

SIMILARITY_MODELS = {
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
        "local_path": "/data1/rjj/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    }
}

# 教师模型配置
TEACHER_MODELS ={
    "gpt-4o-mini": THIRD_PARTY_MODELS["gpt-4o-mini"],
    "gpt-4o": THIRD_PARTY_MODELS["gpt-4o"],
    "deepseek-chat": THIRD_PARTY_MODELS["deepseek-chat"],
    "qwen-plus": THIRD_PARTY_MODELS["qwen-plus"],
    "qwen-4b": LOCAL_MODELS["qwen-4b"],  # 本地部署的教师模型
    # "llama3-70b": LOCAL_MODELS["llama3-70b"]  # 本地部署的教师模型
}

# 学生模型配置
STUDENT_MODELS = {
    "gpt-3.5-turbo": THIRD_PARTY_MODELS["gpt-3.5-turbo"],
    "grok-2": THIRD_PARTY_MODELS["grok-2"],
    "gemini-1.5-flash": THIRD_PARTY_MODELS["gemini-1.5-flash"],
    "qwen-7b": LOCAL_MODELS["qwen-7b"],  # 本地部署的学生模型
    "qwen-8b": LOCAL_MODELS["qwen-8b"], 
    "qwen-0.5b": LOCAL_MODELS["qwen-0.5b"], 
    "qwen-4b": LOCAL_MODELS["qwen-4b"], 
    "llama3-8b": LOCAL_MODELS["llama3-8b"],  # 本地部署的学生模型
    "deepseek-R1-distill":LOCAL_MODELS["deepseek-R1-distill"],
    "gemma-4bt":LOCAL_MODELS["gemma-4bt"]
}

# 默认选择
DEFAULT_TEACHER_MODEL = "gpt-4o"
DEFAULT_STUDENT_MODEL = "qwen-7b"  # 修改为本地模型
DEFAULT_SIMILARITY_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# 模型特定配置
MODEL_CONFIGS = {
    "gpt-4o-mini": {
        "temperature": 0.3,
        "max_tokens": 2000
    },
    "gpt-4o": {
        "temperature": 0.3,
        "max_tokens": 2000
    },
    "gpt-3.5-turbo": {
        "temperature": 0.7,
        "max_tokens": 3000
    },
    "qwen-plus": {
        "temperature": 0.3,
        "max_tokens": 2000
    },
    "qwen-7b": {
        "temperature": 0.9,
        "max_tokens": 3000
    },
    "qwen-14b": {
        "temperature": 0.7,
        "max_tokens": 3000
    },
    "deepseek-chat": {
        "temperature": 0.5,
        "max_tokens": 2500
    },
    "llama3-8b": {
        "temperature": 0.7,
        "max_tokens": 3000
    },
    "llama3-70b": {
        "temperature": 0.3,
        "max_tokens": 2000
    },
    "grok-2": {
        "temperature": 0.7,
        "max_tokens": 3000
    },
    "gemini-1.5-flash": {
        "temperature": 0.7,
        "max_tokens": 3000
    },
    "gpt3-text": {
        "temperature": 0.7,
        "max_tokens": 3000
    }
}

# CUDA_VISIBLE_DEVICES=4 \
# python -m vllm.entrypoints.openai.api_server \
#     --model /data1/rjj/models/Qwen/Qwen2.5-7B-Instruct \
#     --trust-remote-code \
#     --dtype half \
#     --enforce-eager\
#     --gpu-memory-utilization 0.5 \
#     --max-model-len 4096 \
#     --port 8005
# CUDA_VISIBLE_DEVICES=4 \
# python -m vllm.entrypoints.openai.api_server \
#     --model /data1/rjj/models/Qwen/Qwen2.5-7B-Instruct \
#     --served-model-name qwen-7b \  
#     --dtype half \
#     --enforce-eager \
#     --gpu-memory-utilization 0.8 \
#     --max-model-len 32768 \
#     --port 8006
# /data1/rjj/models/Qwen/Qwen3-4B-Instruct-2507
# /data1/rjj/models/Qwen/Qwen3-8B
# /data1/rjj/models/Qwen/Qwen1.5-0.5B-Chat
# /data1/rjj/models/Meta-Llama-3___1-8B
# /data1/rjj/models/AI-ModelScope/gemini-nano

# CUDA_VISIBLE_DEVICES=6 \
# python -m vllm.entrypoints.openai.api_server \
#     --model /data1/rjj/models/Meta-Llama-3___1-8B\
#     --served-model-name llama3-8b \  
#     --dtype half \
#     --enforce-eager \
#     --gpu-memory-utilization 0.8 \
#     --max-model-len 4096 \
#     --port 8002


