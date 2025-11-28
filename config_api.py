# config_api.py
from itertools import cycle

# 多个第三方代理配置
THIRD_PARTY_CONFIGS = [
    {
        "api_key": "sk-03vRYBZjK3GCwyEZFeE30bC507C148808e67A4CdDaDf5440",
        "base_url": "https://api.ai-gaochao.cn/v1"
    },
    {
        "api_key": "sk-xlEiCHw5ePYZTNUNCb5b58F06fC042B7Ad41A532854fE6Ea",
        "base_url": "https://api.ai-gaochao.cn/v1"
    },

    {
        "api_key": "sk-nO1mpGZCcjCccgMO308c3d66F57f40EcB1Aa34B0D170FaB5",
        "base_url": "https://api.ai-gaochao.cn/v1"
    },
    {
        "api_key": "sk-ptqHOAZWI7fCmKCR2152C6F8E55945C997E0B925248eC8Da",
        "base_url": "https://api.ai-gaochao.cn/v1"
    },
    {
        "api_key": "sk-VJEjQ3pWAzk2b829E3B16aEa433943A48a84F2FbA2Fb1822",
        "base_url": "https://api.ai-gaochao.cn/v1"
    }

]

# 创建API密钥循环器
API_KEY_CYCLER = cycle(THIRD_PARTY_CONFIGS)

def get_api_config():
    """获取下一个API配置"""
    return next(API_KEY_CYCLER)