# config_runtime.py
import multiprocessing

# 改写比例配置
DEFAULT_REWRITE_RATIO = 40  # 默认改写40%的段落

# 并发设置
# 根据CPU核心数动态调整，默认不超过8个
MAX_WORKERS = min(8, max(1, multiprocessing.cpu_count() - 1))  # 留一个核心给系统
API_RETRY_LIMIT = 3  # API调用重试次数
REQUEST_TIMEOUT = 120  # 增加请求超时时间（秒），给模型更多思考时间

# 批量处理设置
BATCH_SIZE = 5  # 批量处理大小
MAX_CONCURRENT_REQUESTS = 10  # 最大并发请求数

# 评估相关配置
EVALUATION_CONFIG = {
    "max_segment_length": 2000,  # 最大段落长度
    "min_segment_length": 50,    # 最小段落长度
    "enable_detailed_feedback": True  # 是否启用详细反馈
}

# 日志和调试配置
DEBUG_MODE = False  # 调试模式
LOG_LEVEL = "INFO"  # 日志级别
ENABLE_PROGRESS_BAR = True  # 是否启用进度条

# 缓存配置
ENABLE_API_CACHE = True  # 是否启用API缓存
CACHE_EXPIRE_TIME = 3600  # 缓存过期时间（秒）

# 错误处理配置
ENABLE_FALLBACK_MODELS = True  # 是否启用备用模型
MAX_FALLBACK_ATTEMPTS = 2  # 最大备用模型尝试次数

# 性能监控配置
ENABLE_PERFORMANCE_MONITORING = True  # 是否启用性能监控
PERFORMANCE_LOG_FILE = "performance.log"  # 性能日志文件