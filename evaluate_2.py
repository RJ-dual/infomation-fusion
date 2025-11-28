from typing import List, Dict, Any, Optional, Tuple
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import re
import json
import logging
from collections import Counter, defaultdict
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import os
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
from sentence_transformers import SentenceTransformer
import torch

# 导入配置
from config import (
    get_model_deployment_config,
    DEFAULT_TEACHER_MODEL,
    MAX_WORKERS,
    REQUEST_TIMEOUT,
    LOG_LEVEL,
    DEBUG_MODE,
    ENABLE_PROGRESS_BAR,
    API_RETRY_LIMIT,
    COT_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    BATCH_SIZE,
)

# 设置日志
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# 修复jieba权限问题
jieba_cache_dir = os.path.join(os.path.expanduser("~"), ".jieba_cache")
os.makedirs(jieba_cache_dir, exist_ok=True)
jieba.dt.cache_file = os.path.join(jieba_cache_dir, "jieba.cache")
if DEBUG_MODE:
    logger.info(f"🔧 设置jieba缓存目录: {jieba_cache_dir}")

# ==================== 问题分类定义 ====================

# 第一类：整体文本问题（Global Problems）
GLOBAL_PROBLEMS = {
    "逻辑连贯性断裂": {
        "description": "描述灾难事件发展过程时，句子或段落间缺乏必要逻辑连接，导致信息支离破碎，难以理解事件全貌。",
        "evaluation_method": "通读完整融合文本，检查段落间逻辑连接"
    },
    "语言风格一致性错误": {
        "description": "融合灾难报道时，文体、语态或用语风格发生突然变化，在正式官方通报与口语化表达之间跳跃，破坏文本整体性。",
        "evaluation_method": "检查全文语言风格是否统一"
    },
    "文本结构组织混乱": {
        "description": "融合灾难信息时，信息组织方式杂乱无章，同类信息被分散在不同部分，缺乏清晰逻辑结构和层次划分。",
        "evaluation_method": "评估整体信息组织结构和层次"
    },
    "时间顺序准确性错误": {
        "description": "融合灾难事件时间线信息时，错误排列事件发生的先后顺序。",
        "evaluation_method": "参照原始文本进行检查，检查融合后的文本的时间线逻辑是否和原始文本一致。"
    },
    "因果关系逻辑错误": {
        "description": "融合灾难信息时，擅自添加或颠倒事件间的因果关系。",
        "evaluation_method": "检查全文因果关系是否合理"
    }
}

# 第二类：句子级别问题（Sentence-Level Problems）
SENTENCE_LEVEL_PROBLEMS = {
    "事实完整性缺失": {
        "description": "在融合灾难事件信息时，未能包含原文本中存在的关键事实细节，如受灾范围、伤亡人数、灾害等级或救援进展，导致信息不完整，影响对灾情的全面理解。",
        "evaluation_method": "对比句子evidence与融合文本，检查关键事实是否缺失"
    },
    "核心数据准确性错误": {
        "description": "在融合灾难关键数据时出现数值错误，包括伤亡人数、经济损失金额、震级、风速等核心指标，导致对灾情严重程度的误判。",
        "evaluation_method": "检查数字、时间等核心数据是否准确"
    },
    "信息来源归属错误": {
        "description": "在融合信息时，错误地归属或混淆信息的原始来源机构、个人或出处，包括错误指定发言主体、错误引用权威机构、或错误标注信息出处。",
        "evaluation_method": "检查人物、机构等归属是否正确"
    },
    "虚构内容生成": {
        "description": "在融合灾难信息时，生成与原始文本无关或矛盾的虚构内容，包括编造未发生的灾难、虚构重大伤亡或不存在的救援行动。",
        "evaluation_method": "检查是否存在无来源支持的内容"
    },
    "信息表述精确性不足": {
        "description": "融合信息时，使用模糊、不确定的词语替代原本具体、明确的信息表述，导致关键事实细节丢失。",
        "evaluation_method": "检查具体信息是否被模糊化"
    },
    "分类层级不当": {
        "description": "融合信息时，不恰当地改变信息在分类体系中的层级位置，主要表现为将细粒度的具体类别归纳为粗粒度的上位类别，或错误地进行类别映射，导致分类信息失真。",
        "evaluation_method": "检查分类概念是否被错误提升或降低"
    },
    "内容客观性偏差": {
        "description": "融合灾难信息时，添加不必要的、带有强烈个人或机构倾向性的评价性语言和主观判断，偏离原文本的立场，影响信息客观性。",
        "evaluation_method": "检查是否添加主观评价"
    },
    "冲突信息处理不当": {
        "description": "在融合多个来源信息时，未能识别或妥善处理来源间的明显冲突信息，包括关键事实矛盾、数据不一致、时间冲突等，导致融合结果包含未解决的矛盾或错误采纳了冲突信息。",
        "evaluation_method": "检查conflict_resolved字段是否充分处理冲突,如认为存在冲突，必须输出冲突证据"
    }
}

# ==================== 评估提示模板 ====================

GLOBAL_EVALUATE_PROMPT = """
作为专业新闻编辑，请评估以下融合文本是否存在{problem_type}问题。

## 问题描述: {problem_description}

## 评估要求:
1. 仔细阅读完整融合文本，评估整体质量
2. 判断是否存在{problem_type}问题
3. 如果存在，请提供具体证据
4. 如果不存在，返回空结果

## 文本内容:
**原始文本**: {original_text}
**融合文本**: {fused_text}

## 输出格式: 请以JSON格式返回评估结果，包含以下字段:
- "problem_exists": 布尔值，表示是否存在该问题
- "evidence": 具体证据（如果存在问题）

请开始评估：
"""

SENTENCE_EVALUATE_PROMPT = """
作为专业新闻编辑，请评估以下融合句子是否存在{problem_type}问题。

## 问题描述: {problem_description}

## 评估要求:
1. 仔细对比融合句子与原始证据
2. 判断是否存在{problem_type}问题
3. 如果存在，请提供具体证据
4. 如果不存在，返回空结果

## 句子信息:
**融合句子**: {fused_sentence}
**原始证据**: {evidence_texts}

## 对齐信息:
- 来源版本: {sources}
- 冲突处理: {conflict_resolved}

## 输出格式: 请以JSON格式返回评估结果，包含以下字段:
- "problem_exists": 布尔值，表示是否存在该问题
- "evidence": 具体证据（如果存在问题）

请开始评估：
"""

class TextSimilarityCalculator:
    """文本相似度计算器"""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", local_model_path: str = None, gpu_device: str = "cuda:0"):
        """ 初始化相似度计算器
        Args:
            model_name: 使用的句子转换模型名称
            local_model_path: 本地模型路径，如果提供则优先使用本地模型
            gpu_device: 指定使用的GPU设备，如 "cuda:0", "cuda:1" 等
        """
        self.model_name = model_name
        self.local_model_path = local_model_path
        self.gpu_device = gpu_device
        self.sentence_model = None
        self.tfidf_vectorizer = None
        # 初始化模型
        self._initialize_models()

    def _initialize_models(self):
        """初始化模型"""
        try:
            import torch
            # 设置设备
            if self.gpu_device and torch.cuda.is_available():
                # 检查指定的GPU是否可用
                gpu_id = int(self.gpu_device.split(":")[1])
                if gpu_id < torch.cuda.device_count():
                    self.device = torch.device(self.gpu_device)
                    logger.info(f"✅ 使用设备: {self.device}")
                else:
                    logger.warning(f"⚠️ 指定的GPU设备 {self.gpu_device} 不可用，可用的GPU数量: {torch.cuda.device_count()}，将使用CPU")
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"✅ 自动选择设备: {self.device}")
            # 初始化句子转换模型 - 优先使用本地模型
            if self.local_model_path and os.path.exists(self.local_model_path):
                self.sentence_model = SentenceTransformer(self.local_model_path, device=self.device)
                logger.info(f"✅ 从本地路径加载句子转换模型: {self.local_model_path}")
            else:
                self.sentence_model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"✅ 句子转换模型 {self.model_name} 初始化成功")
        except Exception as e:
            logger.warning(f"⚠️ 句子转换模型初始化失败: {e}, 将使用TF-IDF")
            self.sentence_model = None
        # 初始化TF-IDF向量化器
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=self._tokenize_chinese, min_df=1, max_df=0.8, ngram_range=(1, 2)
        )

    def _tokenize_chinese(self, text: str) -> List[str]:
        """中文分词"""
        return list(jieba.cut(text))

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """ 计算语义相似度（使用句子转换模型） """
        if not text1.strip() or not text2.strip():
            return 0.0
        try:
            if self.sentence_model is not None:
                embeddings = self.sentence_model.encode([text1, text2])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                return float(similarity)
            else:
                return self.calculate_tfidf_similarity(text1, text2)
        except Exception as e:
            logger.error(f"❌ 语义相似度计算失败: {e}")
            return self.calculate_tfidf_similarity(text1, text2)

    def calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """ 计算TF-IDF相似度 """
        if not text1.strip() or not text2.strip():
            return 0.0
        try:
            corpus = [text1, text2]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"❌ TF-IDF相似度计算失败: {e}")
            return 0.0

    def calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """ 计算Jaccard相似度 """
        if not text1.strip() or not text2.strip():
            return 0.0
        try:
            words1 = set(self._tokenize_chinese(text1))
            words2 = set(self._tokenize_chinese(text2))
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return intersection / union if union > 0 else 0.0
        except Exception as e:
            logger.error(f"❌ Jaccard相似度计算失败: {e}")
            return 0.0

    def calculate_levenshtein_similarity(self, text1: str, text2: str) -> float:
        """ 计算基于编辑距离的相似度 """
        if not text1.strip() or not text2.strip():
            return 0.0
        try:
            if text1 == text2:
                return 1.0
            len1, len2 = len(text1), len(text2)
            max_len = max(len1, len2)
            if max_len == 0:
                return 1.0
            distance = self._levenshtein_distance(text1, text2)
            return 1.0 - (distance / max_len)
        except Exception as e:
            logger.error(f"❌ 编辑距离相似度计算失败: {e}")
            return 0.0

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """计算Levenshtein编辑距离"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def calculate_comprehensive_similarity(self, text1: str, text2: str) -> Dict[str, float]:
        """ 计算综合相似度（多种方法） """
        return {
            "semantic_similarity": self.calculate_semantic_similarity(text1, text2),
            "tfidf_similarity": self.calculate_tfidf_similarity(text1, text2),
            "jaccard_similarity": self.calculate_jaccard_similarity(text1, text2),
            "levenshtein_similarity": self.calculate_levenshtein_similarity(text1, text2),
        }

    def calculate_fused_text_similarities(self, news_item: Dict[str, Any]) -> Dict[str, Any]:
        """ 计算融合文本与原始文本及改写版本的相似度 """
        try:
            fused_content = news_item.get("fused_content", "")
            original_content = news_item.get("input_text", "")
            if not fused_content.strip() or not original_content.strip():
                logger.warning("❌ 融合文本或原始文本为空")
                return self._create_empty_similarity_result()
            similarities = {
                "original": self.calculate_comprehensive_similarity(fused_content, original_content)
            }
            rewrite_versions = {}
            for i in range(1, 4):
                rewrite_key_v = f"rewritten_v{i}"
                rewrite_key_n = f"rewrite_{i}"
                if rewrite_key_v in news_item and news_item[rewrite_key_v]:
                    rewrite_versions[rewrite_key_v] = news_item[rewrite_key_v]
                elif rewrite_key_n in news_item and news_item[rewrite_key_n]:
                    rewrite_versions[rewrite_key_n] = news_item[rewrite_key_n]
            for version_key, rewrite_content in rewrite_versions.items():
                if rewrite_content and rewrite_content.strip():
                    try:
                        similarities[version_key] = self.calculate_comprehensive_similarity(fused_content, rewrite_content)
                    except Exception as e:
                        logger.warning(f"⚠️ 计算与{version_key}的相似度失败: {e}")
                        similarities[version_key] = {"comprehensive_score": 0.0, "calculation_success": False}
            avg_similarities = self._calculate_average_similarities(similarities)
            analysis = self._generate_similarity_analysis(similarities, avg_similarities)
            return {
                "similarities": similarities,
                "average_similarities": avg_similarities,
                "analysis": analysis,
                "similarity_calculation_success": True,
            }
        except Exception as e:
            logger.error(f"❌ 相似度计算失败: {e}")
            return self._create_empty_similarity_result()

    def _calculate_average_similarities(self, similarities: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """计算平均相似度"""
        avg_similarities = {}
        methods = ["semantic_similarity", "tfidf_similarity", "jaccard_similarity", "levenshtein_similarity"]
        for method in methods:
            scores = []
            for version_similarities in similarities.values():
                if method in version_similarities:
                    scores.append(version_similarities[method])
            if scores:
                avg_similarities[method] = sum(scores) / len(scores)
            else:
                avg_similarities[method] = 0.0
        return avg_similarities

    def _generate_similarity_analysis(self, similarities: Dict[str, Dict[str, float]], avg_similarities: Dict[str, float]) -> Dict[str, Any]:
        """生成相似度分析报告"""
        analysis = {"overall_assessment": "", "key_observations": [], "recommendations": []}
        semantic_avg = avg_similarities.get("semantic_similarity", 0)
        if semantic_avg >= 0.8:
            analysis["overall_assessment"] = "融合文本与原始文本及改写版本高度相似，信息保留完整"
        elif semantic_avg >= 0.6:
            analysis["overall_assessment"] = "融合文本与原始文本及改写版本中等相似，主要信息得到保留"
        elif semantic_avg >= 0.4:
            analysis["overall_assessment"] = "融合文本与原始文本及改写版本相似度较低，存在信息丢失"
        else:
            analysis["overall_assessment"] = "融合文本与原始文本及改写版本相似度很低，信息保留不完整"
        if "original" in similarities:
            orig_semantic = similarities["original"].get("semantic_similarity", 0)
            if orig_semantic < 0.5:
                analysis["key_observations"].append("融合文本与原始文本语义相似度较低，可能存在重要信息遗漏")
        method_scores = list(avg_similarities.values())
        if len(method_scores) >= 2:
            score_variance = np.var(method_scores)
            if score_variance > 0.1:
                analysis["key_observations"].append("不同相似度计算方法结果差异较大，建议关注语义相似度")
        if semantic_avg < 0.6:
            analysis["recommendations"].append("建议检查融合文本是否保留了原始文本的关键信息")
        if avg_similarities.get("jaccard_similarity", 0) < 0.3:
            analysis["recommendations"].append("词汇重叠度较低，建议检查是否使用了过多不同的表达方式")
        return analysis

    def _create_empty_similarity_result(self) -> Dict[str, Any]:
        """创建空的相似度结果"""
        return {
            "similarities": {},
            "average_similarities": {},
            "analysis": {
                "overall_assessment": "相似度计算失败",
                "key_observations": ["无法计算相似度"],
                "recommendations": ["请检查输入文本"],
            },
            "similarity_calculation_success": False,
        }

class DualEvaluationSystem:
    """双重评估系统 - 支持整体文本评估和句子级别评估"""

    def __init__(self, strict_model_config: Dict[str, Any], cot_model_config: Dict[str, Any] = None, local_similarity_model_path: str = None, gpu_device: str = "cuda:0"):
        self.strict_model_config = strict_model_config
        self.cot_model_config = cot_model_config if cot_model_config else strict_model_config.copy()
        self.strict_model_name = strict_model_config.get("model", "unknown")
        self.cot_model_name = self.cot_model_config.get("model", "unknown")
        self.retry_limit = API_RETRY_LIMIT
        self.gpu_device = gpu_device
        
        if DEBUG_MODE:
            logger.info(f"🤖 初始化双重评估系统，严格模型: {self.strict_model_name}, CoT模型: {self.cot_model_name}")
            logger.info(f"🎯 指定GPU设备: {gpu_device}")
            
        self.llm_strict = self._initialize_llm(strict_model_config)
        self.llm_cot = self._initialize_llm(self.cot_model_config)
        self.similarity_calculator = TextSimilarityCalculator(local_model_path=local_similarity_model_path, gpu_device=gpu_device)

    def _initialize_llm(self, model_config: Dict[str, Any]) -> ChatOpenAI:
        """初始化语言模型"""
        temperature = model_config.get("temperature", 0.7)
        max_tokens = model_config.get("max_tokens", 4000)
        model_name = model_config.get("model_name", model_config.get("model", "gpt-4"))
        api_key = model_config.get("api_key", "")
        base_url = model_config.get("base_url", "https://api.openai.com/v1")
        
        if api_key == "EMPTY" or api_key == "vllm":
            api_key = "vllm"
            if not base_url or base_url == "https://api.openai.com/v1":
                base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
            logger.info(f"🔧 使用vllm本地模型: {model_name}, 地址: {base_url}")
            
        if DEBUG_MODE:
            logger.info(f"🔧 初始化LLM: 模型={model_name}, 温度={temperature}, 最大令牌={max_tokens}, 基础URL={base_url}")
            
        return ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=api_key,
            openai_api_base=base_url,
            request_timeout=REQUEST_TIMEOUT,
            max_retries=self.retry_limit,
        )

    def safe_llm_call(self, messages, llm_instance=None, max_retries=3, base_delay=1):
        """安全的LLM调用"""
        llm = llm_instance if llm_instance else self.llm_strict
        for attempt in range(max_retries + 1):
            try:
                response = llm.invoke(messages)
                return response
            except Exception as e:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    if DEBUG_MODE:
                        logger.warning(f"⚠️ API调用失败，第{attempt + 1}次重试")
                    time.sleep(delay)
                else:
                    model_name = self.strict_model_name if llm_instance is None else self.cot_model_name
                    logger.error(f"❌ {model_name} API调用失败")
        return None

    def safe_json_parse(self, response_text: str) -> Dict[str, Any]:
        """安全解析JSON响应"""
        if isinstance(response_text, dict):
            return response_text
        if isinstance(response_text, str):
            try:
                return json.loads(response_text)
            except:
                json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group())
                    except:
                        pass
        return {"problem_exists": False, "evidence": ""}

    # ==================== 整体文本评估方法 ====================

    def evaluate_global_problem(self, problem_type: str, original_text: str, fused_text: str) -> Dict[str, Any]:
        """评估整体文本问题"""
        try:
            problem_info = GLOBAL_PROBLEMS.get(problem_type, {})
            description = problem_info.get("description", "")
            
            prompt = GLOBAL_EVALUATE_PROMPT.format(
                problem_type=problem_type,
                problem_description=description,
                original_text=original_text,
                fused_text=fused_text,
            )
            
            messages = [
                SystemMessage(content="你是一名专业的新闻编辑，负责评估文本融合的整体质量。"),
                HumanMessage(content=prompt),
            ]
            
            response = self.safe_llm_call(messages, llm_instance=self.llm_strict)
            if not response:
                return {"problem_exists": False, "problem_type": problem_type, "evidence": "评估失败"}
                
            eval_data = self.safe_json_parse(response.content)
            problem_exists = eval_data.get("problem_exists", False)
            evidence = eval_data.get("evidence", "")
            
            return {
                "problem_exists": problem_exists, 
                "problem_type": problem_type, 
                "evidence": evidence,
                "evaluation_method": "global_assessment"
            }
            
        except Exception as e:
            logger.error(f"❌ 整体评估失败 {problem_type}: {e}")
            return {"problem_exists": False, "problem_type": problem_type, "evidence": f"异常: {e}"}

    def global_evaluation_stage(self, original_text: str, fused_text: str) -> Dict[str, Any]:
        """阶段1: 整体文本评估"""
        logger.info("🌐 阶段1: 整体文本评估")
        if not original_text.strip() or not fused_text.strip():
            return {"global_errors": [], "total_global_problems": 0}
            
        global_errors = []
        
        # 并行对每个整体问题类型进行评估
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(GLOBAL_PROBLEMS))) as executor:
            future_to_problem = {
                executor.submit(self.evaluate_global_problem, problem_type, original_text, fused_text): problem_type
                for problem_type in GLOBAL_PROBLEMS.keys()
            }
            
            for future in as_completed(future_to_problem):
                problem_type = future_to_problem[future]
                try:
                    result = future.result()
                    if result.get("problem_exists", False):
                        global_errors.append({
                            "error_type": result["problem_type"],
                            "evidence": result.get("evidence", ""),
                            "original_text": original_text,
                            "fused_text": fused_text,
                        })
                except Exception as e:
                    logger.error(f"❌ 整体问题类型评估异常: {problem_type}, {e}")
                    
        return {"global_errors": global_errors, "total_global_problems": len(global_errors)}

    # ==================== 句子级别评估方法 ====================

    def evaluate_sentence_problem(self, problem_type: str, sentence_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估句子级别问题"""
        try:
            problem_info = SENTENCE_LEVEL_PROBLEMS.get(problem_type, {})
            description = problem_info.get("description", "")
            
            fused_sentence = sentence_data.get("text", "")
            evidence_texts = sentence_data.get("evidence", [])
            sources = sentence_data.get("sources", [])
            conflict_resolved = sentence_data.get("conflict_resolved", "未处理")
            

            prompt = SENTENCE_EVALUATE_PROMPT.format(
                problem_type=problem_type,
                problem_description=description,
                fused_sentence=fused_sentence,
                evidence_texts=json.dumps(evidence_texts, ensure_ascii=False, indent=2),
                sources=sources,
                conflict_resolved=conflict_resolved,
            )
            
            messages = [
                SystemMessage(content="你是一名专业的新闻编辑，负责评估句子级别的融合质量。"),
                HumanMessage(content=prompt),
            ]
            
            response = self.safe_llm_call(messages, llm_instance=self.llm_strict)
            if not response:
                return {"problem_exists": False, "problem_type": problem_type, "evidence": "评估失败"}
                
            eval_data = self.safe_json_parse(response.content)
            problem_exists = eval_data.get("problem_exists", False)
            evidence = eval_data.get("evidence", "")
            
            return {
                "problem_exists": problem_exists, 
                "problem_type": problem_type, 
                "evidence": evidence,
                "sentence_idx": sentence_data.get("sent_idx"),
                "evaluation_method": "sentence_assessment"
            }
            
        except Exception as e:
            logger.error(f"❌ 句子评估失败 {problem_type}: {e}")
            return {"problem_exists": False, "problem_type": problem_type, "evidence": f"异常: {e}"}

    
    def sentence_evaluation_stage(self, alignment_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """阶段2: 句子级别评估"""
        logger.info("📝 阶段2: 句子级别评估")
        if not alignment_data:
            return {"sentence_errors": [], "total_sentence_problems": 0}
            
        sentence_errors = []
        evaluation_tasks = []
        
        # 为每个句子的每个问题类型创建评估任务
        for sentence in alignment_data:
            for problem_type in SENTENCE_LEVEL_PROBLEMS.keys():
                evaluation_tasks.append({
                    "problem_type": problem_type,
                    "sentence_data": sentence
                })
        
        # 并行执行句子级别评估
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(evaluation_tasks))) as executor:
            future_to_task = {
                executor.submit(self.evaluate_sentence_problem, task["problem_type"], task["sentence_data"]): task
                for task in evaluation_tasks
            }
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    if result.get("problem_exists", False):
                        sentence_errors.append({
                            "error_type": result["problem_type"],
                            "evidence": result.get("evidence", ""),
                            "sentence_idx": result.get("sentence_idx"),
                            "sentence_text": task["sentence_data"].get("text", ""),
                            "evaluation_method": result.get("evaluation_method", "")
                        })
                except Exception as e:
                    logger.error(f"❌ 句子级别评估异常: {e}")
                    
        return {"sentence_errors": sentence_errors, "total_sentence_problems": len(sentence_errors)}

    # ==================== 相似度分析 ====================

    def similarity_analysis_stage(self, news_item: Dict[str, Any]) -> Dict[str, Any]:
        """阶段3: 相似度分析"""
        logger.info("🔍 阶段3: 相似度分析")
        try:
            similarity_result = self.similarity_calculator.calculate_fused_text_similarities(news_item)
            return similarity_result
        except Exception as e:
            logger.error(f"❌ 相似度分析失败: {e}")
            return self.similarity_calculator._create_empty_similarity_result()

    # ==================== 主评估流程 ====================

    def extract_alignment_data(self, fused_content: str) -> List[Dict[str, Any]]:
        """从融合内容中提取对齐数据"""
        try:
            # 查找ALIGNMENT分隔符
            alignment_separator = "====ALIGNMENT===="
            if alignment_separator in fused_content:
                parts = fused_content.split(alignment_separator)
                if len(parts) > 1:
                    alignment_text = parts[1].strip()
                    # 解析JSON数组
                    alignment_data = json.loads(alignment_text)
                    return alignment_data
            return []
        except Exception as e:
            logger.error(f"❌ 提取对齐数据失败: {e}")
            return []

    def dual_evaluation(self, news_item: Dict[str, Any]) -> Dict[str, Any]:
        """双重评估主流程"""
        logger.info("🔄 开始双重评估（整体文本 + 句子级别）")
        
        original_text = news_item.get("input_text", "")
        fused_text = news_item.get("fused_content", "")
        
        if not original_text.strip() or not fused_text.strip():
            logger.warning(f"❌ 新闻内容为空，无法评估")
            return {"evaluation_success": False, "error": "原始文本或融合文本为空"}
        
        # 阶段1: 整体文本评估
        global_results = self.global_evaluation_stage(original_text, fused_text)
        
        # 阶段2: 句子级别评估
        alignment_data = self.extract_alignment_data(fused_text)
        sentence_results = self.sentence_evaluation_stage(alignment_data)
        
        # 阶段3: 相似度分析
        similarity_results = self.similarity_analysis_stage(news_item)
        
        # 生成分类统计
        global_classification = self._classify_global_problems(global_results["global_errors"])
        sentence_classification = self._classify_sentence_problems(sentence_results["sentence_errors"])
        
        result = {
            "evaluation_success": True,
            "global_evaluation": global_results,
            "sentence_evaluation": sentence_results,
            "similarity_analysis": similarity_results,
            "problem_classification": {
                "global_problems": global_classification,
                "sentence_problems": sentence_classification,
                "total_problems": global_results["total_global_problems"] + sentence_results["total_sentence_problems"]
            }
        }
        
        logger.info(f"✅ 双重评估完成:")
        logger.info(f" 整体文本问题: {global_results['total_global_problems']}个")
        logger.info(f" 句子级别问题: {sentence_results['total_sentence_problems']}个")
        logger.info(f" 总问题数: {result['problem_classification']['total_problems']}个")
        
        if similarity_results.get("similarity_calculation_success", False):
            avg_similarities = similarity_results.get("average_similarities", {})
            semantic_sim = avg_similarities.get("semantic_similarity", 0)
            logger.info(f" 平均语义相似度: {semantic_sim:.3f}")
            
        return result

    def _classify_global_problems(self, global_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分类整体文本问题"""
        classification = defaultdict(int)
        for error in global_errors:
            error_type = error.get("error_type", "未知类型")
            classification[error_type] += 1
            
        return {
            "by_type": dict(classification),
            "total_count": len(global_errors)
        }

    def _classify_sentence_problems(self, sentence_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分类句子级别问题"""
        classification = defaultdict(int)
        for error in sentence_errors:
            error_type = error.get("error_type", "未知类型")
            classification[error_type] += 1
            
        return {
            "by_type": dict(classification),
            "total_count": len(sentence_errors)
        }

# ==================== 评估处理器 ====================

def create_dual_evaluator(strict_model_name: str = None, cot_model_name: str = None, local_similarity_model_path: str = None, gpu_device: str = "cuda:0"):
    """创建双重评估器"""
    if strict_model_name is None:
        strict_model_name = DEFAULT_TEACHER_MODEL
    if cot_model_name is None:
        cot_model_name = strict_model_name
        
    strict_model_config = get_model_deployment_config(strict_model_name)
    cot_model_config = get_model_deployment_config(cot_model_name)
    
    return DualEvaluationSystem(strict_model_config, cot_model_config, local_similarity_model_path=local_similarity_model_path, gpu_device=gpu_device)

class DualEvaluationProcessor:
    """双重评估处理器"""

    def __init__(self, strict_model_name: str = None, cot_model_name: str = None, local_similarity_model_path: str = None, gpu_device: str = "cuda:0"):
        self.evaluator = create_dual_evaluator(strict_model_name, cot_model_name, local_similarity_model_path=local_similarity_model_path, gpu_device=gpu_device)
        self.max_workers = max(1, MAX_WORKERS // 2)
        self.batch_size = max(1, BATCH_SIZE // 2)

    def process_single_news(self, news_item: Dict[str, Any]) -> Dict[str, Any]:
        """处理单篇新闻的双重评估"""
        try:
            evaluation_result = self.evaluator.dual_evaluation(news_item)
            return self._create_evaluated_news(news_item, evaluation_result)
        except Exception as e:
            logger.error(f"❌ 双重评估失败: {e}")
            return self._create_evaluated_news(news_item, {"error": f"评估异常: {str(e)}", "evaluation_success": False})

    def _create_evaluated_news(self, original_news: Dict[str, Any], evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """创建包含评估结果的新闻数据"""
        evaluated_news = original_news.copy()
        evaluated_news.update({
            "dual_evaluation": evaluation_result,
            "evaluation_success": evaluation_result.get("evaluation_success", False),
        })
        
        if evaluation_result.get("evaluation_success", False):
            evaluated_news.update({
                "global_problem_count": evaluation_result["global_evaluation"]["total_global_problems"],
                "sentence_problem_count": evaluation_result["sentence_evaluation"]["total_sentence_problems"],
                "total_problem_count": evaluation_result["problem_classification"]["total_problems"],
            })
            
        similarity_analysis = evaluation_result.get("similarity_analysis", {})
        if similarity_analysis.get("similarity_calculation_success", False):
            avg_similarities = similarity_analysis.get("average_similarities", {})
            evaluated_news.update({
                "semantic_similarity": avg_similarities.get("semantic_similarity", 0),
                "tfidf_similarity": avg_similarities.get("tfidf_similarity", 0),
                "jaccard_similarity": avg_similarities.get("jaccard_similarity", 0),
            })
            
        return evaluated_news

    def process_news_batch(self, news_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量处理双重评估"""
        logger.info(f"开始批量双重评估 {len(news_list)} 篇新闻")
        processed_news = []
        success_count = 0
        total_global_problems = 0
        total_sentence_problems = 0
        total_semantic_similarity = 0
        
        batch_size = min(self.batch_size, 2)
        
        for batch_start in range(0, len(news_list), batch_size):
            batch_end = min(batch_start + batch_size, len(news_list))
            batch = news_list[batch_start:batch_end]
            logger.info(f"🔄 双重评估批次 {batch_start//batch_size + 1}: {len(batch)} 篇新闻")
            
            batch_results = []
            pbar = tqdm(total=len(batch), desc=f"双重评估批次 {batch_start//batch_size + 1}") if ENABLE_PROGRESS_BAR else None
            
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batch))) as executor:
                future_to_news = {executor.submit(self.process_single_news, news): news for news in batch}
                
                for future in as_completed(future_to_news):
                    news_item = future_to_news[future]
                    try:
                        result = future.result()
                        batch_results.append(result)
                        
                        if result.get("evaluation_success", False):
                            success_count += 1
                            total_global_problems += result.get("global_problem_count", 0)
                            total_sentence_problems += result.get("sentence_problem_count", 0)
                            semantic_sim = result.get("semantic_similarity", 0)
                            total_semantic_similarity += semantic_sim
                            
                    except Exception as e:
                        logger.error(f"❌ 双重评估批次处理失败: {e}")
                        error_news = self._create_evaluated_news(news_item, {"error": f"批次处理异常: {str(e)}", "evaluation_success": False})
                        batch_results.append(error_news)
                    finally:
                        if pbar:
                            pbar.update(1)
                            
            if pbar:
                pbar.close()
                
            processed_news.extend(batch_results)
            batch_success = sum(1 for news in batch_results if news.get("evaluation_success", False))
            logger.info(f"✅ 双重评估批次完成: 成功 {batch_success}/{len(batch_results)}")
            
        avg_semantic_similarity = total_semantic_similarity / success_count if success_count > 0 else 0
        logger.info(f"🎉 双重批量评估完成! 总计: {len(processed_news)} 篇, 成功: {success_count}")
        logger.info(f"📊 总整体问题: {total_global_problems}, 总句子问题: {total_sentence_problems}")
        logger.info(f"🔍 平均语义相似度: {avg_semantic_similarity:.3f}")
        
        return processed_news

# ==================== 工具函数 ====================

def load_news_for_evaluation(file_path: str) -> List[Dict[str, Any]]:
    """加载新闻数据用于评估"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            news_data = json.load(f)
        if isinstance(news_data, list):
            logger.info(f"✅ 成功加载 {len(news_data)} 篇新闻")
            return news_data
        else:
            logger.error("❌ JSON文件格式错误")
            return []
    except Exception as e:
        logger.error(f"❌ 加载文件失败: {e}")
        return []

def save_evaluated_news(news_list: List[Dict[str, Any]], file_path: str):
    """保存评估结果"""
    try:
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(news_list, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 成功保存 {len(news_list)} 篇评估新闻")
    except Exception as e:
        logger.error(f"❌ 保存文件失败: {e}")

def generate_dual_summary_report(news_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """生成双重评估摘要报告"""
    successful_evaluations = [news for news in news_list if news.get("evaluation_success", False)]
    if not successful_evaluations:
        return {
            "summary": {
                "total_news_evaluated": 0,
                "average_global_problems": 0.0,
                "average_sentence_problems": 0.0,
                "average_total_problems": 0.0,
                "average_semantic_similarity": 0.0
            },
            "global_problems_classification": {"by_type": {}, "total_count": 0},
            "sentence_problems_classification": {"by_type": {}, "total_count": 0},
        }
        
    total_news = len(successful_evaluations)
    global_problems_by_type = defaultdict(int)
    sentence_problems_by_type = defaultdict(int)
    semantic_similarities = []
    
    for news in successful_evaluations:
        evaluation = news.get("dual_evaluation", {})
        problem_classification = evaluation.get("problem_classification", {})
        
        # 统计整体问题
        global_problems = problem_classification.get("global_problems", {})
        for problem_type, count in global_problems.get("by_type", {}).items():
            global_problems_by_type[problem_type] += count
            
        # 统计句子问题
        sentence_problems = problem_classification.get("sentence_problems", {})
        for problem_type, count in sentence_problems.get("by_type", {}).items():
            sentence_problems_by_type[problem_type] += count
            
        semantic_sim = news.get("semantic_similarity", 0)
        semantic_similarities.append(semantic_sim)
        
    avg_global_problems = sum(news.get("global_problem_count", 0) for news in successful_evaluations) / total_news
    avg_sentence_problems = sum(news.get("sentence_problem_count", 0) for news in successful_evaluations) / total_news
    avg_total_problems = sum(news.get("total_problem_count", 0) for news in successful_evaluations) / total_news
    avg_semantic_similarity = sum(semantic_similarities) / len(semantic_similarities) if semantic_similarities else 0
    
    return {
        "summary": {
            "total_news_evaluated": total_news,
            "average_global_problems": round(avg_global_problems, 2),
            "average_sentence_problems": round(avg_sentence_problems, 2),
            "average_total_problems": round(avg_total_problems, 2),
            "average_semantic_similarity": round(avg_semantic_similarity, 3),
        },
        "global_problems_classification": {
            "by_type": dict(global_problems_by_type),
            "total_count": sum(global_problems_by_type.values())
        },
        "sentence_problems_classification": {
            "by_type": dict(sentence_problems_by_type),
            "total_count": sum(sentence_problems_by_type.values())
        },
        "similarity_distribution": {
            "min_semantic_similarity": min(semantic_similarities) if semantic_similarities else 0,
            "max_semantic_similarity": max(semantic_similarities) if semantic_similarities else 0,
            "avg_semantic_similarity": round(avg_semantic_similarity, 3),
        },
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="双重新闻评估工具（整体文本 + 句子级别评估）")
    parser.add_argument("--input", "-i", required=True, help="输入JSON文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出JSON文件路径")
    parser.add_argument("--strict-model", "-sm", default=None, help="严格评估阶段模型名称")
    parser.add_argument("--cot-model", "-cm", default=None, help="CoT验证阶段模型名称")
    parser.add_argument("--similarity-model-path", "-smp", default=None, help="本地相似度模型路径")
    parser.add_argument("--gpu-device", "-gpu", default="cuda:0", help="指定使用的GPU设备，如 cuda:0, cuda:1 等，默认 cuda:0")
    parser.add_argument("--sample", "-s", type=int, default=0, help="处理样本数量")
    args = parser.parse_args()
    # python evaluate_2.py --input fused_with_alignment.json --output evaluation_details/动乱2gpt-4o-mini.json --strict-model deepseek-chat --cot-model deepseek-chat --similarity-model-path /data1/rjj/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --gpu-device cuda:3
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        sys.exit(1)
        
    print("📥 加载新闻数据...")
    news_data = load_news_for_evaluation(args.input)
    if not news_data:
        print("❌ 没有加载到新闻数据")
        sys.exit(1)
        
    if args.sample > 0:
        news_data = news_data[:args.sample]
        print(f"🔬 采样处理前 {args.sample} 篇新闻")
        
    print("🚀 初始化双重评估处理器...")
    print(f"🎯 使用GPU设备: {args.gpu_device}")
    processor = DualEvaluationProcessor(
        strict_model_name=args.strict_model, 
        cot_model_name=args.cot_model, 
        local_similarity_model_path=args.similarity_model_path, 
        gpu_device=args.gpu_device
    )
    
    print("🔄 开始双重评估（整体文本 + 句子级别）...")
    evaluated_news = processor.process_news_batch(news_data)
    
    print("💾 保存结果...")
    save_evaluated_news(evaluated_news, args.output)
    
    summary = generate_dual_summary_report(evaluated_news)
    print("\n📊 双重评估摘要:")
    print(f" 评估新闻总数: {summary['summary']['total_news_evaluated']}")
    print(f" 平均整体问题数: {summary['summary']['average_global_problems']}")
    print(f" 平均句子问题数: {summary['summary']['average_sentence_problems']}")
    print(f" 平均总问题数: {summary['summary']['average_total_problems']}")
    print(f" 平均语义相似度: {summary['summary']['average_semantic_similarity']:.3f}")
    
    if summary['global_problems_classification']['total_count'] > 0:
        print(f" 整体问题分类: {summary['global_problems_classification']['by_type']}")
        
    if summary['sentence_problems_classification']['total_count'] > 0:
        print(f" 句子问题分类: {summary['sentence_problems_classification']['by_type']}")
        
    similarity_dist = summary['similarity_distribution']
    print(f" 语义相似度分布: 最低{similarity_dist['min_semantic_similarity']:.3f}, 最高{similarity_dist['max_semantic_similarity']:.3f}, 平均{similarity_dist['avg_semantic_similarity']:.3f}")

if __name__ == "__main__":
    main()