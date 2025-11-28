import json
import math
import re
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import jieba
import jieba.analyse
import logging
from sentence_transformers import SentenceTransformer 
import os
import random
import copy
import hashlib

# 设置日志（与原实现一致）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --------------------------------------------------
# EmbeddingProvider: 负责模型加载、encoding、基本文本工具
# --------------------------------------------------
class EmbeddingProvider:
    def __init__(self, model_path: str = "/data1/rjj/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device_id: int = 1, batch_size: int = 32, max_sequence_length: int = 256):
        # 保留原来对模型路径的严格校验与错误提示
        if not os.path.exists(model_path):
            logger.error(f"模型路径不存在: {model_path}")
            raise FileNotFoundError(f"模型路径不存在: {model_path}")

        # 设备选择：与原实现一致（优先 GPU 指定 id）
        if torch.cuda.is_available():
            gpu_index = device_id
            device_str = f"cuda:{gpu_index}"
            self.device = torch.device(device_str)
        else:
            self.device = torch.device("cpu")
        logger.info(f"使用设备: {self.device}")
        logger.info(f"加载本地 SentenceTransformer 模型: {model_path}")

        # 加载 sentence-transformers（用于快速编码）
        try:
            self.sr_model = SentenceTransformer(model_path, device=self.device)
            logger.info("SentenceTransformer 加载成功")
        except Exception as e:
            logger.error(f"SentenceTransformer 加载失败: {e}")
            raise

        # 尝试同时加载 tokenizer & AutoModel（可选，用于自定义embedding策略）
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.sentence_model = AutoModel.from_pretrained(
                model_path, local_files_only=True, torch_dtype=torch.float16
            ).to(self.device)
            self.sentence_model.eval()
            logger.info("AutoModel/Tokenizer 加载成功")
        except Exception as e:
            logger.warning(f"AutoModel/Tokenizer 加载失败或不可用（继续使用 sr_model）：{e}")
            self.tokenizer = None
            self.sentence_model = None

        # 初始化 jieba（保留原行为）
        try:
            jieba.initialize()
        except Exception:
            pass

        # 保持原始默认参数（未更改）
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length

    # 文本工具（与原实现一致）
    def contains_chinese(self, text: str) -> bool:
        return bool(re.search(r'[\u4e00-\u9fff]', text))

    def tokenize_for_count(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        if self.contains_chinese(text):
            try:
                return [t for t in jieba.cut_for_search(text) if t.strip()]
            except Exception:
                return [c for c in text if not c.isspace()]
        else:
            return re.findall(r"\w+", text.lower())

    # Embedding 编码接口（直接把原来的实现搬过来）
    def encode_text_batch(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.sr_model.get_sentence_embedding_dimension()), dtype=np.float32)
        try:
            embeddings = self.sr_model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"encode_text_batch failed: {e}")
            raise

    def encode_text(self, text: str) -> np.ndarray:
        emb = self.encode_text_batch([text])
        return emb[0] if emb.shape[0] > 0 else np.zeros((self.sr_model.get_sentence_embedding_dimension(),), dtype=np.float32)

    def cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        try:
            embeddings = self.encode_text_batch([text1, text2])
            if embeddings.shape[0] < 2:
                return 0.0
            return float(self.cosine_sim(embeddings[0], embeddings[1]))
        except Exception as e:
            logger.warning(f"语义相似度计算失败: {e}")
            return 0.0

# --------------------------------------------------
# ParagraphFilter: 提取关键句、主题、过滤白/黑名单与相关性判定
# --------------------------------------------------
class ParagraphFilter:
    def __init__(self, embedder: EmbeddingProvider, similarity_threshold: float = 0.2):
        self.embedder = embedder
        # 保留原始默认阈值
        self.similarity_threshold = similarity_threshold

    # extract_key_sentences 完整保留原实现（未改变逻辑）
    def extract_key_sentences(self, text: str,
                              top_k: int = 3,
                              keywords: Optional[List[str]] = None,
                              weights: Tuple[float, float, float] = (0.3, 0.3, 0.4),
                              use_original_order: bool = False) -> List[str]:
        if not text or not text.strip():
            return []

        sentence_delims = r'[。！？!?；;]|[\n\r]+|(?<=[\.\?\!])\s+'
        raw_sentences = [s.strip() for s in re.split(sentence_delims, text) if s and s.strip()]
        if not raw_sentences:
            return []

        n = len(raw_sentences)
        if keywords is None:
            try:
                keywords = jieba.analyse.extract_tags(text, topK=20)
            except Exception:
                keywords = []
        keyword_tokens = set(kw.lower() for kw in keywords)

        length_weight, position_weight, keyword_weight = weights

        def normalize_list(vals):
            if not vals:
                return [0.0] * len(vals)
            mn, mx = min(vals), max(vals)
            if math.isclose(mx, mn):
                return [1.0] * len(vals)
            return [(v - mn) / (mx - mn) for v in vals]

        sentence_lengths = []
        sentence_positions = []
        sentence_keyword_density = []

        for idx, sent in enumerate(raw_sentences):
            sent = sent.strip()
            sent_len_chars = len(sent)
            sentence_lengths.append(sent_len_chars)
            sentence_positions.append((n - idx) / n)
            if self.embedder.contains_chinese(sent):
                tokens = list(jieba.cut_for_search(sent))
                token_count = len([t for t in tokens if t.strip()])
                if token_count == 0:
                    kd = 0.0
                else:
                    match_count = sum(1 for t in tokens if t in keyword_tokens)
                    kd = match_count / token_count
            else:
                tokens = re.findall(r"\w+", sent.lower())
                token_count = len(tokens)
                if token_count == 0:
                    kd = 0.0
                else:
                    match_count = sum(1 for t in tokens if t in keyword_tokens)
                    kd = match_count / token_count
            sentence_keyword_density.append(kd)

        norm_lengths = normalize_list(sentence_lengths)
        norm_positions = normalize_list(sentence_positions)
        norm_kd = normalize_list(sentence_keyword_density)

        scores = []
        for i in range(n):
            score = (length_weight * norm_lengths[i] +
                     position_weight * norm_positions[i] +
                     keyword_weight * norm_kd[i])
            scores.append((i, raw_sentences[i], score))

        scores_sorted = sorted(scores, key=lambda x: x[2], reverse=True)
        top_k = max(1, min(top_k, len(scores_sorted)))
        selected = scores_sorted[:top_k]

        if use_original_order:
            selected_sorted_by_index = sorted(selected, key=lambda x: x[0])
            return [s[1] for s in selected_sorted_by_index]
        else:
            return [s[1] for s in selected]

    def generate_theme_from_content(self, content: str, title: str = "") -> str:
        key_sentences = self.extract_key_sentences(content, 2)
        if not key_sentences:
            return content[:50] + "..." if len(content) > 50 else content
        if title:
            best_sentence = key_sentences[0]
            best_similarity = self.embedder.calculate_semantic_similarity(best_sentence, title)
            for sentence in key_sentences[1:]:
                similarity = self.embedder.calculate_semantic_similarity(sentence, title)
                if similarity > best_similarity:
                    best_sentence = sentence
                    best_similarity = similarity
            theme = best_sentence
        else:
            theme = key_sentences[0]
        theme = re.sub(r'\s+', ' ', theme.strip())
        if len(theme) > 60:
            theme = theme[:57] + "..."
        return theme
    def is_irrelevant_to_main_topic(self, paragraph: str, title: str, main_topic: str) -> bool:
        if len(paragraph) < 25:
            return True
        if self._is_whitelisted_content(paragraph, title, main_topic):
            return False
        if self._is_blacklisted_content(paragraph):
            logger.info(f"过滤黑名单内容: {paragraph[:50]}...")
            return True
        title_similarity = self.embedder.calculate_semantic_similarity(paragraph, title) if title else 0.0
        topic_similarity = self.embedder.calculate_semantic_similarity(paragraph, main_topic) if main_topic else 0.0
        max_similarity = max(title_similarity, topic_similarity)
        keyword_score = self._calculate_keyword_relevance(paragraph, title, main_topic)
        context_score = self._calculate_context_relevance(paragraph, title, main_topic)
        quality_score = self._calculate_content_quality(paragraph)
        confidence_score = self._calculate_confidence_score(max_similarity, keyword_score, context_score, quality_score)
        dynamic_threshold = self._get_dynamic_threshold(paragraph, title, main_topic)
        is_irrelevant = confidence_score < dynamic_threshold
        if is_irrelevant:
            logger.info(f"过滤与主题无关内容: {paragraph[:50]}... (综合评分: {confidence_score:.3f}, 阈值: {dynamic_threshold:.3f})")
        return is_irrelevant

    def _is_whitelisted_content(self, paragraph: str, title: str = "", main_topic: str = "") -> bool:
        if title or main_topic:
            title_keywords = jieba.analyse.extract_tags(title, topK=10) if title else []
            topic_keywords = jieba.analyse.extract_tags(main_topic, topK=10) if main_topic else []
            all_keywords = title_keywords + topic_keywords
            for keyword in all_keywords:
                if len(keyword) > 1 and keyword in paragraph:
                    return True
        whitelist_keywords = [
            '地震', '台风', '洪水', '火灾', '灾害', '灾难', '灾情', '受灾',
            '防灾', '减灾', '应急', '救援', '疏散', '避难', '安全', '防护',
            '震级', '震中', '震源', '余震', '震感', '震波', '震度',
            '建筑', '设施', '结构', '房屋', '学校', '医院', '公园', '避难所',
            '损坏', '倒塌', '裂缝', '倾斜', '加固', '整修', '维护', '修复',
            '历史', '文化', '保护', '遗产', '古迹', '文物', '修复', '保存',
            '政府', '部门', '机构', '组织', '管理', '规划', '政策', '措施',
            '数据', '统计', '报告', '分析', '评估', '监测', '预警', '预测',
            '地区', '城市', '省份', '县市', '乡镇', '街道', '社区',
            '居民', '民众', '市民', '村民', '住户', '家庭'
        ]
        for keyword in whitelist_keywords:
            if keyword in paragraph:
                return True
        if re.search(r'\d{4}年|\d+月|\d+日|\d+人|\d+元|\d+%|\d+级|\d+度|\d+次', paragraph):
            return True
        location_patterns = [
            r'\w+市', r'\w+县', r'\w+区', r'\w+镇', r'\w+村',
            r'\w+街道', r'\w+社区', r'\w+学校', r'\w+医院', r'\w+公园',
            r'\w+省', r'\w+自治区', r'\w+直辖市'
        ]
        for pattern in location_patterns:
            if re.search(pattern, paragraph):
                return True
        quote_indicators = ['表示', '说', '称', '透露', '指出', '强调', '认为', '介绍', '说明', '报道']
        if any(indicator in paragraph for indicator in quote_indicators):
            return True
        return False

    def _is_blacklisted_content(self, paragraph: str) -> bool:
        clean_paragraph = paragraph.strip()
        blacklist_patterns = [
            r'订阅.*电子报', r'请输入.*格式', r'感谢.*订阅',
            r'每天.*分钟.*掌握.*件', r'早安世界.*电子报', r'分钟掌握.*件天下事',
            r'点击.*链接', r'关注.*公众号', r'下载.*APP', r'注册.*账号',
            r'版权.*所有', r'免责.*声明', r'隐私.*政策', r'使用.*条款',
        ]
        for pattern in blacklist_patterns:
            if re.search(pattern, clean_paragraph):
                return True
        format_patterns = [r'^请输入.*$', r'^点击.*$', r'^订阅.*$', r'^感谢.*$']
        for pattern in format_patterns:
            if re.search(pattern, clean_paragraph):
                return True
        critical_keywords = ['订阅', '电子报', '请输入', '电子信箱', '感谢', '早安世界', '分钟掌握', '天下事']
        keyword_count = sum(1 for keyword in critical_keywords if keyword in clean_paragraph)
        if keyword_count >= 3:
            return True
        if len(clean_paragraph) < 50:
            format_words = ['订阅', '点击', '输入', '感谢', '电子报', '请输入']
            if any(word in clean_paragraph for word in format_words):
                return True
        return False

    def _calculate_keyword_relevance(self, paragraph: str, title: str, main_topic: str) -> float:
        para_keywords = set(jieba.analyse.extract_tags(paragraph, topK=10))
        title_keywords = set(jieba.analyse.extract_tags(title, topK=10)) if title else set()
        topic_keywords = set(jieba.analyse.extract_tags(main_topic, topK=10)) if main_topic else set()
        score = 0.0
        if title_keywords:
            title_overlap = len(para_keywords & title_keywords) / max(len(para_keywords), 1)
            score += title_overlap * 0.4
        if topic_keywords:
            topic_overlap = len(para_keywords & topic_keywords) / max(len(para_keywords), 1)
            score += topic_overlap * 0.3
        important_keywords = [
            '地震', '台风', '洪水', '火灾', '灾害', '救援', '疏散', '避难', '伤亡', '损失',
            '防灾', '减灾', '应急', '安全', '防护', '预警', '监测', '评估', '重建', '修复',
            '建筑', '设施', '结构', '损坏', '倒塌', '裂缝', '倾斜', '加固', '整修', '维护',
            '历史', '文化', '保护', '遗产', '古迹', '文物'
        ]
        important_matches = sum(1 for kw in important_keywords if kw in paragraph)
        score += min(important_matches * 0.08, 0.4)
        return min(score, 1.0)

    def _calculate_context_relevance(self, paragraph: str, title: str, main_topic: str) -> float:
        score = 0.0
        time_keywords = ['今天', '昨日', '昨天', '近日', '最近', '目前', '现在', '当时', '事发']
        time_matches = sum(1 for kw in time_keywords if kw in paragraph)
        score += min(time_matches * 0.1, 0.2)
        location_keywords = ['地区', '市', '县', '镇', '村', '街道', '社区', '学校', '医院', '商场']
        location_matches = sum(1 for kw in location_keywords if kw in paragraph)
        score += min(location_matches * 0.1, 0.2)
        numbers = len(re.findall(r'\d+', paragraph))
        score += min(numbers * 0.05, 0.2)
        quote_indicators = ['表示', '说', '称', '透露', '指出', '强调', '认为', '介绍']
        quote_matches = sum(1 for kw in quote_indicators if kw in paragraph)
        score += min(quote_matches * 0.1, 0.2)
        return min(score, 1.0)

    def _calculate_content_quality(self, paragraph: str) -> float:
        score = 0.0
        length = len(paragraph)
        if 50 <= length <= 500:
            score += 0.3
        elif 25 <= length < 50 or 500 < length <= 800:
            score += 0.2
        else:
            score += 0.1
        sentences = re.split(r'[。！？!?]', paragraph)
        complete_sentences = len([s for s in sentences if len(s.strip()) > 10])
        if complete_sentences > 0:
            score += min(complete_sentences * 0.1, 0.3)
        words = len(paragraph.split())
        if words > 0:
            info_density = len(re.findall(r'[\u4e00-\u9fff]', paragraph)) / words
            score += min(info_density * 0.4, 0.4)
        return min(score, 1.0)

    def _calculate_confidence_score(self, similarity: float, keyword_score: float,
                                  context_score: float, quality_score: float) -> float:
        weights = {'similarity': 0.25, 'keyword': 0.35, 'context': 0.25, 'quality': 0.15}
        confidence = (similarity * weights['similarity'] +
                      keyword_score * weights['keyword'] +
                      context_score * weights['context'] +
                      quality_score * weights['quality'])
        return confidence

    def _get_dynamic_threshold(self, paragraph: str, title: str, main_topic: str) -> float:
        base_threshold = self.similarity_threshold
        if len(paragraph) < 50:
            base_threshold += 0.05
        important_keywords = [
            '地震', '台风', '洪水', '火灾', '灾害', '救援', '疏散', '避难',
            '防灾', '减灾', '应急', '安全', '防护', '预警'
        ]
        if any(kw in paragraph for kw in important_keywords):
            base_threshold -= 0.08
        if re.search(r'\d+', paragraph):
            base_threshold -= 0.05
        quote_indicators = ['表示', '说', '称', '透露', '指出', '强调', '认为', '介绍', '说明']
        if any(kw in paragraph for kw in quote_indicators):
            base_threshold -= 0.03
        location_indicators = ['市', '县', '区', '镇', '村', '街道', '社区', '学校', '医院', '公园']
        if any(kw in paragraph for kw in location_indicators):
            base_threshold -= 0.02
        return max(0.05, min(0.5, base_threshold))

# --------------------------------------------------
# ParagraphMerger: 提取原始段落并执行智能合并
# --------------------------------------------------
class ParagraphMerger:
    def __init__(self, embedder: EmbeddingProvider,
                 merge_similarity_threshold: float = 0.5,
                 max_segment_length: int = 800,
                 min_segment_length: int = 100,
                 ideal_segment_length: int = 400):
        self.embedder = embedder
        self.merge_similarity_threshold = merge_similarity_threshold
        self.max_segment_length = max_segment_length
        self.min_segment_length = min_segment_length
        self.ideal_segment_length = ideal_segment_length

    def _extract_original_paragraphs(self, content: str) -> List[str]:
        content = re.sub(r'[ \t]+', ' ', content.strip())
        paragraphs = []
        if '\n\n' in content:
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            logger.info("使用双换行符分段")
        if len(paragraphs) <= 1 and '\n' in content:
            paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
            logger.info("使用单换行符分段")
        if len(paragraphs) <= 1:
            paragraphs = [content]
            logger.info("保持原样不分段")
        return paragraphs

    def _should_merge_paragraphs(self, current_segment: Dict[str, Any],
                                 next_para: Dict[str, Any],
                                 emb_current: np.ndarray,
                                 emb_next: np.ndarray,
                                 params: Dict[str, Any]) -> bool:
        use_char = params.get("use_char_count", True)
        ideal_length = params.get("ideal_length", 500)
        sim_threshold = params.get("sim_threshold", 0.60)
        lower_sim_threshold = params.get("lower_sim_threshold", 0.45)
        max_merge_length = params.get("max_merge_length", 1500)

        sim = self.embedder.cosine_sim(emb_current, emb_next)
        curr_len = current_segment["char_count"] if use_char else current_segment["token_count"]
        next_len = next_para["char_count"] if use_char else next_para["token_count"]
        merged_len = curr_len + next_len

        if sim >= sim_threshold and merged_len <= max_merge_length:
            return True
        if curr_len < ideal_length and next_len < ideal_length and sim >= lower_sim_threshold:
            return True
        if curr_len < 0.5 * ideal_length and sim >= lower_sim_threshold and merged_len <= max_merge_length:
            return True
        return False

    def _smart_paragraph_merging(self,
                                 para_meta: List[Dict[str, Any]],
                                 ideal_length: int = 500,
                                 use_char_count: bool = True,
                                 sim_threshold: float = 0.60,
                                 lower_sim_threshold: float = 0.45,
                                 max_merge_length: int = 1500,
                                 join_sep: str = " ") -> List[Dict[str, Any]]:
        """
        注意：将默认 join_sep 改为单个空格 " "，
        这样合并后的每个新段落内部不会保留原始的双换行分段符号。
        新段落之间仍然可以在外部被 join 为 '\n\n' 来表示新的分段边界。
        """
        if not para_meta:
            return []

        texts = [p["content"] for p in para_meta]
        try:
            embeddings = self.embedder.encode_text_batch(texts)
        except Exception as e:
            logger.warning(f"Embedding 计算失败，退回到基于长度的保守合并: {e}")
            merged = []
            cur = {
                "content": para_meta[0]["content"],
                "char_count": para_meta[0]["char_count"],
                "token_count": para_meta[0]["token_count"],
                "original_indices": [para_meta[0].get("original_index", 0)]
            }
            for meta in para_meta[1:]:
                cur_len = cur["char_count"] if use_char_count else cur["token_count"]
                next_len = meta["char_count"] if use_char_count else meta["token_count"]
                if cur_len < ideal_length and (cur_len + next_len) <= max_merge_length:
                    cur["content"] = cur["content"] + join_sep + meta["content"]
                    cur["char_count"] += meta["char_count"]
                    cur["token_count"] += meta["token_count"]
                    cur["original_indices"].append(meta.get("original_index", 0))
                else:
                    cur["original_paragraphs_count"] = len(cur["original_indices"])
                    merged.append(cur)
                    cur = {
                        "content": meta["content"],
                        "char_count": meta["char_count"],
                        "token_count": meta["token_count"],
                        "original_indices": [meta.get("original_index", 0)]
                    }
            cur["original_paragraphs_count"] = len(cur["original_indices"])
            merged.append(cur)
            return merged

        merged_segments = []
        cur_segment = {
            "content": para_meta[0]["content"],
            "char_count": para_meta[0]["char_count"],
            "token_count": para_meta[0]["token_count"],
            "original_indices": [para_meta[0].get("original_index", 0)]
        }
        emb_cur = embeddings[0].copy()

        params = {
            "use_char_count": use_char_count,
            "ideal_length": ideal_length,
            "sim_threshold": sim_threshold,
            "lower_sim_threshold": lower_sim_threshold,
            "max_merge_length": max_merge_length
        }

        for next_idx in range(1, len(para_meta)):
            next_meta = para_meta[next_idx]
            emb_next = embeddings[next_idx]
            if self._should_merge_paragraphs(cur_segment, next_meta, emb_cur, emb_next, params):
                # 使用 join_sep 连接，而不是始终使用 "\n\n"
                cur_segment["content"] = cur_segment["content"] + join_sep + next_meta["content"]
                cur_segment["char_count"] += next_meta["char_count"]
                cur_segment["token_count"] += next_meta["token_count"]
                cur_segment["original_indices"].append(next_meta.get("original_index", next_idx))
                old_len = cur_segment["char_count"] - next_meta["char_count"]
                total_len = cur_segment["char_count"]
                if old_len > 0:
                    emb_cur = (emb_cur * old_len + emb_next * next_meta["char_count"]) / max(1, total_len)
                else:
                    emb_cur = emb_next
            else:
                cur_segment["original_paragraphs_count"] = len(cur_segment["original_indices"])
                merged_segments.append(cur_segment)
                cur_segment = {
                    "content": next_meta["content"],
                    "char_count": next_meta["char_count"],
                    "token_count": next_meta["token_count"],
                    "original_indices": [next_meta.get("original_index", next_idx)]
                }
                emb_cur = embeddings[next_idx].copy()

        cur_segment["original_paragraphs_count"] = len(cur_segment["original_indices"])
        merged_segments.append(cur_segment)
        return merged_segments

# --------------------------------------------------
# IOHandler: 负责文件读写（load/save）
# --------------------------------------------------
class IOHandler:
    def load_news_data(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                news_data = json.load(f)
            logger.info(f"成功加载 {len(news_data)} 条新闻数据")
            return news_data
        except Exception as e:
            logger.error(f"加载新闻数据失败: {e}")
            return []

    def save_processed_news(self, news_data: List[Dict[str, Any]], output_path: str):
        try:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(news_data, f, ensure_ascii=False, indent=2)
            logger.info(f"成功保存处理后的数据到 {output_path}")
        except Exception as e:
            logger.error(f"保存数据失败: {e}")

    def save_cleaned_news(self, news_data: List[Dict[str, Any]], output_path: str):
        try:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            cleaned_news_data = []
            for news_item in news_data:
                cleaned_item = {
                    "id": news_item.get("id", ""),
                    "title": news_item.get("title", ""),
                    "content": news_item.get("cleaned_content", news_item.get("content", "")),
                    "url": news_item.get("url", ""),
                    "publish_date": news_item.get("publish_date", ""),
                    "seendate": news_item.get("seendate", ""),
                    "source": news_item.get("source", ""),
                    "year": news_item.get("year", ""),
                    "is_disaster_related": news_item.get("is_disaster_related", ""),
                    "main_topic": news_item.get("main_topic", ""),
                    "keywords_found": news_item.get("keywords_found", [])
                }
                cleaned_news_data.append(cleaned_item)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_news_data, f, ensure_ascii=False, indent=2)
            logger.info(f"成功保存重新分段后的数据到 {output_path}")
        except Exception as e:
            logger.error(f"保存重新分段数据失败: {e}")

# --------------------------------------------------
# SegmenterController: 组合上述组件，保持原有流程逻辑
# --------------------------------------------------
class SegmenterController:
    def __init__(self,
                 model_path: str = "/data1/rjj/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 device_id: int = 1):
        # 保持原默认参数与行为（将这些参数名与原类保持一致）
        self.embedder = EmbeddingProvider(model_path=model_path, device_id=device_id)
        # 阈值参数与原来类中默认值一致
        self.similarity_threshold = 0.2
        self.merge_similarity_threshold = 0.5
        self.max_segment_length = 800
        self.min_segment_length = 100
        self.ideal_segment_length = 400
        self.max_sequence_length = 256
        self.batch_size = 32

        self.filter = ParagraphFilter(self.embedder, similarity_threshold=self.similarity_threshold)
        self.merger = ParagraphMerger(self.embedder,
                                      merge_similarity_threshold=self.merge_similarity_threshold,
                                      max_segment_length=self.max_segment_length,
                                      min_segment_length=self.min_segment_length,
                                      ideal_segment_length=self.ideal_segment_length)
        self.io = IOHandler()
        logger.info("优化段落合并的分段系统初始化完成（已拆分为组件）")

    # 把原来类中的方法迁移到这里并调用对应组件

    def _calculate_paragraph_importance(self, paragraph: str, title: str, main_topic: str) -> float:
        importance = 0.0
        if title:
            title_similarity = self.embedder.calculate_semantic_similarity(paragraph, title)
            importance += title_similarity * 0.4
        if main_topic:
            topic_similarity = self.embedder.calculate_semantic_similarity(paragraph, main_topic)
            importance += topic_similarity * 0.3
        length_factor = min(1.0, len(paragraph) / 200)
        importance += length_factor * 0.2
        keywords = jieba.analyse.extract_tags(paragraph, topK=5)
        keyword_density = len(keywords) / max(1, len(paragraph.split()))
        importance += min(keyword_density * 2, 0.1)
        return importance

    def preserve_original_structure_segmentation(self, content: str, title: str = "", main_topic: str = "") -> List[Dict[str, Any]]:
        original_paragraphs = self.merger._extract_original_paragraphs(content)
        logger.info(f"原始分段数: {len(original_paragraphs)}")
        if not original_paragraphs:
            return []

        para_meta = []
        for i, paragraph in enumerate(original_paragraphs):
            if not self.filter.is_irrelevant_to_main_topic(paragraph, title, main_topic):
                para_meta.append({
                    "original_index": i,
                    "content": paragraph,
                    "char_count": len(paragraph),
                    "token_count": len(self.embedder.tokenize_for_count(paragraph))
                })
        logger.info(f"过滤后保留 {len(para_meta)}/{len(original_paragraphs)} 个段落")
        if not para_meta:
            return []

        # 这里传入 join_sep=" "，保证合并后的段落内部不保留原始双换行
        merged_segments = self.merger._smart_paragraph_merging(
            para_meta,
            ideal_length=self.ideal_segment_length,
            use_char_count=True,
            sim_threshold=self.merge_similarity_threshold,
            lower_sim_threshold=self.merge_similarity_threshold - 0.15,
            max_merge_length=self.max_segment_length,
            join_sep=" "
        )

        segments = []
        for i, seg in enumerate(merged_segments):
            segment_content = seg["content"]
            theme = self.filter.generate_theme_from_content(segment_content, title)
            importance = self._calculate_paragraph_importance(segment_content, title, main_topic)
            segments.append({
                "segment_id": i + 1,
                "content": segment_content,
                "theme": theme,
                "char_count": seg.get("char_count", len(segment_content)),
                "token_count": seg.get("token_count", len(self.embedder.tokenize_for_count(segment_content))),
                "importance_score": round(importance, 3),
                "original_paragraphs_count": seg.get("original_paragraphs_count", 1),
                "original_indices": seg.get("original_indices", [])
            })
        return segments

    def process_news_dataset(self, input_file: str, output_file: str, cleaned_output_file: str, sample_size: int = None):
        news_data = self.io.load_news_data(input_file)
        if not news_data:
            logger.error("没有加载到数据，处理终止")
            return
        if sample_size and sample_size < len(news_data):
            news_data = news_data[:sample_size]
            logger.info(f"采样处理前 {sample_size} 条新闻")

        processed_count = 0
        error_count = 0

        for i, news_item in enumerate(news_data):
            try:
                logger.info(f"处理新闻 {i+1}/{len(news_data)}: {news_item.get('title', '无标题')[:30]}...")
                content = news_item.get("content", "")
                title = news_item.get("title", "")
                main_topic = news_item.get("main_topic", "")
                if not content.strip():
                    logger.warning(f"新闻 {news_item.get('id', '未知')} 内容为空")
                    continue

                segments = self.preserve_original_structure_segmentation(content, title, main_topic)
                news_item["segmented_content"] = segments
                news_item["segment_count"] = len(segments)
                news_item["processing_success"] = True
                if segments:
                    # 新段落之间仍用双换行分隔，但每个段落内部已经用空格连接，避免保留原始分段标志
                    cleaned_content = "\n\n".join([segment["content"] for segment in segments])
                    news_item["cleaned_content"] = cleaned_content
                else:
                    news_item["cleaned_content"] = content

                if segments:
                    total_chars = sum(segment["char_count"] for segment in segments)
                    avg_importance = float(np.mean([segment["importance_score"] for segment in segments])) if segments else 0.0
                    original_char_count = len(content)
                    segment_lengths = [segment["char_count"] for segment in segments]
                    avg_segment_length = float(np.mean(segment_lengths)) if segment_lengths else 0.0
                    min_segment_length = min(segment_lengths) if segment_lengths else 0
                    max_segment_length = max(segment_lengths) if segment_lengths else 0

                    news_item["segmentation_stats"] = {
                        "total_chars": total_chars,
                        "average_importance": round(avg_importance, 3),
                        "original_char_count": original_char_count,
                        "filtered_ratio": round(1 - total_chars / original_char_count, 3) if original_char_count else 0,
                        "average_paragraphs_per_segment": float(np.mean([segment.get("original_paragraphs_count", 1) for segment in segments])),
                        "segment_length_stats": {
                            "average": round(avg_segment_length, 1),
                            "min": min_segment_length,
                            "max": max_segment_length
                        }
                    }

                processed_count += 1

                if i == 0 and segments:
                    logger.info("首个新闻处理结果示例:")
                    for segment in segments[:3]:
                        logger.info(f"  段落 {segment['segment_id']}:")
                        logger.info(f"    主题: {segment['theme']}")
                        logger.info(f"    重要性: {segment['importance_score']}, 字符数: {segment['char_count']}")
                        logger.info(f"    合并自原始段落数: {segment['original_paragraphs_count']}")
                        logger.info(f"    原始索引: {segment['original_indices']}")

                if i % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"处理新闻 {news_item.get('id', '未知')} 时出错: {e}")
                news_item["processing_success"] = False
                news_item["error"] = str(e)
                error_count += 1

        self.io.save_processed_news(news_data, output_file)
        self.io.save_cleaned_news(news_data, cleaned_output_file)
        logger.info(f"处理完成! 成功: {processed_count}, 失败: {error_count}")

        successful_news = [news for news in news_data if news.get("processing_success", False)]
        if successful_news:
            avg_segments = float(np.mean([news.get("segment_count", 0) for news in successful_news]))
            avg_paragraphs_per_segment = float(np.mean([
                news.get("segmentation_stats", {}).get("average_paragraphs_per_segment", 1) 
                for news in successful_news 
                if news.get("segmentation_stats")
            ]))
            avg_segment_lengths = [
                news.get("segmentation_stats", {}).get("segment_length_stats", {}).get("average", 0)
                for news in successful_news 
                if news.get("segmentation_stats") and news.get("segmentation_stats", {}).get("segment_length_stats")
            ]
            avg_segment_length = float(np.mean(avg_segment_lengths)) if avg_segment_lengths else 0
            logger.info(f"平均分段数: {avg_segments:.1f}")
            logger.info(f"平均每个分段合并的原始段落数: {avg_paragraphs_per_segment:.1f}")
            logger.info(f"平均段落长度(字符): {avg_segment_length:.1f}")


# --------------------------------------------------
# main: 保留原入口不变（仅替换类名）
# --------------------------------------------------
def main():
    model_path = "/data1/rjj/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    segmenter = SegmenterController(model_path=model_path)
    input_file = "/data1/rjj/a.test/classified_disaster_news/en.json"
    output_file = "/data1/rjj/a.test/processed_news_detail/en.json"
    cleaned_output_file = "/data1/rjj/a.test/processed_news/en.json"
    segmenter.process_news_dataset(input_file, output_file, cleaned_output_file)

if __name__ == "__main__":
    main()