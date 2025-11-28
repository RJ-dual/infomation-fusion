

#!/usr/bin/env python3
# coding: utf-8
"""
inject_conflicts_langchain.py

用途：
- 使用spaCy提取实体，然后让LLM基于实体列表进行智能冲突注入
- 输出注入冲突后的版本和冲突详情
"""

import os
import re
import json
import ast
import argparse
import random
import logging
import time
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# 首先设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LangChain / OpenAI wrapper
try:
    from langchain_openai import ChatOpenAI
except Exception:
    try:
        from langchain_community.chat_models import ChatOpenAI
    except Exception:
        ChatOpenAI = None

from langchain.schema import SystemMessage, HumanMessage

# 导入配置
from config import get_model_deployment_config, DEFAULT_STUDENT_MODEL, REQUEST_TIMEOUT

# spaCy 实体抽取
try:
    import spacy
    nlp = None
    try:
        nlp = spacy.load("zh_core_web_sm")
        logger.info("成功加载 spaCy 中文模型")
    except Exception as e:
        logger.warning(f"无法加载 spaCy 中文模型: {e}")
        nlp = None
except Exception:
    nlp = None
    logger.warning("spaCy 不可用，将无法提取实体")

# ---------- 配置与默认值 ----------
DEFAULT_MODEL_NAME = DEFAULT_STUDENT_MODEL or os.environ.get("MODEL_NAME", "gpt-4o-mini")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_BASE_URL = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
DEFAULT_TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))
DEFAULT_MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "800"))

# ---------- 工具函数 ----------
def extract_entities(text: str) -> List[Dict[str, Any]]:
    """
    使用 spaCy 提取实体
    """
    ents = []
    if nlp:
        doc = nlp(text)
        for e in doc.ents:
            # 关注主要实体类型
            if e.label_ in ["PERSON", "ORG", "GPE", "DATE", "CARDINAL", "MONEY", "PERCENT"]:
                ents.append({
                    "text": e.text,
                    "label": e.label_,
                    "start": e.start_char,
                    "end": e.end_char
                })
        logger.debug(f"spaCy 提取到 {len(ents)} 个实体: {ents}")
    else:
        logger.warning("spaCy 不可用，无法提取实体")

    # 去重
    seen = set()
    out = []
    for e in ents:
        key = (e["text"], e["start"], e["end"])
        if key not in seen:
            seen.add(key)
            out.append(e)

    return out

def locate_span(doc_text: str, span_text: str) -> Tuple[Any, Any]:
    if not span_text:
        return (None, None)
    idx = doc_text.find(span_text)
    if idx == -1:
        return (None, None)
    return (idx, idx + len(span_text))

def _extract_json_by_brace_matching(text: str) -> str:
    """
    从 text 中找到第一个完整 JSON 对象（从第一个 '{' 开始，匹配大括号对，考虑字符串与转义）。
    返回 JSON 子串或 None。
    """
    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if ch == '"' and not escape:
            in_string = not in_string
        if ch == '\\' and not escape:
            escape = True
            # 继续到下一字符；escape 控制在下一循环中复位
            continue
        else:
            escape = False

        if in_string:
            continue

        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None

def _try_repair_json_text(txt: str) -> str:
    """
    对一些常见的 JSON 格式问题做简单修复：
    - 移除对象或数组结尾处的多余逗号（例如 "...", } 或 ...", ]）
    - 如果是使用单引号的 Python 字面量，尝试通过 ast.literal_eval 转换为标准 JSON
    返回修复后的字符串（可能与输入相同）
    """
    # 1) 移除多余的逗号：", } 或 ", ]
    repaired = re.sub(r',\s*(\}|\])', r'\1', txt)

    # 2) 尝试 json.loads，若失败再尝试 ast.literal_eval（处理单引号等）
    try:
        json.loads(repaired)
        return repaired
    except Exception:
        pass

    # 3) 尝试 ast.literal_eval（将 Python 字面量转换为对象，再转为 JSON）
    try:
        obj = ast.literal_eval(repaired)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass

    return repaired

def extract_json_from_text(text: str) -> Tuple[str, str]:
    """
    尝试从 LLM 返回的 text 中提取 JSON 子串，返回 (json_text, error_reason)
    如果无法提取，返回 (None, reason)
    """
    if not text or not isinstance(text, str):
        return (None, "empty_or_not_str")

    # 首先通过大括号匹配提取最可能的 JSON 块
    json_text = _extract_json_by_brace_matching(text)
    if not json_text:
        return (None, "no_brace_match")

    # 尝试直接解析
    try:
        json.loads(json_text)
        return (json_text, "")
    except json.JSONDecodeError as e:
        logger.debug(f"直接 json.loads 失败: {e}; 尝试修复")
        repaired = _try_repair_json_text(json_text)
        try:
            json.loads(repaired)
            return (repaired, "repaired")
        except Exception as e2:
            logger.debug(f"修复后仍然解析失败: {e2}")
            return (None, f"json_decode_failed: {e2}")

# ---------- LLM 客户端（LangChain ChatOpenAI） ----------
class LLMClient:
    def __init__(self, model_name: str = None, model_config: Dict[str,Any] = None, max_retries: int = 3, backoff_factor: float = 1.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        if model_config is None:
            model_name = model_name or DEFAULT_MODEL_NAME
            try:
                model_config = get_model_deployment_config(model_name)
            except Exception as e:
                logger.warning(f"无法通过 get_model_deployment_config 获取配置: {e}, 使用默认环境变量/参数回退")
                model_config = {
                    "model": model_name,
                    "model_name": model_name,
                    "api_key": os.environ.get("OPENAI_API_KEY", DEFAULT_API_KEY),
                    "base_url": os.environ.get("OPENAI_API_BASE", DEFAULT_BASE_URL),
                    "temperature": DEFAULT_TEMPERATURE,
                    "max_tokens": DEFAULT_MAX_TOKENS
                }

        # 兼容字段名
        api_key = model_config.get("api_key", model_config.get("openai_api_key", model_config.get("openai_api_key", DEFAULT_API_KEY)))
        if api_key == "EMPTY":
            api_key = "vllm"
            logger.info("api_key == 'EMPTY' -> 使用 'vllm' 协议占位符以兼容本地 vLLM")
        base_url = model_config.get("base_url", model_config.get("openai_api_base", DEFAULT_BASE_URL))
        temperature = model_config.get("temperature", DEFAULT_TEMPERATURE)
        max_tokens = model_config.get("max_tokens", DEFAULT_MAX_TOKENS)
        llm_model_name = model_config.get("model_name", model_config.get("model", DEFAULT_MODEL_NAME))

        # 初始化 ChatOpenAI
        if ChatOpenAI is None:
            raise RuntimeError("无法导入 ChatOpenAI，请检查 langchain_openai 或 langchain_community 是否已安装")

        init_kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "openai_api_key": api_key,
            "openai_api_base": base_url,
            "request_timeout": REQUEST_TIMEOUT
        }

        try:
            client = ChatOpenAI(model=llm_model_name, **init_kwargs)
        except TypeError:
            try:
                client = ChatOpenAI(model_name=llm_model_name, **init_kwargs)
            except Exception as e:
                logger.exception(f"初始化 ChatOpenAI 失败: {e}")
                raise
        except Exception as e:
            logger.exception(f"初始化 ChatOpenAI 未处理异常: {e}")
            raise

        self.client = client

    def _call_llm(self, messages: List[Any], timeout: int = None) -> str:
        """
        统一调用 LLM 的方法
        """
        attempt = 0
        last_exc = None
        while attempt < self.max_retries:
            try:
                if hasattr(self.client, "invoke"):
                    resp = self.client.invoke(messages)
                    text = getattr(resp, "content", None) or str(resp)
                    return text

                if callable(self.client):
                    resp = self.client(messages)
                    text = getattr(resp, "content", None) or (resp[0].content if isinstance(resp, list) and resp else None) or str(resp)
                    return text

                if hasattr(self.client, "generate"):
                    resp = self.client.generate([messages])
                    try:
                        gens = getattr(resp, "generations", None)
                        if gens:
                            text = gens[0][0].text if isinstance(gens[0], list) else gens[0][0].text
                        else:
                            text = str(resp)
                        return text
                    except Exception:
                        return str(resp)

                logger.debug("LLM client has no recognized call method; attempting str() on client(messages)")
                resp = self.client(messages)
                text = getattr(resp, "content", None) or str(resp)
                return text

            except Exception as e:
                attempt += 1
                wait = self.backoff_factor * (2 ** (attempt - 1))
                logger.warning(f"LLM 调用异常 (尝试 {attempt}/{self.max_retries}): {e}. 回退等待 {wait:.1f}s 重试...")
                time.sleep(wait)
                last_exc = e

        logger.exception(f"LLM 调用失败，超出最大重试次数: {self.max_retries}")
        raise last_exc

    def inject_conflicts(self, original_text: str, entities: List[Dict[str, Any]], num_conflicts: int = 6) -> Dict[str, Any]:
        """
        基于实体列表让LLM进行智能冲突注入
        """
        # 构建实体信息字符串
        entities_info = []
        for i, entity in enumerate(entities):
            entities_info.append(f"{i+1}. 文本: '{entity['text']}', 类型: {entity['label']}, 位置: [{entity['start']}-{entity['end']}]")

        entities_str = "\n".join(entities_info) if entities_info else "未发现实体"

        system_prompt = (
            "你是一个文本冲突注入专家。你的任务是在保持文本流畅性和合理性的前提下，"
            "在文本中注入事实冲突。\n\n"
            "冲突类型包括：\n"
            "- 数字冲突：修改数字、日期、金额等（年份应保持合理，避免出现小数年份）\n"
            "- 实体冲突：替换人名、组织名、地名等（保持同类替换）\n"
            "- 时间冲突：修改时间顺序或日期\n"
            "- 事实冲突：修改事件细节或因果关系\n\n"
            "输出必须是有效的 JSON 格式，包含注入后的文本和冲突详情。"
        )

        user_prompt = {
            "original_text": original_text,
            "extracted_entities": entities_str,
            "num_conflicts": num_conflicts,
            "requirements": [
                "基于提取的实体列表，选择合适的位置注入冲突",
                "年份修改应保持合理性（如2023年可改为2022年或2024年，但不能改为小数）",
                "实体替换应保持语义合理性（人名换人名，组织换组织）",
                "注入的冲突应该自然，不明显破坏文本的流畅性",
                "总共注入约 {num_conflicts} 个冲突",
                "输出必须为严格的JSON格式，注意逗号分隔和双引号"
            ],
            "output_example": {
                "injected_text": "这是注入冲突后的完整文本。",
                "conflicts": [
                    {
                        "original_span": "原文本1",
                        "injected_span": "修改后文本1",
                        "type": "数字",
                        "note": "修改数字"
                    },
                    {
                        "original_span": "原文本2",
                        "injected_span": "修改后文本2",
                        "type": "实体",
                        "note": "替换实体"
                    }
                ]
            },
            "important": "必须输出纯JSON，不要任何额外文字说明，确保JSON格式正确"
        }

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(user_prompt, ensure_ascii=False, indent=2))
        ]

        try:
            raw_text = self._call_llm(messages)
            if not raw_text or not isinstance(raw_text, str):
                raise ValueError("LLM 未返回文本")

            # 尝试提取 JSON
            json_text, reason = extract_json_from_text(raw_text)
            if not json_text:
                logger.warning(f"LLM 未返回可解析的 JSON 格式结果 (reason={reason})。raw_text 示例（前1000字符）:\n{raw_text[:1000]}")
                return self._create_fallback_response(original_text)

            try:
                parsed = json.loads(json_text)
            except Exception as e:
                logger.exception(f"解析提取到的 JSON 失败: {e}. json_text 示例（前1000字符）:\n{json_text[:1000]}")
                return self._create_fallback_response(original_text)

            # 验证必要字段
            if "injected_text" not in parsed or "conflicts" not in parsed:
                logger.warning("LLM 返回的JSON缺少必要字段")
                return self._create_fallback_response(original_text)

            # 确保冲突列表中的位置信息正确
            injected_text = parsed["injected_text"]
            conflicts = parsed["conflicts"]

            for conflict in conflicts:
                if "injected_span" in conflict:
                    start_pos, end_pos = locate_span(injected_text, conflict["injected_span"])
                    conflict["span_start"] = start_pos
                    conflict["span_end"] = end_pos
                else:
                    conflict["span_start"] = None
                    conflict["span_end"] = None

            return parsed

        except Exception as e:
            logger.exception(f"LLM inject_conflicts error: {e}")
            return self._create_fallback_response(original_text)

    def _create_fallback_response(self, original_text: str) -> Dict[str, Any]:
        """创建回退响应"""
        return {
            "injected_text": original_text,
            "conflicts": [
                {
                    "original_span": "",
                    "injected_span": "",
                    "type": "unknown",
                    "note": "LLM 调用失败",
                    "span_start": None,
                    "span_end": None
                }
            ]
        }

# ---------- 注入流程 ----------
class ConflictInjector:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def inject_into_version(self, original_text: str, num_conflicts: int = 6) -> Dict[str, Any]:
        """
        对单个版本进行冲突注入
        """
        # 提取实体
        entities = extract_entities(original_text)
        logger.info(f"提取到 {len(entities)} 个实体")

        # 让LLM进行冲突注入
        result = self.llm.inject_conflicts(original_text, entities, num_conflicts)

        return {
            "injected_text": result["injected_text"],
            "conflicts": result["conflicts"]
        }

# ---------- I/O 与 主流程 ----------
def load_rewritten_news(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and "top_rewrites" in data:
        item = {
            "rewritten_v1": data["top_rewrites"][0].get("final_rewritten_text", "") if len(data["top_rewrites"])>0 else "",
            "rewritten_v2": data["top_rewrites"][1].get("final_rewritten_text", "") if len(data["top_rewrites"])>1 else "",
            "rewritten_v3": data["top_rewrites"][2].get("final_rewritten_text", "") if len(data["top_rewrites"])>2 else "",
            "input_text": data.get("input_text", "")
        }
        return [item]
    elif isinstance(data, list):
        return data
    else:
        return [data]

def save_output(out: Any, path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="输入 JSON 文件（含 top_rewrites）")
    parser.add_argument("--output", "-o", required=True, help="输出 JSON 文件（仅含注入后的版本与冲突详情）")
    parser.add_argument("--conflicts-per-version", type=int, default=6, help="每个版本注入的冲突数量")
    parser.add_argument("--model", default=None, help="模型部署名；若为空则使用默认模型")
    args = parser.parse_args()
    # python conflict_injection.py --input /data1/rjj/a.test/rewrite_outputs/news_001_summary.json --output news_with_conflicts.json --conflicts-per-version 2 --model gpt-4o-mini

    # 获取模型部署配置
    model_key = args.model or DEFAULT_MODEL_NAME
    try:
        model_config = get_model_deployment_config(model_key)
    except Exception as e:
        logger.warning(f"无法通过 get_model_deployment_config 获取配置: {e}，将使用环境变量或默认参数")
        model_config = {
            "model": model_key,
            "model_name": model_key,
            "api_key": os.environ.get("OPENAI_API_KEY", DEFAULT_API_KEY),
            "base_url": os.environ.get("OPENAI_API_BASE", DEFAULT_BASE_URL),
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS
        }

    api_key = model_config.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.error("需要提供 OPENAI API key（通过 model_config 或 环境变量 OPENAI_API_KEY）")
        return

    data_list = load_rewritten_news(args.input)
    llm_client = LLMClient(model_config=model_config)
    injector = ConflictInjector(llm_client)

    results = []
    for item in tqdm(data_list, desc="处理文档"):
        # 依次对三版进行注入
        out_item = {"input_text": item.get("input_text",""), "versions": []}
        for vid in ("rewritten_v1","rewritten_v2","rewritten_v3"):
            txt = item.get(vid, "")
            if not txt:
                continue

            res = injector.inject_into_version(txt, num_conflicts=args.conflicts_per_version)
            out_item["versions"].append({
                "version_id": vid,
                "injected_text": res["injected_text"],
                "conflicts": res["conflicts"]
            })
        results.append(out_item)

    save_output(results, args.output)
    logger.info(f"完成，输出写入：{args.output}")

if __name__ == "__main__":
    main()

