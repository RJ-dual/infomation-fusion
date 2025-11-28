#!/usr/bin/env python3
# coding: utf-8
"""
inject_conflicts_langchain.py

用途：
- 使用 spaCy 提取 rewritten_v1/v2/v3 的**共有/联合实体**
- 让 LLM 基于**共享实体列表**，对每个版本**分别注入不同但合理的冲突**
- 实现**版本间冲突**（inter-version conflict），用于评估一致性
- 详细追踪每个冲突的位置和版本间关系
"""

import os
import re
import json
import ast
import argparse
import logging
import time
from typing import List, Dict, Any, Tuple, Set
from tqdm import tqdm
from collections import defaultdict

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
try:
    from config import get_model_deployment_config, DEFAULT_STUDENT_MODEL, REQUEST_TIMEOUT
except ImportError:
    # 提供默认配置
    def get_model_deployment_config(model_name):
        return {
            "model": model_name,
            "model_name": model_name,
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
            "base_url": os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
            "temperature": float(os.environ.get("TEMPERATURE", "0.0")),
            "max_tokens": int(os.environ.get("MAX_TOKENS", "800"))
        }
    
    DEFAULT_STUDENT_MODEL = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    REQUEST_TIMEOUT = 120

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
            if e.label_ in ["PERSON", "ORG", "GPE", "DATE", "CARDINAL", "MONEY", "PERCENT"]:
                ents.append({
                    "text": e.text,
                    "label": e.label_,
                    "start_char": e.start_char,
                    "end_char": e.end_char
                })
        logger.debug(f"spaCy 提取到 {len(ents)} 个实体: {ents}")
    else:
        logger.warning("spaCy 不可用，无法提取实体")

    # 去重
    seen = set()
    out = []
    for e in ents:
        key = (e["text"], e["label"])
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out

def find_shared_entities_across_versions(versions: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    找出在所有版本中都出现的实体
    """
    entity_occurrence = defaultdict(set)
    
    # 提取每个版本的实体并记录出现位置
    for version_id, text in versions.items():
        entities = extract_entities(text)
        for entity in entities:
            key = (entity['text'], entity['label'])
            entity_occurrence[key].add(version_id)
    
    # 找出在所有版本中都出现的实体
    shared_entities = []
    for (text, label), version_set in entity_occurrence.items():
        if version_set == set(versions.keys()):
            shared_entities.append({
                "text": text,
                "label": label,
                "versions": list(version_set)
            })
    
    logger.info(f"找到 {len(shared_entities)} 个跨版本共享实体")
    return shared_entities

def locate_span(doc_text: str, span_text: str) -> Tuple[int, int]:
    """定位文本片段在文档中的位置"""
    if not span_text:
        return (None, None)
    idx = doc_text.find(span_text)
    if idx == -1:
        return (None, None)
    return (idx, idx + len(span_text))

def _extract_json_by_brace_matching(text: str) -> str:
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
    repaired = re.sub(r',\s*(\}|\])', r'\1', txt)
    try:
        json.loads(repaired)
        return repaired
    except Exception:
        pass

    try:
        obj = ast.literal_eval(repaired)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass

    return repaired

def extract_json_from_text(text: str) -> Tuple[str, str]:
    if not text or not isinstance(text, str):
        return (None, "empty_or_not_str")

    # 清理可能的Markdown代码块
    cleaned_text = text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]
    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[3:]
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]
    cleaned_text = cleaned_text.strip()

    json_text = _extract_json_by_brace_matching(cleaned_text)
    if not json_text:
        return (None, "no_brace_match")

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


# ---------- LLM 客户端 ----------
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

        api_key = model_config.get("api_key", model_config.get("openai_api_key", DEFAULT_API_KEY))
        if api_key == "EMPTY":
            api_key = "vllm"
            logger.info("api_key == 'EMPTY' -> 使用 'vllm' 协议占位符以兼容本地 vLLM")
        base_url = model_config.get("base_url", model_config.get("openai_api_base", DEFAULT_BASE_URL))
        temperature = model_config.get("temperature", DEFAULT_TEMPERATURE)
        max_tokens = model_config.get("max_tokens", DEFAULT_MAX_TOKENS)
        llm_model_name = model_config.get("model_name", model_config.get("model", DEFAULT_MODEL_NAME))

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
            client = ChatOpenAI(model_name=llm_model_name, **init_kwargs)
        except Exception as e:
            logger.exception(f"初始化 ChatOpenAI 失败: {e}")
            raise

        self.client = client

    def _call_llm(self, messages: List[Any], timeout: int = None) -> str:
        attempt = 0
        last_exc = None
        while attempt < self.max_retries:
            try:
                if hasattr(self.client, "invoke"):
                    resp = self.client.invoke(messages)
                    text = getattr(resp, "content", None) or str(resp)
                    return text

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

    def inject_conflicts_for_version(
        self,
        version_text: str,
        shared_entities: List[Dict[str, Any]],
        version_id: str,
        all_version_ids: List[str],
        num_conflicts: int = 6
    ) -> Dict[str, Any]:
        """
        为单个版本注入冲突，但基于共享实体，并与其他版本**故意制造不一致**
        """
        entities_info = []
        for i, entity in enumerate(shared_entities):
            entities_info.append(f"{i+1}. 文本: '{entity['text']}', 类型: {entity['label']}")

        entities_str = "\n".join(entities_info) if entities_info else "未发现实体"

        system_prompt = (
            "你是一个文本冲突注入专家。你的任务是在保持文本流畅的前提下，"
            "基于**共享实体列表**，为当前版本注入**与其他版本不一致**的事实冲突。\n\n"
            "冲突必须从共享实体列表中选择目标！\n\n"
            "冲突类型包括：\n"
            "- 数字冲突：修改数字、日期、金额等（年份应合理，如2023→2022或2024）\n"
            "- 实体冲突：替换人名、组织名、地名（同类替换，如'北京大学'→'清华大学'）\n"
            "- 时间冲突：调整时间顺序\n"
            "- 事实冲突：修改事件细节\n\n"
            "特别要求：\n"
            f"- 你正在处理版本：{version_id}\n"
            f"- 其他版本包括：{', '.join([v for v in all_version_ids if v != version_id])}\n"
            "- **必须确保你注入的冲突值与其他版本不同**（例如：如果其他版本把年份改为2022，你就改2024）\n"
            "- 冲突应自然，不破坏流畅性\n"
            "- 输出必须是严格 JSON，不要任何额外文字\n"
            "- 在冲突记录中必须包含 'entity_text' 字段，记录被修改的原始实体文本"
        )

        user_prompt = {
            "current_version_id": version_id,
            "version_text": version_text,
            "shared_entities": entities_str,
            "num_conflicts": num_conflicts,
            "requirements": [
                "冲突必须基于 shared_entities 中的实体进行修改",
                "年份修改需合理（整数，±1~3年）",
                "同类实体替换（人名换人名，机构换机构）",
                "确保与其它版本的冲突值不同，制造版本间矛盾",
                f"总共注入约 {num_conflicts} 个冲突",
                "输出必须为严格的JSON格式",
                "每个冲突必须包含 'entity_text' 字段记录被修改的原始实体"
            ],
            "output_example": {
                "injected_text": "这是注入冲突后的完整文本。",
                "conflicts": [
                    {
                        "entity_text": "2025年",  # 新增：记录被修改的原始实体
                        "original_span": "2025年",
                        "injected_span": "2024年", 
                        "type": "数字",
                        "note": "修改年份",
                        "target_entity": "DATE"  # 新增：实体类型
                    }
                ]
            },
            "important": "必须输出纯JSON，不要任何额外文字说明"
        }

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(user_prompt, ensure_ascii=False, indent=2))
        ]

        try:
            raw_text = self._call_llm(messages)
            if not raw_text or not isinstance(raw_text, str):
                raise ValueError("LLM 未返回文本")

            json_text, reason = extract_json_from_text(raw_text)
            if not json_text:
                logger.warning(f"LLM 未返回可解析的 JSON (reason={reason})")
                return self._create_fallback_response(version_text)

            parsed = json.loads(json_text)

            if "injected_text" not in parsed or "conflicts" not in parsed:
                logger.warning("LLM 返回的JSON缺少必要字段")
                return self._create_fallback_response(version_text)

            injected_text = parsed["injected_text"]
            conflicts = parsed["conflicts"]

            # 添加 span 位置信息
            for conflict in conflicts:
                if "injected_span" in conflict:
                    start, end = locate_span(injected_text, conflict["injected_span"])
                    conflict["span_start"] = start
                    conflict["span_end"] = end
                else:
                    conflict["span_start"] = None
                    conflict["span_end"] = None

            return parsed

        except Exception as e:
            logger.exception(f"LLM inject_conflicts_for_version error: {e}")
            return self._create_fallback_response(version_text)

    def _create_fallback_response(self, original_text: str) -> Dict[str, Any]:
        return {
            "injected_text": original_text,
            "conflicts": [{
                "entity_text": "",
                "original_span": "", 
                "injected_span": "", 
                "type": "unknown", 
                "note": "LLM 调用失败", 
                "span_start": None, 
                "span_end": None
            }]
        }


# ---------- 冲突注入器 ----------
class ConflictInjector:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def inject_into_versions(
        self,
        versions: Dict[str, str],
        num_conflicts: int = 6
    ) -> Dict[str, Any]:
        """
        对多个版本进行协同冲突注入
        versions: {"rewritten_v1": "...", "rewritten_v2": "...", ...}
        返回包含详细冲突分析的结果
        """
        # 步骤1: 找出共享实体
        shared_entities = find_shared_entities_across_versions(versions)
        logger.info(f"找到 {len(shared_entities)} 个跨版本共享实体")

        version_ids = list(versions.keys())
        version_results = {}
        all_conflicts = []

        # 步骤2: 对每个版本单独调用 LLM，但传入相同的 shared_entities
        for vid, text in versions.items():
            if not text.strip():
                logger.warning(f"版本 {vid} 为空，跳过")
                version_results[vid] = {
                    "injected_text": text,
                    "conflicts": []
                }
                continue

            logger.info(f"正在为版本 {vid} 注入冲突...")
            result = self.llm.inject_conflicts_for_version(
                version_text=text,
                shared_entities=shared_entities,
                version_id=vid,
                all_version_ids=version_ids,
                num_conflicts=num_conflicts
            )

            # 为每个冲突添加版本信息
            for conflict in result["conflicts"]:
                conflict["version_id"] = vid
                all_conflicts.append(conflict)

            version_results[vid] = {
                "injected_text": result["injected_text"],
                "conflicts": result["conflicts"]
            }

        # 步骤3: 分析跨版本冲突关系
        cross_version_conflicts = self._analyze_cross_version_conflicts(all_conflicts, version_ids)
        
        return {
            "injected_versions": version_results,
            "conflict_analysis": cross_version_conflicts,
            "shared_entities": shared_entities
        }

    def _analyze_cross_version_conflicts(self, all_conflicts: List[Dict], version_ids: List[str]) -> Dict[str, Any]:
        """
        分析跨版本冲突关系
        """
        # 按实体分组冲突
        entity_conflicts = defaultdict(list)
        for conflict in all_conflicts:
            entity_text = conflict.get("entity_text", "")
            if entity_text:
                entity_conflicts[entity_text].append(conflict)

        # 分析每个实体的跨版本冲突
        cross_analysis = {
            "entity_based_conflicts": [],
            "version_pair_conflicts": defaultdict(list),
            "summary": {
                "total_conflicts": len(all_conflicts),
                "entities_with_conflicts": len(entity_conflicts),
                "conflicts_by_type": defaultdict(int),
                "conflicts_by_version": defaultdict(int)
            }
        }

        # 统计基本信息
        for conflict in all_conflicts:
            conflict_type = conflict.get("type", "unknown")
            version_id = conflict.get("version_id", "unknown")
            cross_analysis["summary"]["conflicts_by_type"][conflict_type] += 1
            cross_analysis["summary"]["conflicts_by_version"][version_id] += 1

        # 分析每个实体的跨版本冲突
        for entity_text, conflicts in entity_conflicts.items():
            entity_analysis = {
                "entity_text": entity_text,
                "conflict_versions": [],
                "different_values": set(),
                "conflict_types": set()
            }

            # 收集不同版本的不同值
            versions_covered = set()
            for conflict in conflicts:
                versions_covered.add(conflict["version_id"])
                entity_analysis["different_values"].add(conflict["injected_span"])
                entity_analysis["conflict_types"].add(conflict["type"])

            entity_analysis["conflict_versions"] = list(versions_covered)
            entity_analysis["different_values"] = list(entity_analysis["different_values"])
            entity_analysis["conflict_types"] = list(entity_analysis["conflict_types"])
            
            # 只有多个版本有冲突才记录
            if len(versions_covered) > 1:
                cross_analysis["entity_based_conflicts"].append(entity_analysis)

        # 分析版本对之间的冲突
        for i, vid1 in enumerate(version_ids):
            for vid2 in version_ids[i+1:]:
                pair_conflicts = []
                for entity_analysis in cross_analysis["entity_based_conflicts"]:
                    if vid1 in entity_analysis["conflict_versions"] and vid2 in entity_analysis["conflict_versions"]:
                        # 找到这两个版本对该实体的具体冲突值
                        v1_value = None
                        v2_value = None
                        for conflict in entity_conflicts[entity_analysis["entity_text"]]:
                            if conflict["version_id"] == vid1:
                                v1_value = conflict["injected_span"]
                            elif conflict["version_id"] == vid2:
                                v2_value = conflict["injected_span"]
                        
                        if v1_value and v2_value and v1_value != v2_value:
                            pair_conflicts.append({
                                "entity_text": entity_analysis["entity_text"],
                                f"{vid1}_value": v1_value,
                                f"{vid2}_value": v2_value,
                                "conflict_type": entity_analysis["conflict_types"][0] if entity_analysis["conflict_types"] else "unknown"
                            })
                
                if pair_conflicts:
                    cross_analysis["version_pair_conflicts"][f"{vid1}_{vid2}"] = pair_conflicts

        return cross_analysis


# ---------- I/O 与 主流程 ----------
def load_rewritten_news(path: str) -> List[Dict[str, Any]]:
    """加载重写新闻数据"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"加载文件失败: {e}")
        return []
    
    if isinstance(data, dict) and "top_rewrites" in data:
        item = {
            "rewritten_v1": data["top_rewrites"][0].get("final_rewritten_text", "") if len(data["top_rewrites"]) > 0 else "",
            "rewritten_v2": data["top_rewrites"][1].get("final_rewritten_text", "") if len(data["top_rewrites"]) > 1 else "",
            "rewritten_v3": data["top_rewrites"][2].get("final_rewritten_text", "") if len(data["top_rewrites"]) > 2 else "",
            "input_text": data.get("input_text", "")
        }
        return [item]
    elif isinstance(data, list):
        return data
    else:
        return [data]

def save_output(out: Any, path: str):
    """保存输出结果"""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description="跨版本冲突注入工具")
    parser.add_argument("--input", "-i", required=True, help="输入 JSON 文件（含 top_rewrites）")
    parser.add_argument("--output", "-o", required=True, help="输出 JSON 文件（含注入后的版本与详细冲突分析）")
    parser.add_argument("--conflicts-per-version", type=int, default=6, help="每个版本注入的冲突数量")
    parser.add_argument("--model", default=None, help="模型部署名；若为空则使用默认模型")
    args = parser.parse_args()

    # 获取模型配置
    model_key = args.model or DEFAULT_MODEL_NAME
    try:
        model_config = get_model_deployment_config(model_key)
    except Exception as e:
        logger.warning(f"无法通过 get_model_deployment_config 获取配置: {e}")
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
        logger.error("需要提供 OPENAI API key")
        return

    # 加载数据
    data_list = load_rewritten_news(args.input)
    if not data_list:
        logger.error("没有加载到数据，请检查输入文件")
        return
        
    logger.info(f"成功加载 {len(data_list)} 条记录")

    llm_client = LLMClient(model_config=model_config)
    injector = ConflictInjector(llm_client)

    results = []
    for item in tqdm(data_list, desc="处理文档"):
        input_text = item.get("input_text", "")
        versions = {}
        for vid in ("rewritten_v1", "rewritten_v2", "rewritten_v3"):
            if item.get(vid):
                versions[vid] = item[vid]

        if not versions:
            logger.warning("无有效重写版本，跳过")
            continue

        # 注入冲突
        try:
            injection_result = injector.inject_into_versions(versions, num_conflicts=args.conflicts_per_version)
            
            # 构建输出结构
            output_item = {
                "input_text": input_text,
                "injection_results": injection_result
            }
            
            results.append(output_item)
            
        except Exception as e:
            logger.error(f"处理文档时出错: {e}")
            # 创建错误回退项
            output_item = {
                "input_text": input_text,
                "injection_results": {
                    "injected_versions": {vid: text for vid, text in versions.items()},
                    "conflict_analysis": {
                        "error": str(e),
                        "entity_based_conflicts": [],
                        "version_pair_conflicts": {},
                        "summary": {"total_conflicts": 0, "error_occurred": True}
                    },
                    "shared_entities": []
                }
            }
            results.append(output_item)

    # 保存结果
    save_output(results, args.output)
    logger.info(f"完成！输出写入：{args.output}")
    logger.info("输出包含：")
    logger.info("  - 每个版本的注入后文本")
    logger.info("  - 每个冲突的详细信息（包括位置和版本）") 
    logger.info("  - 跨版本冲突分析")
    logger.info("  - 版本对冲突关系")

if __name__ == "__main__":
    main()