

#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import json
import time
import argparse
import logging
import re
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from config import (
    get_model_deployment_config,
    DEFAULT_TEACHER_MODEL,
    MAX_WORKERS,
    REQUEST_TIMEOUT,
    LOG_LEVEL,
    DEBUG_MODE,
    ENABLE_PROGRESS_BAR,
    API_RETRY_LIMIT
)

logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

ALIGNMENT_SEPARATOR = "====ALIGNMENT===="

def init_llm_from_config(model_name: str):
    cfg = get_model_deployment_config(model_name)
    model_name_cfg = cfg.get("model_name", cfg.get("model", model_name))
    temperature = cfg.get("temperature", 0.0)
    max_tokens = cfg.get("max_tokens", 4000)
    api_key = cfg.get("api_key", "")
    base_url = cfg.get("base_url", "https://api.openai.com/v1")
    if api_key == "EMPTY":
        api_key = "vllm"
    if DEBUG_MODE:
        logger.info(f"init_llm_from_config: model={model_name_cfg}, temp={temperature}, base_url={base_url}")
    return ChatOpenAI(
        model_name=model_name_cfg,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=api_key,
        openai_api_base=base_url,
        request_timeout=REQUEST_TIMEOUT,
        max_retries=API_RETRY_LIMIT
    )

def safe_llm_invoke(llm, messages: List[Any], max_retries: int = 3, base_delay: float = 1.0):
    """简单的重试封装，返回LLM的原始响应对象或None"""
    for attempt in range(max_retries + 1):
        try:
            # langchain_community ChatOpenAI 在某些版本上支持 .invoke(messages)
            resp = llm.invoke(messages)
            return resp
        except Exception as e:
            logger.warning(f"LLM invoke failed (attempt {attempt+1}/{max_retries+1}): {e}")
            if attempt < max_retries:
                time.sleep(base_delay * (2 ** attempt))
            else:
                logger.error("LLM invoke exhausted retries.")
                return None

def extract_alignment_from_text(text: str) -> Optional[List[Dict[str, Any]]]:
    """尝试从模型输出文本中提取 ALIGNMENT JSON（在分隔符后），稳健解析 JSON"""
    if ALIGNMENT_SEPARATOR in text:
        parts = text.split(ALIGNMENT_SEPARATOR, 1)
        json_part = parts[1].strip()
        # 尝试直接解析
        try:
            return json.loads(json_part)
        except Exception:
            # 尝试提取第一个 JSON array/braced block
            m = re.search(r'(\[\s*\{.*\}\s*\])', json_part, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    pass
            m2 = re.search(r'(\{.*\})', json_part, re.DOTALL)
            if m2:
                try:
                    parsed = json.loads(m2.group(1))
                    # 如果是单个对象， wrap 成 list
                    if isinstance(parsed, dict):
                        return [parsed]
                except Exception:
                    pass
    else:
        # 可能模型只输出 JSON（没有分隔符）或直接返回 JSON
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                return [parsed]
        except Exception:
            # 尝试提取 first JSON array anywhere
            m = re.search(r'(\[\s*\{.*\}\s*\])', text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    pass
    return None

def build_alignment_prompt(fused_text: str, versions: List[str]) -> str:
    """
    构造给 LLM 的对齐 prompt（中文）。包含严格的输出格式要求：
    - 输出必须包含原始融合文章（可重复）/或者仅提供对齐 JSON，但必须提供 ALIGNMENT 分隔符和 JSON 数组。
    - JSON 每一项包含 sent_idx, text, sources, evidence, conflict_resolved。
    """
    # 附带一个简短示例以便 LLM 模仿输出结构
    example_alignment = """
示例输出格式：
====ALIGNMENT====
[
  {
    "sent_idx": 1,
    "text": "示例句子 A。",
    "sources": [1, 2],
    "evidence": [
      "版本1中的对应片段...",
      "版本2中的对应片段..."
    ],
    "conflict_resolved": "若存在冲突，请简要说明如何选择或为什么选用某版本；若无冲突，留空字符串。"
  },
  ...
]
"""
    prompt = f"""
你是一名资深新闻编辑/信息对齐专家。任务：将下面的“融合后文章”（Fused Article）中的每一句话，与提供的三个改写版本（Version 1 / Version 2 / Version 3）对齐，输出一个 JSON 数组描述每个句子的来源和证据。

严格要求（非常重要）：
1) 只能使用提供的三份改写版本里的信息作为证据。不得添加外部事实或额外信息。
2) 输出格式必须严格为：在一行单独写出分隔符行：{ALIGNMENT_SEPARATOR}，随后是一个 JSON 数组（可多行、可缩进），数组的每个元素对应融合文章中的一个句子（按顺序）。
3) 每个数组元素必须包含以下字段：
   - sent_idx: 句子序号（从1开始，与融合文章中句子顺序一致）
   - text: 该句子在融合文章中的原文（应与融合文章中句子逐字匹配）
   - sources: 一个整数数组，包含支持该句子的源版本编号（1/2/3）
   - evidence: 对应 sources 中每个版本的最小支持性摘录（短语/子句/完整句均可），按 sources 顺序排列；摘录应直接从对应版本文本中复制，不要改写。
   - conflict_resolved: 如果不同版本在该句子事实上存在冲突（如日期/地点/数字不一致），请简要说明冲突是什么、你选择了哪个版本（或哪种规则），以及为何选择；如果没有冲突则填空字符串 ""。
4) JSON 必须能被 JSON 解析器解析（即合法 JSON）。
5) 融合文章句子划分：请按常规书面语句号/问号/感叹号/换行划分（保留融合文章中原句标点）。如果句子非常长或包含多条事实，你可以适当拆分，但请保证 text 字段精确反映融合文章中对应句子片段。
6) 在 evidence 中，尽量只包含支持该句子最小必要片段，不要包含非支持性内容。
7) 输出仅包含分隔符行和 JSON 数组（可附带融合文章原文之前/之后不影响解析的少量文本，但强烈建议只输出分隔符 + JSON）。不要输出额外的解释、注释或其他格式化文本。

现在的输入如下（请严格根据它们生成对齐 JSON）：

---- FUSED ARTICLE ----
{fused_text}

---- VERSION 1 ----
{versions[0] if len(versions) > 0 else ""}

---- VERSION 2 ----
{versions[1] if len(versions) > 1 else ""}

---- VERSION 3 ----
{versions[2] if len(versions) > 2 else ""}

{example_alignment}

请开始并只输出分隔符与 JSON 数组。
"""
    return prompt

def align_single_item(item: Dict[str, Any], llm) -> Dict[str, Any]:
    """
    对单篇新闻进行对齐：调用 LLM 得到 alignment JSON，解析并把 alignment 附回 fused_content。
    返回修改后的 item（添加 fused_content（包含 ALIGNMENT），以及 alignment 字段或 raw_alignment 字段）
    """
    fused = item.get("fused_content", "") or item.get("fused", "") or ""
    if not fused.strip():
        logger.warning("item missing fused_content, skipping alignment")
        item["alignment_error"] = "missing_fused_content"
        return item

    versions = []
    for k in ("rewritten_v1", "rewritten_v2", "rewritten_v3"):
        versions.append(item.get(k, "") or "")

    prompt = build_alignment_prompt(fused, versions)
    messages = [
        SystemMessage(content="你是一个严谨的文本对齐助手，按要求输出 ALIGNMENT JSON。"),
        HumanMessage(content=prompt)
    ]

    resp = safe_llm_invoke(llm, messages, max_retries=API_RETRY_LIMIT, base_delay=1.0)
    if not resp:
        item["alignment_error"] = "llm_invoke_failed"
        return item

    # 获取文本内容：不同版本的返回对象访问方式可能不同
    resp_text = ""
    try:
        # langchain response 对象通常在 .content 或 .generations 中
        if hasattr(resp, "content"):
            resp_text = resp.content
        elif hasattr(resp, "text"):
            resp_text = resp.text
        else:
            resp_text = str(resp)
    except Exception:
        resp_text = str(resp)

    alignment = extract_alignment_from_text(resp_text)
    if alignment is None:
        # 尝试在整个响应中用正则提取 JSON 数组
        alignment = None

    if alignment is not None:
        # 将 alignment JSON 附回 fused_content（按原 DualEvaluation 的提取逻辑）
        try:
            json_str = json.dumps(alignment, ensure_ascii=False, indent=2)
            item["fused_content"] = fused.rstrip() + "\n" + ALIGNMENT_SEPARATOR + "\n" + json_str
            item["alignment"] = alignment
            item["_alignment_processing_log"] = ["success", "parsed_and_attached_alignment"]
        except Exception as e:
            item["_alignment_processing_log"] = ["success_parsed_but_attach_failed", str(e)]
            item["raw_alignment"] = resp_text
    else:
        # 解析失败：把原始输出保留在 raw_alignment 字段并标记 error
        item["_alignment_processing_log"] = ["parse_failed"]
        item["raw_alignment"] = resp_text
        item["alignment_error"] = "parse_failed"

    return item

def process_batch(news_list: List[Dict[str, Any]], llm, max_workers: int = 4) -> List[Dict[str, Any]]:
    results = []
    with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(news_list)))) as executor:
        futures = {executor.submit(align_single_item, item, llm): item for item in news_list}
        iterator = tqdm(as_completed(futures), total=len(futures)) if ENABLE_PROGRESS_BAR else as_completed(futures)
        for fut in iterator:
            try:
                results.append(fut.result())
            except Exception as e:
                logger.error("align task failed: %s", e)
                # 把出错的原始 item 加回列表以便诊断
                original = futures.get(fut, {})
                if original:
                    original_copy = original.copy()
                    original_copy["alignment_error"] = f"exception:{e}"
                    results.append(original_copy)
    return results

def load_input(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # 如果是 dict，尝试寻找列表字段或把 dict 当作单条记录
            # 常见直接顶层为单条记录
            logger.warning("input JSON is object, wrapping into list")
            return [data]
        else:
            logger.error("unsupported input JSON root type")
            return []
    except Exception as e:
        logger.error("load_input failed: %s", e)
        return []

def save_output(news_list: List[Dict[str, Any]], output_path: str):
    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(news_list, f, ensure_ascii=False, indent=2)
        logger.info("saved %d items to %s", len(news_list), output_path)
    except Exception as e:
        logger.error("save_output failed: %s", e)

def parse_args():
    p = argparse.ArgumentParser(description="Fused article -> Alignment helper (attach ====ALIGNMENT==== JSON to fused_content)")
    p.add_argument("--input", "-i", required=True, help="input JSON file (list of news items)")
    p.add_argument("--output", "-o", required=True, help="output JSON file (will include alignment fields)")
    p.add_argument("--model", "-m", default=None, help="model config name (from get_model_deployment_config), default DEFAULT_TEACHER_MODEL")
    p.add_argument("--workers", "-w", type=int, default=MAX_WORKERS, help="max workers for parallel alignment")
    p.add_argument("--sample", "-s", type=int, default=0, help="sample N items for quick test")
    return p.parse_args()
# python aligner.py --input ./fused_news/fused_output.json --output fused_with_alignment.json --model deepseek-chat --workers 4
def main():
    args = parse_args()
    model_name = args.model or DEFAULT_TEACHER_MODEL
    items = load_input(args.input)
    if not items:
        print("no items loaded")
        sys.exit(1)
    if args.sample > 0:
        items = items[: args.sample]
        logger.info("sampling %d items for alignment", len(items))

    llm = init_llm_from_config(model_name)
    aligned = process_batch(items, llm, max_workers=args.workers)
    save_output(aligned, args.output)
    logger.info("alignment finished. output saved to %s", args.output)

if __name__ == "__main__":
    main()


#   python aligner.py --input fused_news.json --output fused_with_alignment.json --model deepseek-chat --workers 4
