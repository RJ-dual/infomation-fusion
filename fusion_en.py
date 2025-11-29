#!/usr/bin/env python3
# coding: utf-8

import json
import logging
import os
import sys
import argparse
import re
import random
import numpy as np
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict

# ✅ 使用新版本的导入方式
try:
    # 新版本 langchain_core (推荐)
    from langchain_core.messages import SystemMessage, HumanMessage
    print("✅ Using langchain_core.messages")
except ImportError as e:
    try:
        # 旧版本回退
        from langchain.schema import SystemMessage, HumanMessage
        print("✅ Using langchain.schema (legacy)")
    except ImportError:
        # 最终回退：自定义实现
        class SystemMessage:
            def __init__(self, content):
                self.content = content
                self.type = "system"
        
        class HumanMessage:
            def __init__(self, content):
                self.content = content
                self.type = "human"
        print("⚠️ Using custom message classes")

try:
    from langchain_openai import ChatOpenAI
    print("✅ Using langchain_openai.ChatOpenAI")
except ImportError:
    try:
        from langchain.chat_models import ChatOpenAI
        print("✅ Using langchain.chat_models.ChatOpenAI (legacy)")
    except ImportError:
        print("❌ Error: Could not import ChatOpenAI")
        sys.exit(1)

# 确保安装了必要的包
try:
    import langchain_core
    import openai
    print(f"✅ langchain_core version: {getattr(langchain_core, '__version__', 'unknown')}")
    print(f"✅ openai version: {getattr(openai, '__version__', 'unknown')}")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")

from config import (
    get_model_deployment_config, DEFAULT_STUDENT_MODEL,
    MAX_WORKERS, BATCH_SIZE, REQUEST_TIMEOUT, LOG_LEVEL, DEBUG_MODE,
    ENABLE_PROGRESS_BAR
)

logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

class FusionMethod:
    DIRECT = "direct"


def call_chat_model_and_get_text(llm, messages: List[Any]) -> str:
    """
    Compatible with different langchain/langchain_openai version return formats,
    try multiple ways to extract model response text:
    - If directly returns string or has .content attribute -> extract
    - If returns LLMResult (with .generations) -> concatenate generations text or message.content
    - If calling method requires keyword arguments messages=..., also supported
    Returns final plain text (stripped).
    """
    try:
        # First try direct call (many versions support)
        res = llm(messages)
    except TypeError:
        # Some implementations require keyword arguments
        try:
            res = llm(messages=messages)
        except Exception as e:
            logger.exception("LLM call failed with both positional and keyword 'messages': %s", e)
            raise

    # If it's a string
    if isinstance(res, str):
        return res.strip()

    # If it's a simple message object (like AIMessage)
    if hasattr(res, 'content') and isinstance(getattr(res, 'content'), str):
        return res.content.strip()

    # If it's LLMResult style, contains .generations
    if hasattr(res, 'generations'):
        texts = []
        try:
            for gen_list in res.generations:
                for gen in gen_list:
                    # In different versions, candidate text might be called text, or message.content
                    txt = getattr(gen, 'text', None)
                    if not txt and hasattr(gen, 'message') and getattr(gen.message, 'content', None):
                        txt = gen.message.content
                    if txt:
                        texts.append(txt)
            return "\n".join(texts).strip()
        except Exception:
            pass

    # If it contains .choices (OpenAI derived), try to parse
    if hasattr(res, 'choices'):
        try:
            parts = []
            for c in res.choices:
                if isinstance(c, dict) and 'message' in c:
                    parts.append(c['message'].get('content', ''))
                else:
                    parts.append(str(c))
            return "\n".join(parts).strip()
        except Exception:
            pass

    # If it's a list or other iterable
    if isinstance(res, list) and res:
        try:
            # Take content of first element or direct str
            first = res[0]
            if hasattr(first, 'content'):
                return first.content.strip()
            return str(first).strip()
        except Exception:
            pass

    # Fallback
    return str(res).strip()


class DirectFusionAgent:
    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        self.model_name = model_config.get("model", "unknown")
        self.llm = self._initialize_llm(model_config)
        if DEBUG_MODE:
            logger.info(f"init DirectFusionAgent model={self.model_name}")

    def _initialize_llm(self, model_config: Dict[str, Any]) -> ChatOpenAI:
        temperature = model_config.get("temperature", 0.2)
        max_tokens = model_config.get("max_tokens", 4000)
        model_name = model_config.get("model_name", model_config.get("model", "gpt-4o-mini"))
        api_key = model_config.get("api_key", "")
        base_url = model_config.get("base_url", "https://api.openai.com/v1")
        if api_key == "EMPTY":
            api_key = "vllm"
        # ChatOpenAI construction parameters may vary in different versions, commonly openai_api_key/openai_api_base
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=api_key,
            openai_api_base=base_url,
            request_timeout=REQUEST_TIMEOUT
        )

    def direct_fusion_basic(self, versions: List[str]) -> Dict[str, Any]:
        """
        Use a single clear fusion prompt (role setting + output requirements), only return the fused article text 
        (no longer requiring alignment JSON).
        Return structure includes fused_text (full text output by the model), and some processing log information.
        Strictly require the model to:
         - Only use information provided in the three versions, do not introduce external facts or assumptions;
         - Output only the fused news text, do not include alignment information, metadata or explanatory JSON;
         - Output in the same language as the input (if input languages are mixed, use the main language);
         - Neutral news tone, concise, coherent, and publishable.
        """
        # Ensure there are 3 elements (possibly empty strings)
        while len(versions) < 3:
            versions.append("")
        v1, v2, v3 = versions[:3]

        # Concise and clear fusion prompt (English instructions). Can be changed to other languages or auto-detected as needed.
        # This prompt includes role setting and strict output requirements (no alignment output).
        fusion_prompt = f"""
You are a senior news editor. Task: Synthesize the three provided rewritten news versions into one coherent, neutral news article text.

Please strictly adhere to the following requirements:
1) Use ONLY the information that appears in the three versions (marked as Version 1 / Version 2 / Version 3). Do not add, infer, or introduce any external facts, data, or time points.
2) If there are factual conflicts between the three versions (such as dates, locations, numbers, etc.), choose the most substantiated or detailed expression in the text; but do not explain the conflict resolution process in the text, nor add annotations. Conflict resolution can be done in external processes, here only the final text is needed.
3) The output content should only contain the final fused news article itself (plain text). Do not output any alignment information, JSON, annotations, metadata, or processing logs.
4) The language should be the same as the input (if the three versions are mainly in Chinese, output in Chinese; if in English, output in English). The text should be neutral, concise, and publishable.
5) Maintain clear paragraph structure, fluent sentences, and avoid repetitive or contradictory expressions.
6) If certain information only appears in one version and is poorly expressed, you can still integrate that expression without introducing new facts (maintain the original meaning without extending details).

Below are the original texts of the three versions (in order: Version 1 / Version 2 / Version 3). Please produce the fused article based on them and return the full text directly.

--- Version 1 ---
{v1}

--- Version 2 ---
{v2}

--- Version 3 ---
{v3}

Please begin producing the fused article (output only the final article text).
"""
        messages = [
            SystemMessage(content="You are a professional news editor. Produce a single coherent news article based only on given versions; do not output alignment or JSON."),
            HumanMessage(content=fusion_prompt)
        ]

        try:
            response_text = call_chat_model_and_get_text(self.llm, messages)
            fused_text = response_text.strip()

            # Clean possible model prefixes
            fused_text = re.sub(
                r'^(Fused News:|Fusion Result:|Final Version:|Fusion Version:|Example Output:)',
                '',
                fused_text,
                flags=re.IGNORECASE
            ).strip()

            return {
                'fused_text': fused_text,
                'fusion_method': FusionMethod.DIRECT,
                'processing_log': ["direct_basic_single_prompt_success"],
                'coverage_stats': {'fusion_method': 'direct_single_prompt'}
            }

        except Exception as e:
            logger.error(f"direct_fusion_basic single-prompt error: {e}", exc_info=True)
            return {
                'fused_text': versions[0] if versions else "",
                'fusion_method': FusionMethod.DIRECT,
                'processing_log': [f"error:{str(e)}"],
                'coverage_stats': {'error': str(e)}
            }


class UnifiedFusionProcessor:
    def __init__(self, model_name=None, local_similarity_model_path=None, gpu_device="cuda:0", fusion_method=FusionMethod.DIRECT):
        self.model_name = model_name or DEFAULT_STUDENT_MODEL
        self.model_config = get_model_deployment_config(self.model_name)
        self.fusion_method = fusion_method
        
        self.agent = DirectFusionAgent(self.model_config)
        self.max_workers = MAX_WORKERS
        self.batch_size = BATCH_SIZE

    def process_single_news(self, news: Dict[str, Any]) -> Dict[str, Any]:
        versions = [news.get(k, "") for k in ["rewritten_v1", "rewritten_v2", "rewritten_v3"] if news.get(k) is not None]
        # If less than two versions are available, return as-is (without calling the model)
        if len([v for v in versions if v.strip()]) < 2:
            # Maintain compatibility: copy the single version directly as fused_content
            news["fused_content"] = versions[0] if versions else news.get("input_text", "")
            news["fusion_method"] = self.fusion_method
            return news
       
        result = self.agent.direct_fusion_basic(versions)
        news["fused_content"] = result.get("fused_text", "")
        # No longer generate alignment or raw_alignment
        news["fusion_method"] = self.fusion_method
        # Optionally save model processing logs for troubleshooting
        news["_processing_log"] = result.get("processing_log", [])
        return news

    def process_news_batch(self, news_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_single_news, n) for n in news_list]
            iterator = tqdm(as_completed(futures), total=len(futures)) if ENABLE_PROGRESS_BAR else as_completed(futures)
            for f in iterator:
                results.append(f.result())
        return results


class WorkflowGraph:
    def __init__(self):
        self.nodes, self.edges = {}, {}

    def add_node(self, name: str, func):
        self.nodes[name] = func
        self.edges.setdefault(name, [])

    def add_edge(self, src: str, dst: str):
        self.edges.setdefault(src, []).append(dst)

    def run(self, start: str, context: Dict[str, Any]):
        q, seen = [start], set()
        while q:
            n = q.pop(0)
            if n in seen:
                continue
            seen.add(n)
            func = self.nodes.get(n)
            if func:
                context = func(context)
            for nxt in self.edges.get(n, []):
                if nxt not in seen:
                    q.append(nxt)
        return context


# ====== Modified load_rewritten_news: Compatible with multiple input formats (including example rewrite_versions array) ======
def load_rewritten_news(path: str) -> List[Dict[str, Any]]:
    """
    Compatible with multiple rewritten result input formats, returns standardized list[dict],
    Each dict contains at least: id (if any), input_text, rewritten_v1/rewritten_v2/rewritten_v3
    Supported input forms include (but are not limited to):
      - Top level is list, each element is in example format: contains rewrite_versions list, each item contains final_rewritten_text, etc.
      - Top level is dict, contains top_rewrites / rewrite_versions (single record)
      - Directly contains rewritten_v1/rewritten_v2/rewritten_v3 fields
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"load_rewritten_news open/load error: {e}", exc_info=True)
        return []

    def _extract_text_from_entry(entry):
        # entry could be dict, or directly a string
        if isinstance(entry, dict):
            # Common field priority: final_rewritten_text / final_text / text / content / summary
            return entry.get('final_rewritten_text') or entry.get('final_text') or entry.get('text') or entry.get('content') or entry.get('summary') or ''
        elif isinstance(entry, str):
            return entry
        else:
            return ''

    def normalize_item(item: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        # May have id field
        out['id'] = item.get('id') or item.get('news_id') or item.get('item_id') or ''
        # Original input text
        out['input_text'] = item.get('input_text') or item.get('text') or item.get('original_text') or item.get('input') or ''
        # Try to process rewrite list (rewrite_versions in example or old top_rewrites)
        rewrite_list = item.get('rewrite_versions') or item.get('top_rewrites') or item.get('rewrites') or []
        if not isinstance(rewrite_list, list):
            rewrite_list = []

        # Priority: item already contains direct fields rewritten_v1..3
        if 'rewritten_v1' in item or 'rewritten_v2' in item or 'rewritten_v3' in item:
            out['rewritten_v1'] = item.get('rewritten_v1', '') or ''
            out['rewritten_v2'] = item.get('rewritten_v2', '') or ''
            out['rewritten_v3'] = item.get('rewritten_v3', '') or ''
            # If input_text is empty, try to supplement from original fields
            if not out['input_text']:
                out['input_text'] = item.get('input_text') or item.get('original_text') or ''
            return out

        # Extract first three texts from rewrite_list
        for i in range(3):
            key = f'rewritten_v{i+1}'
            if i < len(rewrite_list):
                out[key] = _extract_text_from_entry(rewrite_list[i])
            else:
                out[key] = ''

        # Compatible with object form in top_rewrites (e.g., each item contains final_rewritten_text)
        if not any(out[f'rewritten_v{i+1}'] for i in range(3)):
            # If the item itself contains a top_rewrites list (treat as single record)
            tr = item.get('top_rewrites')
            if isinstance(tr, list) and tr:
                for i in range(min(3, len(tr))):
                    out[f'rewritten_v{i+1}'] = _extract_text_from_entry(tr[i])

        # If still no rewritten text obtained, try other possible fields in item (compatible legacy)
        if not any(out[f'rewritten_v{i+1}'] for i in range(3)):
            # Directly try to extract from some common keys in item
            out['rewritten_v1'] = out.get('rewritten_v1') or item.get('final_rewritten_text') or item.get('final_text') or ''
            out['rewritten_v2'] = out.get('rewritten_v2') or item.get('alt_rewrite_1') or ''
            out['rewritten_v3'] = out.get('rewritten_v3') or item.get('alt_rewrite_2') or ''

        return out

    normalized: List[Dict[str, Any]] = []
    # Top level is dict (single record or wrapper)
    if isinstance(data, dict):
        # If top level is a container containing multiple records (e.g., key -> list), try to find list values and unpack
        if any(k in data for k in ('rewrite_versions', 'top_rewrites', 'rewrites', 'rewritten_v1')):
            # Directly process this dict as a single record
            normalized.append(normalize_item(data))
        else:
            # Try to find possible list-of-records fields in dict
            found_list = None
            for k, v in data.items():
                if isinstance(v, list) and v:
                    # List elements are dicts and contain fields we expect, consider as record collection
                    if isinstance(v[0], dict) and any(key in v[0] for key in ('rewrite_versions', 'top_rewrites', 'rewritten_v1', 'input_text', 'id')):
                        found_list = v
                        break
            if found_list is not None:
                for it in found_list:
                    if isinstance(it, dict):
                        normalized.append(normalize_item(it))
            else:
                # Fallback: treat entire dict as a single record
                normalized.append(normalize_item(data))
    elif isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                normalized.append(normalize_item(entry))
            else:
                # If list item is string or other, only fill input_text
                normalized.append({
                    'id': '',
                    'input_text': entry if isinstance(entry, str) else '',
                    'rewritten_v1': '',
                    'rewritten_v2': '',
                    'rewritten_v3': ''
                })
    else:
        # Illegal format
        logger.error("load_rewritten_news: unexpected JSON root type: %s", type(data))
        return []

    return normalized
# ====== end load_rewritten_news ======

def save_fused_news(news: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(news, f, ensure_ascii=False, indent=2)


def node_load(ctx):
    a = ctx['args']
    n = load_rewritten_news(a.input)
    if a.sample > 0:
        n = n[:a.sample]
    ctx['news'] = n
    return ctx


def node_init(ctx):
    a = ctx['args']
    ctx['processor'] = UnifiedFusionProcessor(a.model, a.similarity_model_path, a.gpu_device, a.fusion_method)
    return ctx


def node_fuse(ctx):
    p = ctx['processor']
    ctx['fused'] = p.process_news_batch(ctx['news'])
    return ctx


def node_save(ctx):
    a = ctx['args']
    save_fused_news(ctx['fused'], a.output)
    return ctx


def node_stats(ctx):
    f = ctx.get('fused', [])
    ctx['stats'] = {'total': len(f), 'success': sum(1 for x in f if x.get('fused_content'))}
    return ctx


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--model", "-m", default=None)
    p.add_argument("--similarity-model-path", "-smp", default=None)
    p.add_argument("--gpu-device", "-gpu", default="cuda:0")
    p.add_argument("--sample", "-s", type=int, default=0)
    p.add_argument("--fusion-method", "-fm", default=FusionMethod.DIRECT)
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.input):
        print(f"input not found: {args.input}")
        sys.exit(1)
    ctx = {'args': args}
    wf = WorkflowGraph()
    wf.add_node('load', node_load)
    wf.add_node('init', node_init)
    wf.add_node('fuse', node_fuse)
    wf.add_node('save', node_save)
    wf.add_node('stats', node_stats)
    wf.add_edge('load', 'init')
    wf.add_edge('init', 'fuse')
    wf.add_edge('fuse', 'save')
    wf.add_edge('save', 'stats')
    result = wf.run('load', ctx)
    s = result.get('stats', {})
    print(f"done total={s.get('total',0)} success={s.get('success',0)} output={args.output}")


if __name__ == "__main__":
    main()
