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

# ✅ 修改导入：使用新版 langchain_openai 包
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import langchain

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
    兼容不同 langchain/langchain_openai 版本的返回格式，
    尝试以多种方式提取模型返回的文本：
    - 如果直接返回字符串或有 .content 属性 -> 取出
    - 如果返回 LLMResult（含 .generations） -> 拼接 generations 的 text 或 message.content
    - 如果调用方式需要关键词参数 messages=...，也支持
    返回最终的纯文本（strip 后）。
    """
    try:
        # 首先尝试直接调用（很多版本支持）
        res = llm(messages)
    except TypeError:
        # 有些实现要求关键字参数
        try:
            res = llm(messages=messages)
        except Exception as e:
            logger.exception("LLM call failed with both positional and keyword 'messages': %s", e)
            raise

    # 如果是字符串
    if isinstance(res, str):
        return res.strip()

    # 如果是简单的消息对象（如 AIMessage）
    if hasattr(res, 'content') and isinstance(getattr(res, 'content'), str):
        return res.content.strip()

    # 如果是 LLMResult 风格，包含 .generations
    if hasattr(res, 'generations'):
        texts = []
        try:
            for gen_list in res.generations:
                for gen in gen_list:
                    # 在不同版本中，候选文本可能叫 text，或 message.content
                    txt = getattr(gen, 'text', None)
                    if not txt and hasattr(gen, 'message') and getattr(gen.message, 'content', None):
                        txt = gen.message.content
                    if txt:
                        texts.append(txt)
            return "\n".join(texts).strip()
        except Exception:
            pass

    # 如果是包含 .choices（OpenAI 衍生），尝试解析
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

    # 如果是列表或其他可迭代
    if isinstance(res, list) and res:
        try:
            # 取第一个元素的 content 或直接 str
            first = res[0]
            if hasattr(first, 'content'):
                return first.content.strip()
            return str(first).strip()
        except Exception:
            pass

    # 兜底
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
        # ChatOpenAI 的构造在不同版本里参数名可能不同，常见为 openai_api_key/openai_api_base
        return ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=api_key,
            openai_api_base=base_url,
            request_timeout=REQUEST_TIMEOUT
        )

    def direct_fusion_basic(self, versions: List[str]) -> Dict[str, Any]:
        """
        使用单个清晰的融合 prompt（角色设定 + 输出要求），只返回融合后的文章文本（不再要求对齐 JSON）。
        返回结构包含 fused_text（模型输出的全文），以及一些处理日志信息。
        严格要求模型：
         - 只使用三个版本中提供的信息，不得引入外部事实或假设；
         - 输出仅为融合后的新闻文本，不得包含对齐信息、元数据或解释性 JSON；
         - 以与输入相同的语言输出（若输入语言混合，则以主要语言为准）；
         - 中性新闻语气，简洁、连贯、可发布。
        """
        # 保证有 3 个元素（可能为空字符串）
        while len(versions) < 3:
            versions.append("")
        v1, v2, v3 = versions[:3]

        # 精简而明确的融合 prompt（中文说明）。根据需要可改为英文或自动检测语言。
        # 该 prompt 包含角色设定、严格输出要求（不输出对齐）。
        fusion_prompt = f"""
你是资深新闻编辑。任务：将下面提供的三份新闻改写版本合成为一篇连贯、中性的新闻报道文本。
请严格遵守以下要求：
1) 仅使用三份版本（标记为 Version 1 / Version 2 / Version 3）中出现的信息。不得添加、推断或引入任何外部事实、数据或时间点。
2) 如果三份版本之间存在事实冲突（如日期、地点、数字等），请在文中选择最有依据或更详尽的表述；但不要在文本中解释冲突处理过程，也不要添加注释。冲突解析可在外部流程中完成，这里只需给出最终文本。
3) 输出内容只包含最终融合后的新闻文章本身（纯文本）。不要输出任何对齐信息、JSON、注释、元数据或处理日志。
4) 语言与输入相同（若三版主要为中文，请输出中文；若为英文，请输出英文）。文本应中性、简洁、可发布。
5) 保持段落结构清晰，句子流畅，避免重复或矛盾表达。
6) 如果某条信息仅出现在某一版本且表述不详，仍可在不引入新事实的前提下整合该表述（保持原话意，不扩展细节）。

下面是三份版本的原文（按顺序为 Version 1 / Version 2 / Version 3）。请基于它们产出融合后的文章并直接返回该全文。

--- Version 1 ---
{v1}

--- Version 2 ---
{v2}

--- Version 3 ---
{v3}

请开始产出融合文章（仅输出最终文章文本）。
"""
        messages = [
            SystemMessage(content="You are a professional news editor. Produce a single coherent news article based only on given versions; do not output alignment or JSON."),
            HumanMessage(content=fusion_prompt)
        ]

        try:
            response_text = call_chat_model_and_get_text(self.llm, messages)
            fused_text = response_text.strip()

            # 清理可能的模型前缀
            fused_text = re.sub(
                r'^(融合后的新闻:|融合结果:|最终版本:|融合版本:|示例输出:)',
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
        # 如果没有至少两版可用，直接返回原样（不调用模型）
        if len([v for v in versions if v.strip()]) < 2:
            # 保持兼容：把可能存在的单版直接复制为 fused_content
            news["fused_content"] = versions[0] if versions else news.get("input_text", "")
            news["fusion_method"] = self.fusion_method
            return news
       
        result = self.agent.direct_fusion_basic(versions)
        news["fused_content"] = result.get("fused_text", "")
        # 不再生成 alignment 或 raw_alignment
        news["fusion_method"] = self.fusion_method
        # 可选保存模型处理日志，便于排查
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


# ====== 修改后的 load_rewritten_news：兼容多种输入格式（包含示例的 rewrite_versions 数组） ======
def load_rewritten_news(path: str) -> List[Dict[str, Any]]:
    """
    兼容多种重写结果的输入格式，返回标准化的 list[dict]，
    每个 dict 包含至少：id (若有)、input_text、rewritten_v1/rewritten_v2/rewritten_v3
    支持的输入形式包括（但不限于）：
      - 顶层为 list，每个元素为示例格式：含 rewrite_versions 列表，每项含 final_rewritten_text 等字段
      - 顶层为 dict，含 top_rewrites / rewrite_versions（单条记录）
      - 直接包含 rewritten_v1/rewritten_v2/rewritten_v3 字段
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"load_rewritten_news open/load error: {e}", exc_info=True)
        return []

    def _extract_text_from_entry(entry):
        # entry 可能是 dict，也可能直接是字符串
        if isinstance(entry, dict):
            # 常见字段优先级： final_rewritten_text / final_text / text / content / summary
            return entry.get('final_rewritten_text') or entry.get('final_text') or entry.get('text') or entry.get('content') or entry.get('summary') or ''
        elif isinstance(entry, str):
            return entry
        else:
            return ''

    def normalize_item(item: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        # 可能存在 id 字段
        out['id'] = item.get('id') or item.get('news_id') or item.get('item_id') or ''
        # 原始输入文本
        out['input_text'] = item.get('input_text') or item.get('text') or item.get('original_text') or item.get('input') or ''
        # 尝试处理 rewrite 列表（示例中的 rewrite_versions 或旧的 top_rewrites）
        rewrite_list = item.get('rewrite_versions') or item.get('top_rewrites') or item.get('rewrites') or []
        if not isinstance(rewrite_list, list):
            rewrite_list = []

        # 优先处理当项已经包含直接字段 rewritten_v1..3
        if 'rewritten_v1' in item or 'rewritten_v2' in item or 'rewritten_v3' in item:
            out['rewritten_v1'] = item.get('rewritten_v1', '') or ''
            out['rewritten_v2'] = item.get('rewritten_v2', '') or ''
            out['rewritten_v3'] = item.get('rewritten_v3', '') or ''
            # 如果 input_text 为空，尝试从原字段中补
            if not out['input_text']:
                out['input_text'] = item.get('input_text') or item.get('original_text') or ''
            return out

        # 从 rewrite_list 提取前三个文本
        for i in range(3):
            key = f'rewritten_v{i+1}'
            if i < len(rewrite_list):
                out[key] = _extract_text_from_entry(rewrite_list[i])
            else:
                out[key] = ''

        # 兼容 top_rewrites 中的对象形式（如每项含 final_rewritten_text）
        if not any(out[f'rewritten_v{i+1}'] for i in range(3)):
            # 如果 item 本身含有 top_rewrites 列表（当作单条记录）
            tr = item.get('top_rewrites')
            if isinstance(tr, list) and tr:
                for i in range(min(3, len(tr))):
                    out[f'rewritten_v{i+1}'] = _extract_text_from_entry(tr[i])

        # 如果仍未获得任何重写文本，尝试 item 中其他可能字段（兼容 legacy）
        if not any(out[f'rewritten_v{i+1}'] for i in range(3)):
            # 直接尝试从 item 的一些常见键提取
            out['rewritten_v1'] = out.get('rewritten_v1') or item.get('final_rewritten_text') or item.get('final_text') or ''
            out['rewritten_v2'] = out.get('rewritten_v2') or item.get('alt_rewrite_1') or ''
            out['rewritten_v3'] = out.get('rewritten_v3') or item.get('alt_rewrite_2') or ''

        return out

    normalized: List[Dict[str, Any]] = []
    # 顶层为 dict（单条或包裹）
    if isinstance(data, dict):
        # 如果顶层就是一个包含多条记录的容器（例如 key -> list），尽量发现 list 值并解包
        if any(k in data for k in ('rewrite_versions', 'top_rewrites', 'rewrites', 'rewritten_v1')):
            # 直接把这个 dict 作为一条记录处理
            normalized.append(normalize_item(data))
        else:
            # 尝试在 dict 中寻找可能的 list-of-records 字段
            found_list = None
            for k, v in data.items():
                if isinstance(v, list) and v:
                    # 列表元素为 dict 且包含我们期望的字段，认为是记录集合
                    if isinstance(v[0], dict) and any(key in v[0] for key in ('rewrite_versions', 'top_rewrites', 'rewritten_v1', 'input_text', 'id')):
                        found_list = v
                        break
            if found_list is not None:
                for it in found_list:
                    if isinstance(it, dict):
                        normalized.append(normalize_item(it))
            else:
                # 回退为把整个 dict 当作一条记录处理
                normalized.append(normalize_item(data))
    elif isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                normalized.append(normalize_item(entry))
            else:
                # 如果列表项是字符串或其他，仅填充 input_text
                normalized.append({
                    'id': '',
                    'input_text': entry if isinstance(entry, str) else '',
                    'rewritten_v1': '',
                    'rewritten_v2': '',
                    'rewritten_v3': ''
                })
    else:
        # 非法格式
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
    # python fusion_system2direct.py   --input /data1/rjj/a.test/all_rewritten/summary_finance.json --output ./fused_news/outputdeepseek-chat.json   --model qwen-7b   --fusion-method direct

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