#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rewrite_agent_langgraph.py Strict implementation of rewrite agent based on Langgraph (requires langgraph installation).
- Uses StateGraph to register nodes and routing
- Records every tool/model call to state["tool_calls"]
- Node-level trace records to state["trace"]
- Evaluator prioritizes reduced_text as reference
- Iterative rewriting: each iteration directly modifies problematic parts based on previous version
- Evaluation logic modified: passes as long as faithfulness evaluation passes, no longer enforces diversity threshold
- Multi-API key support for faster parallel processing
"""
import json
import random
import re
import time
import uuid
import logging
import copy
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, TypedDict
from types import SimpleNamespace

# Configuration import: can be adjusted based on project
try:
    from config_models import LOCAL_MODELS
except Exception:
    LOCAL_MODELS = {}

try:
    from config_runtime import DEBUG_MODE, REQUEST_TIMEOUT
except Exception:
    DEBUG_MODE = True
    REQUEST_TIMEOUT = 60

try:
    from config import MODEL_CONFIGS, get_model_deployment_config, DEFAULT_TEACHER_MODEL
except Exception:
    MODEL_CONFIGS = {}
    def get_model_deployment_config(model_name: str) -> Dict[str, Any]:
        return MODEL_CONFIGS.get(model_name, {})
    DEFAULT_TEACHER_MODEL = "gpt-4o"

# Force dependency on langgraph
try:
    from langgraph.graph import StateGraph  # type: ignore
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None

# langchain openai wrapper (based on environment)
try:
    from langchain_openai import ChatOpenAI  # type: ignore
except Exception:
    ChatOpenAI = None

try:
    from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
except Exception:
    class SystemMessage:
        def __init__(self, content: str):
            self.content = content
        def __repr__(self):
            return f"SystemMessage({self.content[:30]})"
    class HumanMessage:
        def __init__(self, content: str):
            self.content = content
        def __repr__(self):
            return f"HumanMessage({self.content[:30]})"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rewrite_agent_langgraph")

# ---------------- Types ----------------
class AgentState(TypedDict, total=False):
    news_text: str
    original_text: str
    num_versions: int
    rewrite_attempts: int
    max_attempts: int
    quality_threshold: float
    model_config: Dict[str, Any]
    segments: List[Dict[str, Any]]
    analyzed_segments: List[Dict[str, Any]]
    masks: List[Any]
    reduced_versions: List[Dict[str, Any]]
    rewritten_versions: List[Dict[str, Any]]
    scores: List[Dict[str, Any]]
    best_score: float
    best_version: Dict[str, Any]
    trace: List[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]
    errors: List[str]
    terminated_at: str
    final_status: str
    summary: Dict[str, Any]

# ---------------- Helper Functions ----------------
def _parse_json_from_model_output(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    match = re.search(r'\{[\s\S]*\}', raw)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except Exception:
        try:
            cleaned = match.group().replace("'", '"')
            return json.loads(cleaned)
        except Exception:
            return None

def _create_llm(model_config: Optional[Dict[str, Any]] = None, api_config: Optional[Dict[str, Any]] = None):
    if model_config is None:
        model_config = get_model_deployment_config(DEFAULT_TEACHER_MODEL)
    temperature = model_config.get("temperature", 0.7)
    max_tokens = model_config.get("max_tokens", 2000)
    model_name = model_config.get("model_name", model_config.get("model", DEFAULT_TEACHER_MODEL))
    
    # Use provided API config or get from model_config
    if api_config:
        api_key = api_config.get("api_key", "")
        base_url = api_config.get("base_url", "https://api.openai.com/v1")
    else:
        api_key = model_config.get("api_key", "")
        base_url = model_config.get("base_url", "https://api.openai.com/v1")
    
    if api_key == "EMPTY":
        api_key = "vllm"
    if DEBUG_MODE:
        logger.info(f"Initializing local vLLM: model={model_name}")
    if ChatOpenAI is None:
        raise RuntimeError("ChatOpenAI client unavailable, please install or adapt your LLM client.")
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=api_key,
        openai_api_base=base_url,
        request_timeout=REQUEST_TIMEOUT
    )
    return llm

# Mapping helper: map model output keep_id (various formats) to full_tag_id
def map_keep_ids_to_full(segment_id: str, standardized_tags: List[Dict[str, Any]], keep_raw: List[str]):
    mapped = []
    warnings = []
    tag_by_local = {t["tag_id"]: t["full_tag_id"] for t in standardized_tags if t.get("tag_id")}
    full_ids = {t["full_tag_id"] for t in standardized_tags}
    for kid in (keep_raw or []):
        if not isinstance(kid, str):
            continue
        kid = kid.strip()
        if not kid:
            continue
        # already full form?
        if kid.startswith(segment_id + "_"):
            if kid in full_ids:
                mapped.append(kid)
            else:
                mapped.append(kid)
                warnings.append(f"full_id_format_not_found:{kid}")
            continue
        # exact local tag_id match
        if kid in tag_by_local:
            mapped.append(tag_by_local[kid])
            continue
        # exact match to any full id
        if kid in full_ids:
            mapped.append(kid)
            continue
        # numeric heuristic
        d = re.search(r'\d+', kid)
        if d:
            num = d.group(0)
            found = False
            for cand_local, cand_full in tag_by_local.items():
                if num in cand_local or num in cand_full:
                    mapped.append(cand_full)
                    found = True
                    break
            if found:
                continue
            cand = f"{segment_id}_local_{num}"
            if cand in full_ids:
                mapped.append(cand)
                continue
        # text match against tag text/sentence
        text_match = next((t["full_tag_id"] for t in standardized_tags if (t.get("text") or "").strip() == kid or (t.get("sentence") or "").strip() == kid), None)
        if text_match:
            mapped.append(text_match)
            continue
        # fallback
        fallback = f"{segment_id}_{kid}"
        mapped.append(fallback)
        warnings.append(f"fallback_mapped:{kid}->{fallback}")
    deduped = list(dict.fromkeys(mapped))
    return deduped, warnings

# ---------------- Tools ----------------
def create_segmenter_tool():
    def segmenter(input_data: Dict[str, Any]) -> Dict[str, Any]:
        text = input_data.get("text", "") if isinstance(input_data, dict) else ""
        try:
            if not isinstance(text, str):
                raise ValueError("Parameter 'text' must be a string")
            pattern = re.compile(r'(?s).*?(?:\n\s*\n|$)')
            segments = []
            for idx, m in enumerate(pattern.finditer(text), start=1):
                raw = m.group()
                trimmed = raw.strip()
                if not trimmed:
                    continue
                local_off = raw.find(trimmed)
                start_char = m.start() + local_off
                end_char = start_char + len(trimmed)
                segments.append({
                    "segment_id": f"seg_{idx:03d}",
                    "text": trimmed,
                    "start_char": start_char,
                    "end_char": end_char
                })
            return {"segments": segments}
        except Exception as e:
            return {"error": str(e)}
    return SimpleNamespace(name="segmenter", func=segmenter)

def create_analyze_segment_tool(model_config: Dict[str, Any], max_retries: int = 2):
    def analyze_segment(input_data: Dict[str, Any]) -> Dict[str, Any]:
        segment_id = input_data.get("segment_id", "seg_unknown")
        text = input_data.get("text", "")
        if not isinstance(text, str) or not text.strip():
            return {"segment_id": segment_id, "tags": [], "reduced_versions": [], "tagged_text": "", "raw_model_output": ""}
        
        # Get API config for this specific call
        try:
            from config_api import get_api_config
            api_config = get_api_config()
        except Exception:
            api_config = None
            
        system_prompt = (
            "You are a professional news content analyst. Task: 1. Analyze the following news paragraph, identify key information and label with tags. 2. Analyze in detail which information in the original text can be independently rewritten to obtain multiple new articles related to this topic, and fill the corresponding tags into reduced_versions.\n\n"
            " Strictly output JSON, strictly do not output any additional text or explanation.\n\n"
            "Output structure requirements (must be followed):\n"
            "- tags: list, each element contains fields {\"tag_id\": \"local_id\", \"tag_type\": ..., \"text\": ..., \"sentence\": \"...\", \"reason\": \"...\"}\n"
            "- reduced_versions: list, containing several versions (e.g., reduced_v1, reduced_v2, reduced_v3), each version must contain:\n"
            " {\"version_id\":\"reduced_v1\", \"keep_tag_ids\": [\"{segment_id}_{tag_id}\" or \"tag_id\"], \"suggestions\": \"Rewriting direction and suggested short sentences\", \"reason\":\"Reason for selecting these contents in this version\"}\n"
            "Explanation: Model can output keep_tag_ids in two forms:\n"
            " 1) Already in full format '{segment_id}_{tag_id}' (recommended);\n"
            " 2) Can also output only local tag_id (e.g., 'tag1'), backend will attempt to map to full_tag_id ({segment_id}_tag1).\n"
            "Important: keep_tag_ids represent \"tags that should be kept in this version\". Tags not included in keep_tag_ids will be considered as \"removal/mask\" targets (i.e., will be removed later).\n\n"
            "Other requirements:\n"
            "- Each tag's sentence field must be a continuous segment of text from the original (do not concatenate).\n"
            "- Do not output any non-JSON text.\n"
        )
        human_prompt = text
        last_err = None
        for attempt in range(max_retries):
            try:
                llm = _create_llm(model_config, api_config)
                messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
                try:
                    resp = llm.invoke(messages)
                    raw = getattr(resp, "content", "") or str(resp)
                except Exception:
                    resp = llm(messages)
                    raw = getattr(resp, "content", "") or str(resp)
                parsed = _parse_json_from_model_output(raw)
                if parsed is None:
                    last_err = f"Cannot parse JSON (attempt {attempt+1}): raw={(raw or '')[:400]}"
                    time.sleep(0.3 * (attempt + 1))
                    continue
                # Standardize tags: supplement full_tag_id and safe_tag
                tags = parsed.get("tags", []) or []
                standardized_tags = []
                for tag in tags:
                    local_tag_id = str(tag.get("tag_id") or "").strip()
                    if not local_tag_id:
                        local_tag_id = f"tag_{uuid.uuid4().hex[:8]}"
                    full_tag_id = f"{segment_id}_{local_tag_id}"
                    safe_tag = re.sub(r'[^a-zA-Z0-9_]', '_', full_tag_id)
                    new_tag = dict(tag)
                    new_tag["tag_id"] = local_tag_id
                    new_tag["full_tag_id"] = full_tag_id
                    new_tag["safe_tag"] = safe_tag
                    standardized_tags.append(new_tag)
                # Parse reduced_versions (map keep_tag_ids to full_tag_id)
                reduced_versions_raw = parsed.get("reduced_versions", []) or parsed.get("reduced_masks", []) or []
                reduced_versions = []
                for idx, item in enumerate(reduced_versions_raw):
                    if not isinstance(item, dict):
                        continue
                    if "version_id" in item:
                        vid = item.get("version_id")
                        keep_raw = item.get("keep_tag_ids", []) or item.get("keep_ids", []) or []
                        sugg = item.get("suggestions", "") or item.get("advice", "")
                        reason = item.get("reason", "")
                    else:
                        keys = list(item.keys())
                        if keys:
                            vid = keys[0]
                            sub = item.get(vid, {}) or {}
                            keep_raw = sub.get("keep_tag_ids") or sub.get("keep_ids") or sub.get("keep") or sub.get("mask_ids") or []
                            sugg = sub.get("suggestions", "") or sub.get("advice", "")
                            reason = sub.get("reason", "")
                        else:
                            continue
                    mapped_keep = []
                    for kid in (keep_raw or []):
                        if not isinstance(kid, str):
                            continue
                        kid = kid.strip()
                        if kid.startswith(segment_id + "_"):
                            mapped_keep.append(kid)
                        else:
                            match = next((t["full_tag_id"] for t in standardized_tags if t.get("tag_id") == kid or t.get("full_tag_id") == kid), None)
                            if match:
                                mapped_keep.append(match)
                            else:
                                mapped_keep.append(f"{segment_id}_{kid}")
                    reduced_versions.append({
                        "version_id": vid if isinstance(vid, str) else f"reduced_v{idx+1}",
                        "keep_tag_ids": list(dict.fromkeys(mapped_keep)),
                        "suggestions": sugg,
                        "reason": reason
                    })
                # Generate tagged_text (wrap sentence with safe_tag)
                sentences_to_tag = []
                seen = set()
                for tag in standardized_tags:
                    sent = (tag.get("sentence") or "").strip()
                    if sent and sent not in seen:
                        sentences_to_tag.append(sent)
                        seen.add(sent)
                tagged_text = text
                for sent in reversed(sentences_to_tag):
                    tag_candidates = [t.get("safe_tag") for t in standardized_tags if (t.get("sentence") or "").strip() == sent]
                    tag_name = tag_candidates[0] if tag_candidates else ("tag_" + uuid.uuid4().hex[:6])
                    tagged_text = tagged_text.replace(sent, f"<{tag_name}>{sent}</{tag_name}>", 1)
                # Automatically generate auto tag for paragraphs not covered by any tag
                paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
                auto_count = 0
                for para in paragraphs:
                    covered = False
                    for sent in sentences_to_tag:
                        if sent and sent in para:
                            covered = True
                            break
                    if not covered:
                        auto_count += 1
                        auto_local = f"auto_{auto_count}"
                        full_tag_id = f"{segment_id}_{auto_local}"
                        safe_tag = re.sub(r'[^a-zA-Z0-9_]', '_', full_tag_id)
                        auto_tag = {
                            "tag_id": auto_local,
                            "full_tag_id": full_tag_id,
                            "safe_tag": safe_tag,
                            "tag_type": "background_auto",
                            "text": para,
                            "sentence": para,
                            "reason": "auto-generated to preserve untagged paragraph"
                        }
                        standardized_tags.append(auto_tag)
                        tagged_text = tagged_text.replace(para, f"<{safe_tag}>{para}</{safe_tag}>", 1)
                parsed_out = {
                    "segment_id": segment_id,
                    "tags": standardized_tags,
                    "reduced_versions": reduced_versions,
                    "tagged_text": tagged_text,
                    "raw_model_output": raw,
                    "system_prompt": system_prompt,
                    "human_prompt": human_prompt
                }
                return parsed_out
            except Exception as e:
                last_err = str(e)
                time.sleep(0.3 * (attempt + 1))
        return {"segment_id": segment_id, "tags": [], "reduced_versions": [], "tagged_text": "", "raw_model_output": "", "error": last_err}
    return SimpleNamespace(name="analyze_segment", func=analyze_segment)

def create_apply_reduction_mask_tool():
    def apply_reduction_mask(input_data: Dict[str, Any]) -> Dict[str, Any]:
        segments = input_data.get("segments", [])
        mask = set(input_data.get("mask", []))
        reduced_parts = []
        for seg in segments:
            tagged_text = seg.get("tagged_text", seg.get("text", ""))
            seg_id = seg.get("segment_id", "")
            tags_to_remove = set()
            for t in seg.get("tags", []):
                fid = t.get("full_tag_id")
                if fid and fid in mask:
                    st = t.get("safe_tag") or re.sub(r'[^a-zA-Z0-9_]', '_', fid)
                    tags_to_remove.add(st)
            current = tagged_text
            for tag_name in tags_to_remove:
                current = re.sub(rf"<{re.escape(tag_name)}>[\s\S]*?</{re.escape(tag_name)}>", "", current)
            current = re.sub(r'<[^>]+>', '', current)
            cleaned = re.sub(r'\n\s*\n', '\n\n', current).strip()
            reduced_parts.append({"segment_id": seg_id, "text": cleaned})
        combined = "\n\n".join(p["text"] for p in reduced_parts)
        return {"reduced_text": combined, "reduced_segments": reduced_parts}
    return SimpleNamespace(name="apply_reduction_mask", func=apply_reduction_mask)

# Modified create_rewrite_tool: supports iterative modification mode
def create_rewrite_tool(model_config: Dict[str, Any], max_retries: int = 2):
    def _safe_str(x):
        try:
            if x is None:
                return ""
            if isinstance(x, str):
                return x
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)

    def rewrite_text(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Supports iterative modification mode: based on previous_text and evaluation_feedback for partial modifications
        Return fields: rewritten_text, raw_model_output, system_prompt, human_prompt, used_feedback, evaluation_summary, sampling_params
        """
        import json as _json
        import re as _re
        import time as _time

        # External injectable dependencies (use global if defined in global scope)
        _llm_creator = globals().get("_create_llm", _create_llm)
        _SystemMessage = globals().get("SystemMessage", SystemMessage)
        _HumanMessage = globals().get("HumanMessage", HumanMessage)
        _global_safe_str = globals().get("_safe_str", _safe_str)

        # Read parameters from input_data, provide reasonable defaults
        reduced_text = input_data.get("reduced_text", None)
        if reduced_text is None:
            reduced_text = input_data.get("text", "")
        previous_text = input_data.get("previous_text", None)
        evaluation_summary = input_data.get("evaluation_summary", None)
        preserve_tags = input_data.get("preserve_tags", []) or []
        feedback = input_data.get("feedback", None)
        sampling_params = input_data.get("sampling_params", {}) or {}

        # Optional configuration items
        local_max_retries = input_data.get("max_retries", max_retries)
        local_model_config = input_data.get("model_config", model_config or {})
        
        # Get API config for this specific call
        try:
            from config_api import get_api_config
            api_config = get_api_config()
        except Exception:
            api_config = None

        # Determine base text: if previous_text exists, modify based on it; otherwise generate based on reduced_text
        if previous_text and isinstance(previous_text, str) and previous_text.strip():
            base_text = previous_text.strip()
            is_iterative = True
        else:
            base_text = reduced_text.strip() if isinstance(reduced_text, str) else ""
            is_iterative = False

        if not base_text:
            return {"rewritten_text": "", "error": "Base text is empty"}

        # Build prompt
        if is_iterative:
            system_prompt = (
                "You are a professional news editor responsible for making precise modifications to text based on evaluation feedback.\n\n"
                "**Modification Requirements:**\n"
                "1. Only modify the problematic parts pointed out in the evaluation feedback, keep other content unchanged\n"
                "2. For \"missing\" content: Add missing information at appropriate locations\n"
                "3. For \"extra\" content: Delete or correct extra information\n"
                "4. For \"suggested modifications\": Make precise modifications according to suggestions\n"
                "5. Maintain text fluency and coherence\n\n"
                "Please strictly modify based on the previous version, do not rewrite the entire text."
            )
            human_parts = []
            human_parts.append("Previous version:\n" + base_text)

            if evaluation_summary and isinstance(evaluation_summary, dict):
                faith = evaluation_summary.get("faithfulness", {}) or {}
                if faith.get("missing"):
                    human_parts.append("\nMissing content (needs to be added):")
                    for m in faith.get("missing", []):
                        if isinstance(m, dict):
                            human_parts.append(f"- {m.get('text', '')} (location hint: {m.get('where_hint', '')})")
                        else:
                            human_parts.append(f"- {str(m)}")
                if faith.get("extra"):
                    human_parts.append("\nExtra content (needs to be deleted or corrected):")
                    for ex in faith.get("extra", []):
                        if isinstance(ex, dict):
                            human_parts.append(f"- {ex.get('text', '')} (location hint: {ex.get('where_hint', '')})")
                        else:
                            human_parts.append(f"- {str(ex)}")
                if faith.get("suggested_edits"):
                    human_parts.append("\nSuggested edits:")
                    for edit in faith.get("suggested_edits", []):
                        if isinstance(edit, dict):
                            action = edit.get('action', 'Modify')
                            text = edit.get('text', '')
                            hint = edit.get('position_hint', '')
                            human_parts.append(f"- {action}: {text} (position: {hint})")
                        else:
                            human_parts.append(f"- {str(edit)}")

            if feedback:
                if isinstance(feedback, str):
                    human_parts.append(f"\nAdditional feedback: {feedback}")
                else:
                    try:
                        human_parts.append("\nAdditional feedback: " + _json.dumps(feedback, ensure_ascii=False))
                    except Exception:
                        human_parts.append("\nAdditional feedback: " + str(feedback))

            human_parts.append("\nPlease output the modified complete text (only modify problematic parts, keep other content unchanged):")
            human_prompt = "\n".join(human_parts)
        else:
            reduced_text_for_prompt = reduced_text if isinstance(reduced_text, str) else ""
            system_prompt = (
                "You are a professional text editor specializing in paragraph-by-paragraph optimization of text manuscripts. Your core task is to precisely rewrite the single text paragraph I provide, ensuring style transformation and language innovation while strictly maintaining original meaning, facts, and length, so it can be directly replaced back into the original text.\n\n"
                "I will provide the following information in order, please pay close attention:\n\n"
                "[Full Text]: The content of the entire article for your analysis.\n"
                "[Paragraph to Rewrite]: The current paragraph you need to process.\n\n"
                "Core Rewriting Principles (Must Strictly Follow)\n\n"
                "Zero Fact Alteration: Strictly prohibit adding, deleting, or distorting any core facts, data, or character viewpoints.\n\n"
                "Length Consistency: The word count, information density, and length of the rewritten paragraph should be highly similar to the original, ensuring it can be directly \"embedded\" into the original position.\n\n"
                "Logical Cohesion: The beginning and end of the rewritten paragraph must smoothly transition with the preceding and following paragraphs, maintaining the overall fluency of the article.\n\n"
                "Full Text: {reduced_text}\n\n"
                "Paragraph to Rewrite: {base_text}\n\n"
            ).format(reduced_text=reduced_text_for_prompt, base_text=base_text)

            human_parts = []
            human_parts.append("Writing Goal: Rewrite this paragraph based on the provided full text content and this paragraph's content")
            human_parts.append("\nReference Text:\n" + base_text)
            if feedback:
                if isinstance(feedback, str):
                    human_parts.append("\nModification Instructions (Please Strictly Execute):\n" + feedback)
                else:
                    try:
                        human_parts.append("\nModification Instructions (Please Strictly Execute):\n" + _json.dumps(feedback, ensure_ascii=False))
                    except Exception:
                        human_parts.append("\nModification Instructions (Please Strictly Execute):\n" + str(feedback))
            human_parts.append("\nOnly return the rewritten paragraph text (no extra explanations).")
            human_prompt = "\n\n".join(human_parts)

        last_err = None
        for attempt in range(1, local_max_retries + 1):
            try:
                # Merge model configuration with sampling parameters
                local_model_cfg = dict(local_model_config or {})
                for k, v in (sampling_params or {}).items():
                    if v is not None:
                        local_model_cfg[k] = v

                if not _llm_creator:
                    return {"rewritten_text": "", "error": "_create_llm not defined"}

                llm = _llm_creator(local_model_cfg, api_config)
                messages = [_SystemMessage(content=system_prompt), _HumanMessage(content=human_prompt)]

                try:
                    resp = llm.invoke(messages)
                    raw = getattr(resp, "content", "") or str(resp)
                except Exception:
                    resp = llm(messages)
                    raw = getattr(resp, "content", "") or str(resp)

                rewritten = (raw or "").strip()
                rewritten = _re.sub(r'<[^>]+>', '', rewritten).strip()

                if not rewritten:
                    last_err = f"Empty output (attempt {attempt})"
                    _time.sleep(0.2 * attempt)
                    continue

                result = {
                    "rewritten_text": rewritten,
                    "raw_model_output": raw,
                    "system_prompt": system_prompt,
                    "human_prompt": human_prompt,
                    "used_feedback": _global_safe_str(feedback or evaluation_summary),
                    "evaluation_summary": evaluation_summary,
                    "sampling_params": sampling_params,
                    "is_iterative": is_iterative,
                    "preserve_tags": preserve_tags,
                }
                return result

            except Exception as e:
                last_err = str(e)
                _time.sleep(0.2 * attempt)

        return {"rewritten_text": "", "error": last_err}

    return SimpleNamespace(name="rewrite_text", func=rewrite_text)

def create_faithfulness_evaluator(model_config: Dict[str, Any], max_retries: int = 2):
    def evaluate(input_data: Dict[str, Any]) -> Dict[str, Any]:
        reduced = input_data.get("reduced_text", None)
        if reduced is None:
            reduced = input_data.get("original_text", "").strip()
        else:
            reduced = str(reduced).strip()
        rewritten = input_data.get("rewritten_text", "").strip()
        if not reduced:
            return {"passed": False, "score": 0.0, "feedback": "Reference text (reduced) is empty"}
        if not rewritten:
            return {"passed": False, "score": 0.0, "feedback": "Rewritten text is empty"}
        
        # Get API config for this specific call
        try:
            from config_api import get_api_config
            api_config = get_api_config()
        except Exception:
            api_config = None
            
        # Require more detailed structured output: missing/extra/suggested_edits
        prompt = f"""You are a fact-checking expert. Please strictly judge whether the rewritten text is faithful to the given reference text (the reference text is the reduced version).
Reference (Reduced) Text: {reduced}
Rewritten Text: {rewritten}
Please output strict JSON, containing fields:
- passed: true/false
- score: 0.0~1.0
- missing: list, containing objects {{ "text": "...", "where_hint": "..." }} (indicating which facts or key points from the reference are missing in the rewrite)
- extra: list, containing objects {{ "text": "...", "where_hint": "..." }} (indicating facts added in the rewrite that should not be there or are uncertain)
- suggested_edits: list, containing objects {{ "action":"insert|delete|rewrite", "text":"...", "position_hint":"..." }}
- explanation: natural language explanation of the issues
Only return JSON, no additional text or explanation. Example: {{\"passed\":true,\"score\":0.95, ...}}"""
        last_err = None
        for attempt in range(max_retries):
            try:
                llm = _create_llm(model_config, api_config)
                messages = [SystemMessage(content="You are a strict fact-checker. Only output JSON."), HumanMessage(content=prompt)]
                try:
                    resp = llm.invoke(messages)
                    raw = getattr(resp, "content", "") or str(resp)
                except Exception:
                    resp = llm(messages)
                    raw = getattr(resp, "content", "") or str(resp)
                parsed = _parse_json_from_model_output(raw)
                if parsed and "passed" in parsed:
                    parsed["raw_model_output"] = raw
                    parsed["system_prompt"] = "You are a strict fact-checker. Only output JSON."
                    parsed["human_prompt"] = prompt
                    # Compatibility: ensure suggested_edits is in list format
                    if "suggested_edits" not in parsed:
                        parsed.setdefault("suggested_edits", [])
                    return parsed
                last_err = "Cannot parse returned JSON"
            except Exception as e:
                last_err = str(e)
                time.sleep(0.3 * (attempt + 1))
        return {"passed": False, "score": 0.0, "feedback": last_err or "Evaluation failed", "suggested_edits": []}
    return SimpleNamespace(name="faithfulness_eval", func=evaluate)

def create_diversity_evaluator(model_config: Dict[str, Any], max_retries: int = 2):
    def evaluate(input_data: Dict[str, Any]) -> Dict[str, Any]:
        reduced = input_data.get("reduced_text", None)
        if reduced is None:
            reduced = input_data.get("original_text", "").strip()
        else:
            reduced = str(reduced).strip()
        rewritten = input_data.get("rewritten_text", "").strip()
        if not reduced or not rewritten:
            return {"passed": False, "score": 0.0, "feedback": "Reference or rewritten text is empty"}
        
        # Get API config for this specific call
        try:
            from config_api import get_api_config
            api_config = get_api_config()
        except Exception:
            api_config = None
            
        prompt = f"""You are a text diversity evaluation expert. Judge whether the rewritten text is significantly different in expression from the reference (reduced) text.
Reference (Reduced) Text: {reduced}
Rewritten Text: {rewritten}
Please answer (only output JSON):
- Diversity score (0.0~1.0, 0=almost identical, 1=highly different expression but semantically consistent)
- Is sufficiently diverse (true/false)
- Improvement suggestions (try diversity from style, role, dissemination form perspectives)
Output strictly as JSON, e.g.: {{\"passed\":true,\"score\":..,\"feedback\":\"...\"}}"""
        last_err = None
        for attempt in range(max_retries):
            try:
                llm = _create_llm(model_config, api_config)
                messages = [SystemMessage(content="You are a text diversity expert. Only output JSON."), HumanMessage(content=prompt)]
                try:
                    resp = llm.invoke(messages)
                    raw = getattr(resp, "content", "") or str(resp)
                except Exception:
                    resp = llm(messages)
                    raw = getattr(resp, "content", "") or str(resp)
                parsed = _parse_json_from_model_output(raw)
                if parsed and "passed" in parsed:
                    parsed["raw_model_output"] = raw
                    parsed["system_prompt"] = "You are a text diversity expert. Only output JSON."
                    parsed["human_prompt"] = prompt
                    return parsed
                last_err = "Cannot parse returned JSON"
            except Exception as e:
                last_err = str(e)
                time.sleep(0.3 * (attempt + 1))
        return {"passed": False, "score": 0.0, "feedback": last_err or "Evaluation failed"}
    return SimpleNamespace(name="diversity_eval", func=evaluate)

# ---------------- build tools ----------------
def build_tools_for_agent(model_config: Dict[str, Any]) -> Dict[str, Callable]:
    tools = [
        create_segmenter_tool(),
        create_analyze_segment_tool(model_config),
        create_apply_reduction_mask_tool(),
        create_rewrite_tool(model_config),
        create_faithfulness_evaluator(model_config),
        create_diversity_evaluator(model_config),
    ]
    return {t.name: t.func for t in tools}

# ---------------- Graph nodes (stateless functions that operate on state) ----------------
def make_graph_nodes(tools: Dict[str, Callable], max_analyze_workers: int = 4):
    # common helper to record tool calls
    def record_tool_call(state: Dict[str, Any], node: str, tool: str, inp: Dict[str, Any], out: Dict[str, Any], duration: Optional[float] = None):
        state.setdefault("tool_calls", [])
        entry = {
            "ts": time.time(),
            "node": node,
            "tool": tool,
            "input": inp,
            "output": out
        }
        if duration is not None:
            entry["duration"] = duration
        state["tool_calls"].append(entry)
        if DEBUG_MODE:
            logger.debug("tool_call: %s %s duration=%.3f", node, tool, duration or 0.0)
        return state

    def node_segmenter(state: Dict[str, Any]) -> Dict[str, Any]:
        start_ts = time.time()
        state.setdefault("trace", []).append({"node": "segmenter_start", "ts": start_ts})
        txt = state.get("news_text", "")
        t0 = time.time()
        res = tools["segmenter"]({"text": txt})
        t1 = time.time()
        duration = t1 - t0
        state = record_tool_call(state, "segmenter", "segmenter", {"text": txt}, res, duration=duration)
        if "error" in res:
            state.setdefault("errors", []).append(f"segmenter_error:{res['error']}")
            state["segments"] = []
        else:
            state["segments"] = res["segments"]
        end_ts = time.time()
        state.setdefault("trace", []).append({"node": "segmenter_end", "ts": end_ts, "duration": end_ts - start_ts})
        return state

    def node_analyze_segments(state: Dict[str, Any]) -> Dict[str, Any]:
        start_ts = time.time()
        state.setdefault("trace", []).append({"node": "analyze_segments_start", "ts": start_ts})
        segments = state.get("segments", [])
        state.setdefault("tool_calls", [])
        analyzed_results = [None] * len(segments)
        def call_and_record(idx, seg):
            inp = {"segment_id": seg.get("segment_id"), "text": seg.get("text")}
            t0 = time.time()
            out = tools["analyze_segment"](inp)
            t1 = time.time()
            out.setdefault("segment_id", inp["segment_id"])
            out.setdefault("text", inp["text"])
            out.setdefault("tags", out.get("tags", []))
            duration = t1 - t0
            record_tool_call(state, "analyze_segments", "analyze_segment", inp, out, duration=duration)
            return idx, out
        workers = min(max_analyze_workers, max(1, len(segments)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(call_and_record, idx, seg): idx for idx, seg in enumerate(segments)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    _, out = fut.result()
                except Exception as e:
                    out = {"segment_id": segments[idx].get("segment_id"), "text": segments[idx].get("text"), "tags": [], "error": str(e)}
                record_tool_call(state, "analyze_segments", "analyze_segment", {"segment_id": segments[idx].get("segment_id"), "text": segments[idx].get("text")}, out)
                analyzed_results[idx] = out
        state["analyzed_segments"] = analyzed_results
        end_ts = time.time()
        state.setdefault("trace", []).append({"node": "analyze_segments_end", "ts": end_ts, "duration": end_ts - start_ts})
        return state

    def node_generate_masks(state: Dict[str, Any]) -> Dict[str, Any]:
        """ Prioritize using reduced_versions returned by analyze_segment (version suggestions for each segment) to construct global keep_tag_lists (merge keep for each segment by version).
        If analyze does not return reasonable keep lists (all versions empty or abnormal), fallback to simple random mask generation.
        Final output:
        state["masks"] : list of masks (each is a list of full_tag_id to remove)
        state["mask_keep_lists"] : list of keep full_tag_id lists (per version)
        state["mask_suggestions"] : list of dict(s) containing version_id / suggestions / keep_tag_ids
        """
        start_ts = time.time()
        state.setdefault("trace", []).append({"node": "generate_masks_start", "ts": start_ts})
        segs = state.get("analyzed_segments", []) or []
        num_versions = int(state.get("num_versions", 3))
        # gather all full tag ids
        all_full_ids = []
        for seg in segs:
            for t in seg.get("tags", []):
                fid = t.get("full_tag_id")
                if fid:
                    all_full_ids.append(fid)
        # collect per-segment reduced_versions
        per_seg_rvs = []
        version_ids = []
        for seg in segs:
            rvs = seg.get("reduced_versions", []) or []
            per_seg_rvs.append((seg.get("segment_id"), rvs))
            if rvs and not version_ids:
                version_ids = [rv.get("version_id") for rv in rvs if rv.get("version_id")]
        if not version_ids:
            version_ids = [f"reduced_v{i+1}" for i in range(num_versions)]
        mapped_keep_lists = []
        mask_suggestions = []
        # For each version id, aggregate keep_tag_ids and suggestions across segments
        for vid in version_ids:
            keep_union = set()
            suggestions = []
            for seg_id, rvs in per_seg_rvs:
                for rv in rvs:
                    if rv.get("version_id") == vid:
                        keep_list = rv.get("keep_tag_ids", []) or []
                        for k in keep_list:
                            if isinstance(k, str) and k:
                                keep_union.add(k)
                        s = rv.get("suggestions")
                        if s:
                            suggestions.append(s)
            mapped_keep_lists.append(list(keep_union))
            mask_suggestions.append({"version_id": vid, "suggestions": suggestions, "keep_tag_ids": list(keep_union)})
        # If no keep information found at all, fallback to generator
        has_any_keep = any(len(k) > 0 for k in mapped_keep_lists)
        if not has_any_keep:
            masks = []
            for i in range(num_versions):
                keep_count = max(1, len(all_full_ids) // 3) if all_full_ids else 0
                keep_set = set(random.sample(all_full_ids, keep_count)) if all_full_ids else set()
                mask = [fid for fid in all_full_ids if fid not in keep_set]
                masks.append(mask)
            state["masks"] = masks
            state["mask_keep_lists"] = [[] for _ in range(num_versions)]
            state["mask_suggestions"] = []
        else:
            masks = []
            valid_ids = set(all_full_ids)
            for keep in mapped_keep_lists:
                mask = [fid for fid in all_full_ids if fid not in set(keep)]
                mask = [m for m in mask if m in valid_ids]
                masks.append(mask)
            state["masks"] = masks
            state["mask_keep_lists"] = mapped_keep_lists
            state["mask_suggestions"] = mask_suggestions
        end_ts = time.time()
        state.setdefault("trace", []).append({"node": "generate_masks_end", "ts": end_ts, "duration": end_ts - start_ts})
        return state

    def node_apply_masks(state: Dict[str, Any]) -> Dict[str, Any]:
        start_ts = time.time()
        state.setdefault("trace", []).append({"node": "apply_masks_start", "ts": start_ts})
        segs = state.get("analyzed_segments", [])
        masks = state.get("masks", [])
        reduced_versions = []
        for i, mask in enumerate(masks):
            inp = {"segments": segs, "mask": mask}
            t0 = time.time()
            res = tools["apply_reduction_mask"](inp)
            t1 = time.time()
            duration = t1 - t0
            record_tool_call(state, "apply_masks", "apply_reduction_mask", inp, res, duration=duration)
            if "error" in res:
                reduced_versions.append({"version_id": f"reduced_v{i+1}", "error": res["error"]})
            else:
                keep_list = []
                suggestions = {}
                if state.get("mask_keep_lists") and i < len(state.get("mask_keep_lists")):
                    keep_list = state.get("mask_keep_lists")[i]
                if state.get("mask_suggestions") and i < len(state.get("mask_suggestions")):
                    suggestions = state.get("mask_suggestions")[i]
                reduced_versions.append({
                    "version_id": f"reduced_v{i+1}",
                    "text": res.get("reduced_text", ""),
                    "segments": res.get("reduced_segments", []),
                    "mask": mask,
                    "keep_tag_ids": keep_list,
                    "suggestions": suggestions
                })
        state["reduced_versions"] = reduced_versions
        end_ts = time.time()
        state.setdefault("trace", []).append({"node": "apply_masks_end", "ts": end_ts, "duration": end_ts - start_ts})
        return state

    # Modified iterative rewrite node: separately rewrite each segment within each version, then reassemble at version level
    # Evaluation logic: only use faithfulness's passed field as criteria for passing (no longer relies on diversity threshold)
    def node_rewrite_versions(state: Dict[str, Any]) -> Dict[str, Any]:
        start_node_ts = time.time()
        state.setdefault("trace", []).append({"node": "rewrite_versions_start", "ts": start_node_ts})
        reduced = state.get("reduced_versions", []) or []
        per_version_max_attempts = state.get("per_segment_max_attempts", 3)
        rewrites = []
        state.setdefault("tool_calls", [])
        
        for i, rv in enumerate(reduced):
            reduced_version_id = rv.get("version_id")
            segments = rv.get("segments", [])
            if not segments:
                seg_entry = {"segment_id": f"{reduced_version_id}_seg_1", "text": (rv.get("text", "") or "").strip()}
                segments = [seg_entry]
            
            per_segment_results = []
            successful_segments_count = 0  # Count successfully rewritten segments
            
            for seg_idx, seg in enumerate(segments):
                seg_id = seg.get("segment_id", f"seg_{seg_idx+1}")
                reduced_text = (seg.get("text", "") or "").strip()
                
                if not reduced_text:
                    per_segment_results.append({
                        "segment_id": seg_id,
                        "reduced_text": "",
                        "final_rewritten_text": "",
                        "final_passed": False,
                        "used_fallback": True,  # Mark as using fallback
                        "history": []
                    })
                    continue

                current_text = None
                history = []
                final_passed = False
                used_fallback = False  # Mark whether fallback was used

                for attempt in range(1, per_version_max_attempts + 1):
                    # Build rewrite input
                    inp = {
                        "reduced_text": reduced_text,
                        "previous_text": current_text,  # Iterate based on previous result
                        "preserve_tags": [],
                        "sampling_params": rv.get("sampling_params", {})
                    }

                    # If subsequent iteration, add evaluation feedback
                    if attempt > 1 and history:
                        last_iter = history[-1]
                        faith_result_prev = last_iter.get("faithfulness", {}) or {}
                        div_result_prev = last_iter.get("diversity", {}) or {}

                        feedback_parts = []
                        # Only treat faithfulness issues as must-fix items
                        if not faith_result_prev.get("passed", False):
                            faith_feedback = faith_result_prev.get("feedback", "")
                            if faith_feedback:
                                feedback_parts.append(f"Faithfulness issues: {faith_feedback}")
                            if faith_result_prev.get("missing"):
                                feedback_parts.append("Content to add:")
                                for m in faith_result_prev.get("missing", []):
                                    feedback_parts.append(f"- {m.get('text', '')} (position: {m.get('where_hint', '')})")
                            if faith_result_prev.get("extra"):
                                feedback_parts.append("Content to delete or correct:")
                                for ex in faith_result_prev.get("extra", []):
                                    feedback_parts.append(f"- {ex.get('text', '')} (position: {ex.get('where_hint', '')})")
                            if faith_result_prev.get("suggested_edits"):
                                feedback_parts.append("Specific modification suggestions:")
                                for edit in faith_result_prev.get("suggested_edits", []):
                                    action = edit.get('action', 'Modify')
                                    text = edit.get('text', '')
                                    hint = edit.get('position_hint', '')
                                    feedback_parts.append(f"- {action}: {text} (position: {hint})")
                        # Optionally record diversity feedback for reference, but not enforce as passing condition
                        if not div_result_prev.get("passed", False):
                            div_feedback = div_result_prev.get("feedback", "")
                            if div_feedback:
                                feedback_parts.append(f"Diversity suggestions: {div_feedback}")

                        if feedback_parts:
                            inp["feedback"] = "\n".join(feedback_parts)
                        inp["evaluation_summary"] = {
                            "faithfulness": faith_result_prev,
                            "diversity": div_result_prev
                        }

                    # Call rewrite tool
                    t0 = time.time()
                    try:
                        res = tools["rewrite_text"](inp)
                    except Exception as e:
                        res = {"rewritten_text": "", "error": str(e)}
                    t1 = time.time()
                    duration = t1 - t0
                    record_tool_call(state, "rewrite_versions", "rewrite_text", inp, res, duration=duration)

                    rewritten_text = res.get("rewritten_text", "") or ""
                    if not rewritten_text:
                        rewritten_text = current_text or reduced_text
                    current_text = rewritten_text

                    # Evaluate faithfulness
                    faith_inp = {"reduced_text": reduced_text, "rewritten_text": rewritten_text}
                    try:
                        faith_result = tools["faithfulness_eval"](faith_inp)
                    except Exception as e:
                        faith_result = {"passed": False, "score": 0.0, "feedback": f"eval_error:{str(e)}", "suggested_edits": []}

                    # Evaluate diversity
                    div_inp = {"reduced_text": reduced_text, "rewritten_text": rewritten_text}
                    try:
                        diversity_result = tools["diversity_eval"](div_inp)
                    except Exception as e:
                        diversity_result = {"passed": False, "score": 0.0, "feedback": f"eval_error:{str(e)}"}

                    iter_record = {
                        "attempt": attempt,
                        "rewritten_text": rewritten_text,
                        "faithfulness": faith_result,
                        "diversity": diversity_result,
                        "is_iterative": res.get("is_iterative", attempt > 1),
                        "feedback_used": inp.get("feedback", "")
                    }
                    history.append(iter_record)

                    faith_passed = bool(faith_result.get("passed", False))

                    # New logic: consider passed as long as faith_passed
                    if faith_passed:
                        final_passed = True
                        break

                    if attempt >= per_version_max_attempts:
                        # Reached maximum attempts without passing, use fallback strategy
                        break

                # ========== MODIFIED: Fallback handling logic ==========
                if not final_passed:
                    # Use original reduced text as fallback solution
                    current_text = reduced_text
                    used_fallback = True
                    # 降级文本直接通过事实度评估，不进行评估
                    final_passed = True
                    
                    # Record fallback operation
                    history.append({
                        "attempt": "fallback",
                        "rewritten_text": reduced_text,
                        "faithfulness": {"passed": True, "score": 1.0, "feedback": "Fallback: using original reduced text"},
                        "diversity": {"passed": False, "score": 0.0, "feedback": "Using original text, no diversity"},
                        "is_iterative": False,
                        "feedback_used": "Fallback: using original reduced text"
                    })
                else:
                    successful_segments_count += 1  # Count successful segments

                per_seg_final_rewritten = current_text or (history[-1]["rewritten_text"] if history else reduced_text)

                per_segment_results.append({
                    "segment_id": seg_id,
                    "reduced_text": reduced_text,
                    "final_rewritten_text": per_seg_final_rewritten,
                    "final_passed": final_passed,
                    "used_fallback": used_fallback,  # Mark whether fallback was used
                    "history": history
                })

            # Version-level aggregation: calculate success rate
            combined_texts = []
            seg_pass_flags = []
            fallback_segments_count = 0
            
            for seg_res in per_segment_results:
                combined_texts.append(seg_res.get("final_rewritten_text", ""))
                seg_pass_flags.append(bool(seg_res.get("final_passed", False)))
                if seg_res.get("used_fallback", False):
                    fallback_segments_count += 1
            
            final_rewritten = "\n\n".join(t for t in combined_texts if t is not None)
            version_passed = all(seg_pass_flags) if per_segment_results else False
            
            # ========== MODIFIED: Success rate statistics ==========
            total_segments = len(per_segment_results)
            success_ratio = successful_segments_count / total_segments if total_segments > 0 else 0.0
            fallback_ratio = fallback_segments_count / total_segments if total_segments > 0 else 0.0
            
            # Version quality label
            if success_ratio == 1.0:
                quality_label = "Fully Successful"
            elif success_ratio >= 0.7:
                quality_label = "Highly Successful"
            elif success_ratio >= 0.3:
                quality_label = "Partially Successful" 
            else:
                quality_label = "Mainly Fallback"

            # Use success_ratio as the version score
            version_score = success_ratio

            rewrites.append({
                "version_id": f"rewritten_v{i+1}",
                "reduced_version_id": reduced_version_id,
                "reduced_text": rv.get("text", ""),
                "segments": per_segment_results,
                "final_rewritten_text": final_rewritten,
                "version_score": version_score,  # Use success_ratio as score
                "final_passed": version_passed,
                "success_ratio": success_ratio,  # Successful rewrite ratio
                "fallback_ratio": fallback_ratio,  # Fallback usage ratio
                "quality_label": quality_label,  # Quality label
                "successful_segments_count": successful_segments_count,  # Successful segment count
                "total_segments_count": total_segments,  # Total segment count
                "history": per_segment_results
            })

        # MODIFIED: Use success_ratio for scoring
        scores = []
        all_passed = True
        best = {"version_id": None, "score": 0.0}
        
        for r in rewrites:
            score = float(r.get("version_score", 0.0) or 0.0)  # Use version_score instead of final_score
            passed = bool(r.get("final_passed", False))
            success_ratio = r.get("success_ratio", 0.0)
            
            scores.append({
                "version_id": r.get("version_id"), 
                "score": score,  # This is now success_ratio
                "passed": passed,
                "success_ratio": success_ratio,
                "quality_label": r.get("quality_label", "Unknown")
            })
            
            if score > best.get("score", 0.0):
                best = {
                    "version_id": r.get("version_id"), 
                    "score": score,
                    "success_ratio": success_ratio,
                    "quality_label": r.get("quality_label", "Unknown")
                }
            if not passed:
                all_passed = False

        state["rewritten_versions"] = rewrites
        state["scores"] = scores
        state["all_versions_passed"] = all_passed
        state["best_score"] = best.get("score", 0.0)
        state["best_version"] = best

        # Generate summary including success rate statistics
        try:
            input_text = state.get("original_text", state.get("news_text", ""))
            top_sorted = sorted(rewrites, key=lambda x: float(x.get("version_score", 0.0) or 0.0), reverse=True)
            top_three = []
            for item in top_sorted[:3]:
                top_three.append({
                    "version_id": item.get("version_id"),
                    "version_score": item.get("version_score"),  # Use version_score
                    "success_ratio": item.get("success_ratio"),
                    "quality_label": item.get("quality_label"),
                    "final_rewritten_text": item.get("final_rewritten_text")
                })
            state["summary"] = {
                "input_text": input_text, 
                "top_rewrites": top_three,
                "overall_stats": {
                    "total_versions": len(rewrites),
                    "fully_successful_versions": len([r for r in rewrites if r.get("success_ratio", 0) == 1.0]),
                    "average_success_ratio": sum(r.get("success_ratio", 0) for r in rewrites) / len(rewrites) if rewrites else 0
                }
            }
        except Exception as e:
            state["summary"] = {
                "input_text": state.get("original_text", ""), 
                "top_rewrites": [],
                "error": str(e)
            }

        end_node_ts = time.time()
        state.setdefault("trace", []).append({"node": "rewrite_versions_end", "ts": end_node_ts, "duration": end_node_ts - start_node_ts})
        return state

    def node_evaluate_rewrites(state: Dict[str, Any]) -> Dict[str, Any]:
        start_node_ts = time.time()
        state.setdefault("trace", []).append({"node": "evaluate_rewrites_start", "ts": start_node_ts})
        rewritten = state.get("rewritten_versions", []) or []
        results = []
        state.setdefault("tool_calls", [])
        for rw in rewritten:
            version_id = rw.get("version_id")
            # Use version_score (success_ratio) as the score
            version_score = float(rw.get("version_score", 0.0) or 0.0)
            passed = bool(rw.get("final_passed", False))
            results.append({"version_id": version_id, "score": version_score, "passed": passed})
        state["scores"] = results
        # New passing condition: all versions pass (based only on faithfulness)
        all_passed = len(results) > 0 and all(r.get("passed", False) for r in results)
        state["all_versions_passed"] = all_passed
        best = max(results, key=lambda x: x["score"], default={"version_id": None, "score": 0.0})
        state["best_score"] = best["score"]
        state["best_version"] = best
        end_node_ts = time.time()
        state.setdefault("trace", []).append({"node": "evaluate_rewrites_end", "ts": end_node_ts, "duration": end_node_ts - start_node_ts})
        return state

    def node_increment_attempts(state: Dict[str, Any]) -> Dict[str, Any]:
        state["rewrite_attempts"] = state.get("rewrite_attempts", 0) + 1
        state.setdefault("trace", []).append({
            "node": "inc_attempt",
            "ts": time.time(),
            "attempt": state["rewrite_attempts"]
        })
        return state

    def node_fallback(state: Dict[str, Any]) -> Dict[str, Any]:
        segs = state.get("segments", [])
        if not segs:
            state.setdefault("errors", []).append("fallback_no_segments")
            return state
        keep_count = max(1, int(len(segs) * 0.7))
        kept = segs[:keep_count]
        reduced_text = "\n\n".join(s["text"] for s in kept)
        inp = {"text": reduced_text}
        t0 = time.time()
        res = tools["rewrite_text"](inp)
        t1 = time.time()
        duration = t1 - t0
        record_tool_call(state, "fallback", "rewrite_text", inp, res, duration=duration)
        fallback = {"reduced_text": reduced_text}
        if "error" in res:
            fallback["rewritten_text"] = ""
            fallback["error"] = res["error"]
        else:
            fallback["rewritten_text"] = res.get("rewritten_text", "")
        state["fallback_result"] = fallback
        return state

    def node_end_success(state: Dict[str, Any]) -> Dict[str, Any]:
        state["terminated_at"] = "end_success"
        state["final_status"] = "success"
        return state

    def node_end_fail(state: Dict[str, Any]) -> Dict[str, Any]:
        state["terminated_at"] = "end_fail"
        state["final_status"] = "failed"
        return state

    return {
        "segmenter": node_segmenter,
        "analyze_segments": node_analyze_segments,
        "generate_masks": node_generate_masks,
        "apply_masks": node_apply_masks,
        "rewrite_versions": node_rewrite_versions,
        "evaluate_rewrites": node_evaluate_rewrites,
        "inc_attempts": node_increment_attempts,
        "fallback": node_fallback,
        "end_success": node_end_success,
        "end_fail": node_end_fail,
    }

# ---------------- Conditions ----------------
def cond_quality_ok(state: Dict[str, Any]) -> bool:
    return state.get("all_versions_passed", False)

def cond_need_retry(state: Dict[str, Any]) -> bool:
    attempts = state.get("rewrite_attempts", 0)
    max_attempts = state.get("max_attempts", 2)
    return attempts < max_attempts and not state.get("all_versions_passed", False)

def cond_give_up(state: Dict[str, Any]) -> bool:
    attempts = state.get("rewrite_attempts", 0)
    max_attempts = state.get("max_attempts", 2)
    return attempts >= max_attempts and not state.get("all_versions_passed", False)

# ---------------- Build & Run Graph (Langgraph) ----------------
def build_and_run_graph_langgraph(news_text: str, model_config: Dict[str, Any]):
    if not LANGGRAPH_AVAILABLE or StateGraph is None:
        raise RuntimeError("langgraph not installed or unavailable. Please install langgraph and ensure import successful.")
    tools = build_tools_for_agent(model_config)
    nodes = make_graph_nodes(tools)
    class StateSchema(TypedDict, total=False):
        news_text: str
        original_text: str
        num_versions: int
        rewrite_attempts: int
        max_attempts: int
        quality_threshold: float
        model_config: Dict[str, Any]
        segments: List[Dict[str, Any]]
        analyzed_segments: List[Dict[str, Any]]
        masks: List[Any]
        reduced_versions: List[Dict[str, Any]]
        rewritten_versions: List[Dict[str, Any]]
        scores: List[Dict[str, Any]]
        best_score: float
        best_version: Dict[str, Any]
        trace: List[Dict[str, Any]]
        tool_calls: List[Dict[str, Any]]
        errors: List[str]
        terminated_at: str
        final_status: str
        summary: Dict[str, Any]
    g = StateGraph(StateSchema)
    # add nodes
    g.add_node("segment", nodes["segmenter"])
    g.add_node("analyze", nodes["analyze_segments"])
    g.add_node("gen_masks", nodes["generate_masks"])
    g.add_node("apply_masks", nodes["apply_masks"])
    g.add_node("rewrite", nodes["rewrite_versions"])
    g.add_node("evaluate", nodes["evaluate_rewrites"])
    g.add_node("inc_attempt", nodes["inc_attempts"])
    g.add_node("fallback", nodes["fallback"])
    g.add_node("end_success", nodes["end_success"])
    g.add_node("end_fail", nodes["end_fail"])
    # entry and edges
    g.set_entry_point("segment")
    g.add_edge("segment", "analyze")
    g.add_edge("analyze", "gen_masks")
    g.add_edge("gen_masks", "apply_masks")
    g.add_edge("apply_masks", "rewrite")
    g.add_edge("rewrite", "evaluate")
    g.add_edge("inc_attempt", "rewrite")
    g.add_edge("fallback", "end_fail")
    # conditional routing after evaluate
    def route_after_evaluate(state):
        # Prioritize success
        if cond_quality_ok(state):
            return "end_success"
        # If retry needed (not meeting standards and still have attempts), go inc_attempt -> rewrite
        if cond_need_retry(state):
            return "inc_attempt"
        # If reaching give up condition, go fallback (further degradation)
        if cond_give_up(state):
            return "fallback"
        # Default to fallback
        return "fallback"
    g.add_conditional_edges("evaluate", route_after_evaluate)
    initial_state: Dict[str, Any] = {
        "news_text": news_text,
        "original_text": news_text,
        "num_versions": 3,
        "rewrite_attempts": 0,
        "max_attempts": 0,
        "model_config": model_config or {},
        "trace": [],
        "tool_calls": [],
        "errors": [],
        "random_seed": 42,
    }
    compiled = g.compile()
    final_state = compiled.invoke(initial_state)
    return final_state

# ---------------- main example ----------------
if __name__ == "__main__":
    import pathlib
    input_file = "/data1/zhc/LLM/a.test/muti_domain_news.json"
    output_dir = pathlib.Path("./muti_outputs")
    output_dir.mkdir(exist_ok=True)
    model_config = get_model_deployment_config(DEFAULT_TEACHER_MODEL)
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError("langgraph not installed. Please install langgraph and retry.")
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            news_list = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read input file: {e}")
    
    # 添加起始索引设置
    start_index = 1  # 修改这个值来设置起始编号
    results_summary = []
    
    for idx, item in enumerate(news_list):
        current_index = idx + start_index  # 计算当前文件的编号
        print(f"\n=== Processing news {current_index}/{len(news_list) + start_index - 1} ===")
        content = item.get("content", "").strip()
        if not content:
            print(" Skip: content empty")
            continue
        result = build_and_run_graph_langgraph(content, model_config=model_config)
        
        # 修改输出文件名使用 current_index
        output_path = output_dir / f"news_{current_index:03d}_result.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        summary = result.get("summary", {"input_text": content, "top_rewrites": []})
        summary_path = output_dir / f"news_{current_index:03d}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"summary saved to: {summary_path}")
        
        trace_dir = pathlib.Path("./rewrite_trace_outputs")
        trace_dir.mkdir(exist_ok=True)
        trace_path = trace_dir / f"trace_{int(time.time())}.json"
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(result.get("tool_calls", []), f, ensure_ascii=False, indent=2)
        print(f"tool_calls saved to: {trace_path}")
        
        out_dir = pathlib.Path(".") / "rewrite_trace_outputs"
        out_dir.mkdir(exist_ok=True)
        trace_path2 = out_dir / f"trace_{int(time.time())}.json"
        with open(trace_path2, "w", encoding="utf-8") as f:
            json.dump(result.get("trace", []), f, ensure_ascii=False, indent=2)
        print(f"\ntrace saved to: {trace_path2}")