from __future__ import annotations

import re

from graphrag_benchmark.config import Settings, get_settings
from graphrag_benchmark.models import AccuracyResult
from graphrag_benchmark.torch_device import bert_score_device


def evaluate_accuracy(
    question: str,
    reference_answer: str,
    candidate_answer: str,
    *,
    settings: Settings | None = None,
    run_bert_score: bool = True,
    run_judge: bool = True,
) -> AccuracyResult:
    s = settings or get_settings()
    judge_raw = ""
    judge_pass: bool | None = None
    err_parts: list[str] = []

    if run_judge:
        try:
            judge_raw, judge_pass = _llm_judge(
                question, reference_answer, candidate_answer, settings=s
            )
        except Exception as e:
            err_parts.append(f"judge: {e}")
            judge_pass = None

    bf1 = bp = br = None
    if run_bert_score:
        try:
            from bert_score import score as bert_score_fn
        except ImportError:
            err_parts.append("bert_score: package not installed")
        else:
            try:
                p, r, f1 = bert_score_fn(
                    [candidate_answer],
                    [reference_answer],
                    lang="en",
                    verbose=False,
                    device=bert_score_device(s.embed_device),
                )
                bp = float(p[0].item())
                br = float(r[0].item())
                bf1 = float(f1[0].item())
            except Exception as e:
                err_parts.append(f"bert_score: {e}")

    return AccuracyResult(
        judge_pass=judge_pass,
        judge_raw=judge_raw,
        bertscore_f1=bf1,
        bertscore_precision=bp,
        bertscore_recall=br,
        error="; ".join(err_parts) if err_parts else None,
    )


def _llm_judge(
    question: str,
    reference: str,
    candidate: str,
    *,
    settings: Settings,
) -> tuple[str, bool | None]:
    prompt = (
        "You grade answers. Reply with exactly one word: PASS or FAIL.\n"
        "PASS if the candidate is factually consistent with the reference for the question.\n"
        "FAIL if it contradicts the reference or misses key facts.\n\n"
        f"Question: {question}\n"
        f"Reference: {reference}\n"
        f"Candidate: {candidate}\n\n"
        "Verdict:"
    )

    if settings.hf_token:
        from huggingface_hub import InferenceClient

        client = InferenceClient(model=settings.hf_judge_model, token=settings.hf_token)
        out_text = ""
        try:
            msg = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16,
                temperature=0.0,
            )
            out_text = (msg.choices[0].message.content or "").strip()
        except Exception:
            gen = client.text_generation(prompt, max_new_tokens=16, temperature=0.0)
            out_text = (gen or "").strip()

        m = re.search(r"\b(PASS|FAIL)\b", out_text.upper())
        if not m:
            return out_text, None
        return out_text, m.group(1) == "PASS"

    if settings.openai_api_key:
        from graphrag_benchmark.llm_client import chat_completion

        reply, _, _, _ = chat_completion(
            [{"role": "user", "content": prompt}],
            settings=settings,
            temperature=0.0,
        )
        m = re.search(r"\b(PASS|FAIL)\b", reply.upper())
        if not m:
            return reply, None
        return reply, m.group(1) == "PASS"

    raise ValueError("Set HUGGINGFACEHUB_API_TOKEN or OPENAI_API_KEY for LLM-as-judge")
