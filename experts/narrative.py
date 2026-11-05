"""Arabic narrative for an expert report.

Two paths, both keeping NUMBERS under program control:

  - deterministic (default, and the only path without an API key): a
    rule-based Arabic summary built from the computed facts. Clearly labelled;
    it is NOT a claim of real language-model reasoning.
  - llm: the real model (via the existing ``agent_community.llm`` interface)
    is asked to WORD the already-computed evidence in Arabic. Its output is
    validated — any number it prints that is not one of the computed/allowed
    numbers (or a small count) means it invented a level, so the output is
    rejected and the deterministic text is used instead.

The model may never change a calculated value or add evidence; it only phrases.
"""

from __future__ import annotations

import re
from typing import Any

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_DIR_AR = {"long": "صاعد", "short": "هابط", "neutral": "محايد"}
_STATE_AR = {"trend_up": "اتجاه صاعد", "trend_down": "اتجاه هابط", "range": "تذبذب عرضي",
             "volatile": "تذبذب عالٍ", "unclear": "غير واضح"}


def _deterministic(expert, comp, report) -> str:
    dir_ar = _DIR_AR.get(report.direction.value, report.direction.value)
    state_ar = _STATE_AR.get(report.market_state.value, report.market_state.value)
    parts = [f"يرى خبير {expert.specialty} أن السوق في حالة {state_ar} مع انحياز {dir_ar}."]
    for pt in comp.analysis_points[:8]:
        parts.append(f"• {pt}")
    if comp.proposal is not None:
        pr = comp.proposal
        parts.append(f"الفرصة وفق المنهج: دخول بين {pr.entry_low} و{pr.entry_high}، "
                     f"وقف عند {pr.stop}، أهداف {pr.targets}؛ شرط التفعيل: {pr.activation_condition}.")
    elif report.wait_reason:
        parts.append(f"القرار: انتظار — {report.wait_reason}.")
    if comp.missing:
        parts.append(f"بيانات/سياق ناقص: {'، '.join(comp.missing)}.")
    return " ".join(parts)


def _allowed_number_strings(comp, report) -> set[str]:
    nums: list[float] = list(comp.allowed_numbers)
    for z in comp.zones:
        nums += [z.low, z.high] + ([z.invalidation] if z.invalidation is not None else [])
    for lv in comp.levels:
        nums.append(lv.value)
    if comp.proposal is not None:
        pr = comp.proposal
        nums += [pr.entry_low, pr.entry_high, pr.stop, *pr.targets]
    out: set[str] = set()
    for n in nums:
        out.add(f"{n}")
        out.add(f"{round(float(n), 2)}")
        out.add(f"{round(float(n), 1)}")
        out.add(f"{int(round(float(n)))}")
    return out


def _llm_output_is_clean(text: str, allowed: set[str]) -> bool:
    for tok in _NUM_RE.findall(text):
        if tok in allowed:
            continue
        # permit small standalone integers used as counts/periods (0..300)
        if "." not in tok and 0 <= int(tok) <= 300:
            continue
        return False
    return True


def narrate(expert, comp, report, run_mode: str, llm_backend) -> tuple[str, str]:
    """Return (arabic_text, used_mode)."""
    deterministic_text = _deterministic(expert, comp, report)
    if run_mode != "llm":
        return deterministic_text, "deterministic"
    # llm requested
    if llm_backend is None or getattr(llm_backend, "is_mock", True):
        # no real backend available -> fall back, but say so plainly
        return deterministic_text, "deterministic (llm requested, no key)"
    allowed = _allowed_number_strings(comp, report)
    system = (f"أنت خبير {expert.specialty} في تحليل الذهب. اكتب فقرة عربية موجزة تشرح الأدلة "
              "المعطاة فقط. ممنوع منعًا باتًا اختراع أي رقم أو مستوى سعري غير وارد في الأدلة، "
              "وممنوع تغيير أي قيمة محسوبة. لا تضف توصية جديدة من عندك.")
    user = ("الأدلة المحسوبة:\n- " + "\n- ".join(comp.analysis_points) +
            f"\nالاتجاه: {report.direction.value}; حالة السوق: {report.market_state.value}; "
            f"التوصية: {report.recommendation.value if report.recommendation else 'انتظار'}.")
    try:
        resp = llm_backend.complete(system, user)
        text = (resp.text or "").strip()
    except Exception as exc:  # pragma: no cover - needs real backend
        return deterministic_text, f"deterministic (llm error: {type(exc).__name__})"
    if not text or not _llm_output_is_clean(text, allowed):
        # the model introduced a number/level not in the evidence -> reject it
        return deterministic_text, "deterministic (llm output rejected: unverified numbers)"
    return text, f"llm:{getattr(llm_backend, 'model', 'real')}"
