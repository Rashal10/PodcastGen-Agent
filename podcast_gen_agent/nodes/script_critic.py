"""LLM-as-critic rewrite loop for podcast scripts."""

import logging
import re

from ..config import settings
from ..state import PodcastState, coerce_script
from ..utils.decorators import node_handler, with_retries
from ..utils.llm import generate_chat, unload_llm
from ..utils.scripting import estimate_speech_seconds, script_word_count

logger = logging.getLogger(__name__)

SCORE_RE = re.compile(r"SCORE:\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
NOTES_RE = re.compile(r"NOTES:\s*(.*)", re.IGNORECASE | re.DOTALL)


def format_script_for_critic(state: PodcastState) -> str:
    lines = []
    for line in coerce_script(state.get("script") or []):
        lines.append(f"[{line.speaker.title()}]: {line.text}")
    return "\n".join(lines)


def parse_critic_response(raw_text: str) -> tuple[float, str]:
    """Extract SCORE and NOTES from critic output."""
    score_match = SCORE_RE.search(raw_text)
    notes_match = NOTES_RE.search(raw_text)
    score = float(score_match.group(1)) if score_match else 5.0
    score = min(max(score, 1.0), 10.0)
    notes = notes_match.group(1).strip() if notes_match else raw_text.strip()
    return score, notes


def should_rewrite_script(state: PodcastState) -> str:
    """Router after critic: rewrite, continue to voice, or fail."""
    if state.get("error"):
        return "fail"
    score = state.get("critic_score")
    attempts = int(state.get("critic_attempts") or 0)
    if score is not None and score >= settings.critic_min_score:
        return "voice"
    if attempts >= settings.critic_max_retries:
        return "voice"
    return "rewrite"


@with_retries()
def _generate_critique(prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict podcast editor. Score the draft from 1 to 10. "
                "Output exactly two fields:\nSCORE: <number>\nNOTES: <rewrite guidance>"
            ),
        },
        {"role": "user", "content": prompt},
    ]
    return generate_chat(messages, max_new_tokens=400)


@node_handler("critic")
def script_critic_node(state: PodcastState) -> dict:
    """Score the script and request a rewrite when it is below the quality bar."""
    script_text = format_script_for_critic(state)
    duration = state["duration_mins"]
    estimated = estimate_speech_seconds(state.get("script") or [])
    target_secs = duration * 60
    facts = list(state.get("facts_to_hit") or [])
    fact_block = "\n".join(f"- {fact}" for fact in facts) or "- (no structured facts)"

    prompt = f"""Topic: {state["topic"]}
Requested length: {duration} minutes ({target_secs} seconds)
Estimated spoken length: {estimated:.1f} seconds
Word count: {script_word_count(state.get("script") or [])}

Facts the script should use:
{fact_block}

Draft:
{script_text}

Score 1-10 on: coherence, host/guest balance, fact use, concrete language, and length.
If the draft is generic filler, score 5 or lower.
NOTES must tell the writer what to change. Do not rewrite the full script here.
"""

    attempts = int(state.get("critic_attempts") or 0) + 1
    score = 5.0
    notes = "Critic could not parse a score. Tighten the draft and use the research facts."
    try:
        raw = _generate_critique(prompt)
        score, notes = parse_critic_response(raw)
        logger.info("Critic score %.1f on attempt %d", score, attempts)
    except Exception:
        logger.exception("Script critic failed; keeping a midpoint score")

    passing = score >= settings.critic_min_score
    exhausted = attempts >= settings.critic_max_retries
    result: dict = {
        "critic_score": score,
        "critic_notes": notes,
        "critic_attempts": attempts,
    }

    if passing or exhausted:
        unload_llm()
        if not passing:
            logger.warning(
                "Proceeding with critic score %.1f after %d attempt(s)",
                score,
                attempts,
            )
            result["quality_notes"] = ["critic:low_score"]
            result["quality"] = "degraded"
        else:
            logger.info("Script passed critic with score %.1f", score)
    else:
        logger.info("Routing script back for rewrite (score %.1f)", score)

    return result
