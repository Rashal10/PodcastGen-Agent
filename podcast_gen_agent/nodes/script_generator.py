import logging
import re

from ..config import settings
from ..state import DialogueLine, PodcastState, coerce_script
from ..utils.decorators import node_handler, with_retries
from ..utils.llm import generate_chat
from ..utils.scripting import (
    compute_target_lines,
    compute_target_words,
    estimate_speech_seconds,
    refine_script,
    script_word_count,
)
from ..utils.text import clean_text_for_tts

logger = logging.getLogger(__name__)

LINE_PATTERN = re.compile(
    r"^\s*[\[\*]*(host|guest)[\]\*]*\s*:\s*(.+)$",
    re.IGNORECASE,
)


def compute_max_new_tokens(duration_mins: int) -> int:
    """Estimate token budget from target episode duration."""
    words = duration_mins * settings.speech_words_per_minute
    tokens = int(words * 1.5)
    return min(max(tokens, 512), 8192)


def _parse_script(raw_text: str) -> list[DialogueLine]:
    """Extract dialogue lines from LLM output."""
    lines: list[DialogueLine] = []
    for line in raw_text.splitlines():
        match = LINE_PATTERN.match(line.strip())
        if not match:
            continue
        speaker, text = match.groups()
        cleaned = clean_text_for_tts(text.strip().strip('"').strip("'"))
        if cleaned:
            lines.append(DialogueLine(speaker=speaker.lower(), text=cleaned))
    return refine_script(lines)


def _fallback_script(topic: str, research: str = "") -> list[DialogueLine]:
    context = " ".join(research.split())
    if len(context) > 500:
        context = context[:500].rsplit(" ", 1)[0]
    lines = [
        DialogueLine("host", f"Welcome to our podcast about {topic}."),
        DialogueLine("guest", f"Thanks for having me. {topic} is a fascinating subject."),
        DialogueLine("host", "Let us begin with the essential context."),
    ]
    if context:
        lines.append(DialogueLine("guest", context))
    lines.extend(
        [
            DialogueLine(
                "host",
                f"What should listeners remember most when they think about {topic}?",
            ),
            DialogueLine(
                "guest",
                "The key is to understand both the practical value and the limitations. "
                "Good results require clear goals, careful evaluation, and responsible use.",
            ),
            DialogueLine(
                "host",
                "That is a useful perspective. Thank you for joining us today.",
            ),
            DialogueLine(
                "guest",
                "Thank you for having me, and thank you to everyone listening.",
            ),
        ]
    )
    return refine_script(lines)


def _script_prompt(
    topic: str,
    duration: int,
    research: str,
    facts: list[str],
    target_lines: int,
    target_words: int,
    critic_notes: str,
) -> str:
    fact_block = "\n".join(f"- {fact}" for fact in facts) or "- Use the research notes below."
    feedback = ""
    if critic_notes.strip():
        feedback = (
            "\nRewrite feedback from the editor:\n"
            f"{critic_notes.strip()}\n"
            "Fix every issue. Do not repeat generic filler.\n"
        )
    return f"""Write a podcast script for a {duration}-minute episode about: {topic}

Target about {target_lines} dialogue lines and about {target_words} spoken words.
Estimated speaking pace is {settings.speech_words_per_minute} words per minute.

You MUST use these facts in the conversation (paraphrase, do not invent citations):
{fact_block}

Background research:
{research}
{feedback}
Structure:
- Cold open: host and guest jump into a concrete hook in the first two lines.
- Do not use generic filler such as calling the topic fascinating.
- Two or three named segments with a clear question or disagreement in each.
- Guest should challenge or add a limitation, not only agree.
- Close with one specific takeaway listeners can repeat.

Format rules:
- Two speakers: [Host] and [Guest]
- Alternate speakers. Each line is 1 to 3 natural sentences.
- About {target_lines} lines. Output only lines starting with [Host]: or [Guest]:

Example format:
[Host]: We are starting with the part of {topic} that most explainers skip.
[Guest]: That is the right place to start, because the popular version leaves out a hard constraint.

Write the full script:"""


@with_retries()
def _generate_script_text(prompt: str, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a podcast scriptwriter. Output only dialogue lines prefixed "
                "with [Host]: or [Guest]: on separate lines. Ground every major claim "
                "in the provided facts. Do not write filler praise."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    return generate_chat(messages, max_new_tokens)


def _maybe_extend_script(
    script: list[DialogueLine],
    prompt: str,
    duration: int,
    target_words: int,
    max_new_tokens: int,
) -> list[DialogueLine]:
    if script_word_count(script) >= int(target_words * 0.55):
        return script

    logger.info(
        "Script is short (%d words vs %d target); requesting additional lines",
        script_word_count(script),
        target_words,
    )
    continue_prompt = (
        f"{prompt}\n\nThe draft so far is too short for a {duration}-minute episode. "
        "Continue from the last line with more [Host]: and [Guest]: lines. "
        "Do not repeat earlier lines. Stay on the remaining segments and a closer."
    )
    extra = _parse_script(_generate_script_text(continue_prompt, max_new_tokens))
    return refine_script(script + extra)


@node_handler("script")
def script_generator_node(state: PodcastState) -> dict:
    """Generate podcast script using LLM."""
    duration = state["duration_mins"]
    research = state["research_data"]
    topic = state["topic"]
    facts = list(state.get("facts_to_hit") or [])
    target_lines = compute_target_lines(duration)
    target_words = compute_target_words(duration)
    max_new_tokens = compute_max_new_tokens(duration)
    critic_notes = str(state.get("critic_notes") or "")

    prompt = _script_prompt(
        topic,
        duration,
        research,
        facts,
        target_lines,
        target_words,
        critic_notes,
    )

    script: list[DialogueLine] = []
    used_fallback = False
    try:
        logger.info(
            "Generating script (~%d lines, ~%d words, max %d tokens)",
            target_lines,
            target_words,
            max_new_tokens,
        )
        response = _generate_script_text(prompt, max_new_tokens)
        script = _parse_script(response)

        if not script:
            logger.warning("Script parse failed, retrying with stricter format")
            retry_prompt = (
                f"{prompt}\n\nYour previous output was invalid. "
                "Return only [Host]: and [Guest]: lines, one per line."
            )
            response = _generate_script_text(retry_prompt, max_new_tokens)
            script = _parse_script(response)

        if script:
            script = _maybe_extend_script(
                script,
                prompt,
                duration,
                target_words,
                max_new_tokens,
            )
    except Exception:
        logger.exception("LLM script generation failed; using fallback script")

    if not script:
        previous = coerce_script(state.get("script") or [])
        if previous:
            logger.warning("Keeping previous script after a failed rewrite")
            script = previous
        else:
            script = _fallback_script(topic, research)
            used_fallback = True

    estimated = estimate_speech_seconds(script)
    logger.info(
        "Prepared %d dialogue lines (~%.1fs spoken vs %d min requested)",
        len(script),
        estimated,
        duration,
    )

    result: dict = {
        "script": script,
        "current_line_idx": 0,
        "estimated_duration_secs": estimated,
    }
    if used_fallback:
        result["quality_notes"] = ["script:fallback"]
        result["quality"] = "degraded"
    return result
