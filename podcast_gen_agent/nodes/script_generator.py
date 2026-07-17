import logging
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..config import settings
from ..state import DialogueLine, PodcastState
from ..utils.decorators import node_handler, with_retries
from ..utils.gpu import clear_gpu_cache, require_gpu_memory
from ..utils.text import clean_text_for_tts

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None

LINE_PATTERN = re.compile(
    r"^\s*[\[\*]*(host|guest)[\]\*]*\s*:\s*(.+)$",
    re.IGNORECASE,
)


def compute_max_new_tokens(duration_mins: int) -> int:
    """Estimate token budget from target episode duration."""
    words = duration_mins * 150
    tokens = int(words * 1.5)
    return min(max(tokens, 512), 8192)


def compute_target_lines(duration_mins: int) -> int:
    """Estimate dialogue line count from target episode duration."""
    return max(duration_mins * 15, 10)


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    logger.info("Loading LLM: %s", settings.llm_model)
    require_gpu_memory()

    quant_config = BitsAndBytesConfig(**settings.quantization_config)
    _tokenizer = AutoTokenizer.from_pretrained(settings.llm_model)
    _model = AutoModelForCausalLM.from_pretrained(
        settings.llm_model,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return _model, _tokenizer


def unload_model() -> None:
    """Release the LLM from memory."""
    global _model, _tokenizer
    _model = None
    _tokenizer = None
    clear_gpu_cache()
    logger.info("Unloaded LLM")


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
    return lines


@with_retries()
def _generate_script_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a podcast scriptwriter. Output only dialogue lines prefixed "
                "with [Host]: or [Guest]: on separate lines."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(settings.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=settings.llm_temperature,
            top_p=settings.llm_top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs.input_ids.shape[1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def _fallback_script(topic: str) -> list[DialogueLine]:
    return [
        DialogueLine("host", f"Welcome to our podcast about {topic}."),
        DialogueLine("guest", f"Thanks for having me. {topic} is a fascinating subject."),
        DialogueLine("host", "Let us dive in."),
    ]


@node_handler("script")
def script_generator_node(state: PodcastState) -> dict:
    """Generate podcast script using LLM."""
    model, tokenizer = _load_model()

    duration = state["duration_mins"]
    research = state["research_data"]
    topic = state["topic"]
    target_lines = compute_target_lines(duration)
    max_new_tokens = compute_max_new_tokens(duration)

    prompt = f"""Write a podcast script for a {duration}-minute episode about: {topic}

Background research:
{research}

Format rules:
- Two speakers: [Host] and [Guest]
- Natural conversational tone
- Start with intro, cover main points, end with summary
- About {target_lines} dialogue exchanges
- Each line should be 1-3 sentences
- Output only lines starting with [Host]: or [Guest]:

Example format:
[Host]: Welcome to the show! Today we are discussing...
[Guest]: Thanks for having me. This topic is fascinating because...

Write the full script:"""

    logger.info("Generating script (~%d lines, max %d tokens)", target_lines, max_new_tokens)
    response = _generate_script_text(model, tokenizer, prompt, max_new_tokens)
    script = _parse_script(response)

    if not script:
        logger.warning("Script parse failed, retrying with stricter format")
        retry_prompt = (
            f"{prompt}\n\nYour previous output was invalid. "
            "Return only [Host]: and [Guest]: lines, one per line."
        )
        response = _generate_script_text(model, tokenizer, retry_prompt, max_new_tokens)
        script = _parse_script(response)

    if not script:
        logger.warning("Using fallback script after parse failure")
        script = _fallback_script(topic)

    logger.info("Generated %d dialogue lines", len(script))
    clear_gpu_cache()
    return {"script": script, "current_line_idx": 0}
