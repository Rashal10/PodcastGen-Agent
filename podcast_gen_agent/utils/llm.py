"""Shared local LLM load and chat generation for script and critic nodes."""

import logging

import torch

from ..config import settings
from ..utils.gpu import clear_gpu_cache, require_gpu_memory

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None


def load_llm():
    """Load the quantized script LLM, reusing a process-level cache."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    logger.info("Loading LLM: %s", settings.llm_model)
    require_gpu_memory()
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(**settings.quantization_config)
    _tokenizer = AutoTokenizer.from_pretrained(settings.llm_model)
    _model = AutoModelForCausalLM.from_pretrained(
        settings.llm_model,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return _model, _tokenizer


def unload_llm() -> None:
    """Release the LLM from memory."""
    global _model, _tokenizer
    if _model is None and _tokenizer is None:
        return
    _model = None
    _tokenizer = None
    clear_gpu_cache()
    logger.info("Unloaded LLM")


def generate_chat(messages: list[dict[str, str]], max_new_tokens: int) -> str:
    """Run a chat-template completion with the cached LLM."""
    model, tokenizer = load_llm()
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
