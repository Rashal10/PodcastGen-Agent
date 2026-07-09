import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from ..config import LLM_MODEL, DEVICE, QUANTIZATION_CONFIG
from ..state import PodcastState, DialogueLine


_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    
    print(f"[Script] Loading {LLM_MODEL}...")
    
    quant_config = BitsAndBytesConfig(**QUANTIZATION_CONFIG)
    
    _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    _model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return _model, _tokenizer


def _parse_script(raw_text: str) -> list[DialogueLine]:
    """Extract dialogue lines from LLM output."""
    lines = []
    pattern = r"[*\s\[]*(host|guest)[*\s\]]*:?[*\s]*(.+?)(?=(?:[*\s\[]*(?:host|guest)[*\s\]]*:?|$))"
    
    matches = re.findall(pattern, raw_text, re.IGNORECASE | re.DOTALL)
    for speaker, text in matches:
        cleaned = text.strip().strip('"').strip()
        if cleaned:
            lines.append(DialogueLine(speaker=speaker.lower(), text=cleaned))
    
    return lines


def script_generator_node(state: PodcastState) -> dict:
    """Generate podcast script using LLM."""
    model, tokenizer = _load_model()
    
    duration = state["duration_mins"]
    research = state["research_data"]
    topic = state["topic"]
    
    target_lines = (duration * 150) // 10
    
    prompt = f"""Write a podcast script for a {duration}-minute episode about: {topic}

Background research:
{research}

Format rules:
- Two speakers: [Host] and [Guest]
- Natural conversational tone
- Start with intro, cover main points, end with summary
- About {target_lines} dialogue exchanges
- Each line should be 1-3 sentences

Example format:
[Host]: Welcome to the show! Today we're discussing...
[Guest]: Thanks for having me. This topic is fascinating because...

Write the full script:"""

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    
    print(f"[Script] Generating script (~{target_lines} lines)...")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    script = _parse_script(response)
    
    if not script:
        script = [
            DialogueLine("host", f"Welcome to our podcast about {topic}."),
            DialogueLine("guest", f"Thanks for having me. {topic} is a fascinating subject."),
            DialogueLine("host", "Let's dive in."),
        ]
    
    print(f"[Script] Generated {len(script)} dialogue lines")
    
    torch.cuda.empty_cache()
    
    return {"script": script, "current_line_idx": 0}
