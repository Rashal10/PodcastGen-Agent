import re


def clean_text_for_tts(text: str) -> str:
    """Clean text for better TTS output."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)

    abbrevs = {
        "AI": "A.I.",
        "ML": "M.L.",
        "API": "A.P.I.",
        "GPU": "G.P.U.",
        "CPU": "C.P.U.",
    }
    for abbr, expanded in abbrevs.items():
        text = re.sub(rf"\b{abbr}\b", expanded, text)
 
    text = " ".join(text.split())
    
    return text


def chunk_text(text: str, max_chars: int = 250) -> list[str]:
    """Split long text into chunks at sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    
    chunks = []
    current = ""
    
    for sent in sentences:
        if len(current) + len(sent) < max_chars:
            current += " " + sent if current else sent
        else:
            if current:
                chunks.append(current.strip())
            current = sent
    
    if current:
        chunks.append(current.strip())
    
    return chunks
