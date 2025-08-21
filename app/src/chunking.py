from typing import List, Dict
import re
import blingfire
import tiktoken

def sentence_split(text: str) -> List[str]:
    # BlingFire handles many scripts robustly
    sents = blingfire.text_to_sentences(text).split("\n")
    # Remove empty lines
    return [s.strip() for s in sents if s.strip()]

def chunk_by_tokens(text: str, max_tokens: int = 400, overlap: int = 80, model_name: str = "gpt-4o-mini") -> List[Dict]:
    enc = tiktoken.get_encoding("cl100k_base")  # broadly compatible
    sents = sentence_split(text)
    chunks = []
    buf = []
    buf_tokens = 0

    def tok_len(t): 
        return len(enc.encode(t))

    for s in sents:
        s_tokens = tok_len(s)
        if buf_tokens + s_tokens <= max_tokens:
            buf.append(s); buf_tokens += s_tokens
        else:
            # flush buffer
            chunk_text = " ".join(buf).strip()
            if chunk_text:
                chunks.append({"text": chunk_text})
            # start new with overlap: take last sentences until overlap covered
            # create overlap by sliding window on sentence level
            while buf and tok_len(" ".join(buf)) > overlap:
                buf.pop(0)
            buf.append(s)
            buf_tokens = tok_len(" ".join(buf))
    # final flush
    chunk_text = " ".join(buf).strip()
    if chunk_text:
        chunks.append({"text": chunk_text})
    return chunks
