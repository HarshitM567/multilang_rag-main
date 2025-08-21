import os
from langdetect import detect
from typing import Optional

def detect_lang(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"

def translate_text(text: str, target_lang: str, provider_priority: Optional[list] = None) -> str:
    """
    Translate using preferred providers if keys available.
    provider_priority: e.g., ["openai", "deepl"]
    """
    if not provider_priority:
        provider_priority = ["openai", "deepl"]

    for provider in provider_priority:
        if provider == "openai" and os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                client = OpenAI()
                sys = f"You are a professional translator. Translate to '{target_lang}'. Keep meaning and tone; preserve technical terms."
                out = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": sys},
                        {"role": "user", "content": text},
                    ],
                    temperature=0.2
                )
                return out.choices[0].message.content.strip()
            except Exception:
                pass
        if provider == "deepl" and os.getenv("DEEPL_API_KEY"):
            try:
                import deepl
                translator = deepl.Translator(os.getenv("DEEPL_API_KEY"))
                # auto-detect source
                res = translator.translate_text(text, target_lang=target_lang.upper())
                return res.text
            except Exception:
                pass
    # fallback: no-op
    return text
