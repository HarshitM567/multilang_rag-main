import os
from typing import List

def generate_answer(query: str, contexts: List[str], target_lang: str) -> str:
    """
    Azure OpenAI-based generation if Azure credentials are present; otherwise extractive synthesis.
    """
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if azure_key and azure_endpoint and azure_deployment:
        try:
            from openai import AzureOpenAI
            client = AzureOpenAI(
                api_key=azure_key,
                azure_endpoint=azure_endpoint,
                api_version="2024-12-01-preview"
            )
            system = (
                "You are a helpful, concise teacher. "
                "Answer using ONLY the provided context. If unsure, say you don't know. "
                "Cite bullet points from context as needed. Keep chain-of-thought hidden."
            )
            prompt = "Question: " + query + "\n\nContext:\n- " + "\n- ".join(contexts) + "\n\nAnswer:"
            out = client.chat.completions.create(
                model=azure_deployment,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                temperature=1
            )
            return out.choices[0].message.content.strip()
        except Exception:
            pass
    # Fallback extractive: return top few sentences from contexts that match keywords
    import re
    key_terms = re.findall(r"\w+", query.lower())
    scored = []
    for c in contexts:
        score = sum(c.lower().count(k) for k in key_terms)
        scored.append((score, c))
    scored.sort(reverse=True, key=lambda x: x[0])
    summary = "\n".join([c for _, c in scored[:2]])
    return "Based on the material:\n" + summary