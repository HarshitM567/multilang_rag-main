"""
Minimal RAGAS evaluation entry point. Fill with your (question, answer, contexts) tuples.
"""
from typing import List, Dict
import pandas as pd

def to_dataset(records: List[Dict]):
    """
    records: [{"question":..., "answer":..., "contexts":[...]}, ...]
    """
    return pd.DataFrame(records)

if __name__ == "__main__":
    # Example stub
    data = [
        {
            "question": "What are the four stages of the water cycle?",
            "answer": "Evaporation, condensation, precipitation, and collection.",
            "contexts": ["The water cycle has four main stages: evaporation, condensation, precipitation, and collection."]
        }
    ]
    df = to_dataset(data)
    print(df.head())
