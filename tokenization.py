import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path("data/inference_outputs")

TOKENIZER_NAME = "meta-llama/Llama-3.1-8B-Instruct"              
tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

def prompt_tokens_llama3(prompt: str) -> int:
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=False
    )
    return len(ids)

for fp in ROOT.rglob("*.csv"):
    df = pd.read_csv(fp)

    if {"tokens_small", "tokens_medium", "tokens_large"}.issubset(df.columns):
        print(f"{fp.name}: already done")
        continue

    tqdm.pandas(desc=f"{fp.name}: prompt-tokenise")
    p_tok = df["input_text"].progress_apply(prompt_tokens_llama3)

    df["tokens_small"]  = p_tok
    df["tokens_medium"] = p_tok
    df["tokens_large"]  = p_tok

    df.to_csv(fp, index=False)
    print(f"{fp.name}: token columns written ✓")