import pickle
import wandb
import threading
import torch
import sys
import os
import gc
import contextlib
import transformers
import pandas as pd
import textstat

from pathlib import Path

from transformers import AutoTokenizer
from datasets import load_dataset

from vllm import LLM, SamplingParams
from utils.gpu_management import reset_vllm_gpu_environment

from zeus.monitor import ZeusMonitor


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


print(f"CUDA available: {torch.cuda.is_available()}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")

with open(f"{Path.home()}/.cache/huggingface/token", "r") as f:
    HF_TOKEN = f.read()
    f.close()


MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct", 
    "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16", 
    # "google/gemma-2-2b-it", 
    # "solidrust/gemma-2-9b-it-AWQ"
]

MAX_SEQ_LEN = 8192
NUM_SAMPLES = 10_000

# See for reference: https://docs.vllm.ai/en/v0.5.5/dev/sampling_params.html
SAMPLING_PARAMS = SamplingParams(
    temperature=0.8, 
    top_p=0.95,
    min_tokens=1,
    max_tokens=128,
)

NUM_GPUS = torch.cuda.device_count()


def add_instruction(sentence_pair, tokenizer: AutoTokenizer = None):

    message = [
        {"role": "system", "content": "You are a helpful chatbot that translates text from German to English. Only provide the translation, nothing else."},
        {"role": "user", "content": {sentence_pair['translation']['de']}}
        # {"role": "user", "content": f"Please translate the following sentence from German to English: \n\n{sentence_pair['translation']['de']}"}
    ]

    sentence_pair["input_formatted"] = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    sentence_pair["target"] = sentence_pair["translation"]["en"]
    
    return sentence_pair


def compute_text_metrics(row):

    text = row["input_text"]

    row["flesch_reading_ease"] = textstat.flesch_reading_ease(text)
    row["smog_index"] = textstat.smog_index(text)
    row["automated_readability_index"] = textstat.automated_readability_index(text)
    row["lexical_diversity"] = len(set(text.split())) / len(text.split()) if len(text.split()) > 0 else 0
    row["syllable_count"] = textstat.syllable_count(text)
    row["complex_word_count"] = textstat.difficult_words(text)
    row["avg_word_length"] = sum(len(word) for word in text.split()) / len(text.split()) if len(text.split()) > 0 else 0
    row["sentence_length"] = len(text.split())
    row["flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(text)
    row["coleman_liau_index"] = textstat.coleman_liau_index(text)
    row["dale_chall_readability_score"] = textstat.dale_chall_readability_score(text)
    row["linsear_write_formula"] = textstat.linsear_write_formula(text)
    row["text_standard"] = textstat.text_standard(text)
    row["fernandez_huerta"] = textstat.fernandez_huerta(text)
    row["szigriszt_pazos"] = textstat.szigriszt_pazos(text)
    row["gutierrez_polini"] = textstat.gutierrez_polini(text)
    row["crawford"] = textstat.crawford(text)
    row["gulpease_index"] = textstat.gulpease_index(text)
    row["osman"] = textstat.osman(text)

    return row


def main(): 
    dataset = load_dataset('wmt14', 'de-en', split='train')
    dataset = dataset.shuffle().select(range(NUM_SAMPLES))

    df = pd.DataFrame()
    df["input_text"] = [dataset[idx]['translation']['de'] for idx in range(len(dataset["translation"]))]

    # When using Zeus, you must disable RAPL CPU monitoring as this will cause the program to fail. 
    # Change "return True" to "return False" in file venv/lib/python3.12/site-packages/zeus/device/cpu/rapl.py (line 137)
    monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])

    for model_name in MODELS: 

        tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512)
        tokenizer.pad_token = tokenizer.eos_token
        # print(tokenizer.chat_template)
        dataset_formatted = dataset.map(lambda sentence_pair: add_instruction(sentence_pair, tokenizer), remove_columns=['translation'])

        model = LLM(
            model_name, 
            max_model_len=MAX_SEQ_LEN, 
            trust_remote_code=True
        )
        
        monitor.begin_window("pass")
        outputs = model.generate(dataset_formatted["input_formatted"], SAMPLING_PARAMS)
        measurement = monitor.end_window("pass")

        processing_time_per_sample = measurement.time / NUM_SAMPLES
        energy_per_sample = measurement.total_energy / NUM_SAMPLES

        for idx, output in enumerate(outputs): 
            df.loc[df["input_text"] == dataset[idx]['translation']['de'], f"output_{model_name.replace('/', '_')}"] = output.outputs[0].text
            df.loc[df["input_text"] == dataset[idx]['translation']['de'], f"energy_{model_name.replace('/', '_')}"] = energy_per_sample
            df.loc[df["input_text"] == dataset[idx]['translation']['de'], f"time_{model_name.replace('/', '_')}"] = processing_time_per_sample
        
        df.to_csv(f"simulation_data.csv")
        reset_vllm_gpu_environment(model)

    df = df.apply(compute_text_metrics, axis=1)
    df.to_csv(f"simulation_data.csv")


if __name__ == "__main__": 
    main()
    