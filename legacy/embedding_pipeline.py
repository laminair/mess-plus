import torch 
import math
import pandas as pd

from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset

from tqdm import tqdm

from torch.utils.data import DataLoader

torch.set_float32_matmul_precision('high')


MODEL_NAME = "answerdotai/ModernBERT-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(input_file_path: str, use_pca: bool = False, pca_dims: list = (1, 10, 25, 100), minibatch_size: int = 64):

    df = pd.read_csv(input_file_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    model = model.to(DEVICE)

    input_data = df["input_text"].tolist()
    tokenized_data = tokenizer.batch_encode_plus(
        input_data,        	
        truncation=True,   
        padding='max_length',
        return_tensors='pt',  	
        add_special_tokens=True, 
        max_length=512
    )
    
    input_ids_chunked = torch.split(tokenized_data["input_ids"], minibatch_size)
    attn_mask_chunked = torch.split(tokenized_data["attention_mask"], minibatch_size)

    batch_counter = 0
    for idx, (input_ids, attn_mask) in enumerate(zip(input_ids_chunked, attn_mask_chunked)): 

        base_idx = batch_counter * minibatch_size
        df_idx = base_idx + idx
        
        input_ids = input_ids.to(DEVICE)
        attn_mask = attn_mask.to(DEVICE)
    
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        
        word_embeddings = outputs.hidden_states[-1]

        if use_pca: 
            # This is the PCA code. Depending on what we want to do, there are some interesting resources available: 
            #     - PyTorch PCA: https://pytorch.org/docs/stable/generated/torch.pca_lowrank.html 
            #     - Discussion on an embedding approximation: https://stackoverflow.com/questions/75796047/how-to-evaluate-the-quality-of-pca-returned-by-torch-pca-lowrank
            #     - Batched processing: https://github.com/pytorch/pytorch/issues/99705
            for q in pca_dims:
                # TODO: We may have to assess the PCA quality at some point.
                U, S, V = torch.pca_lowrank(word_embeddings, q=q, center=True, niter=2)

                df.loc[df.index == df_idx, f"pca_dim_{q}"] = S

        document_embedding = torch.nanmean(word_embeddings, dim=1)
        df.loc[df.index == df_idx, f"text_embedding_avg_pool"] = document_embedding

        batch_counter += 1

        df.to_csv(input_file_path)

if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Embedding generator.')
    parser.add_argument('--input-file-path', type=str, help='Path to CSV file containing a column named "input_text".')
    parser.add_argument('--use-pca', type=bool, default=store_false, help='Enable PCA analysis.')
    parser.add_argument('--pca-dims', type=list, help='Comma-separated list of PCA dimensions.')
    # MBS = 64 requires about 20GB VRAM.
    parser.add_argument('--minibatch-size', type=int, default=64, help='Minibatch size to use with the Embedding Model.')
    args = parser.parse_args()

    main(input_file_path=args.input_file_path, use_pca=args.use_pca, pca_dims=args.pca_dims)
    