import argparse
import torch
import os
import pandas as pd
import pyarrow

from transformers import AutoTokenizer, AutoModelForMaskedLM


torch.set_float32_matmul_precision('high')


MODEL_NAME = "answerdotai/ModernBERT-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mean_pooling(model_output, attention_mask):
    # We need to make sure we only pool actual tokens and not padding.
    # Please see for reference: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
    return torch.sum(model_output * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def main(source_folder: str, target_folder: str, include_pca: bool = False, pca_dim: int = 25, minibatch_size: int = 64):

    print(f"Using PCA: {include_pca}.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    model = model.to(DEVICE)

    files = os.listdir(source_folder)
    print(f"Found {len(files)} files for processing.")

    for idx, file_name in enumerate(files):
        try:
            df = pd.read_parquet(f"{source_folder}/{file_name}")
        except pyarrow.lib.ArrowInvalid as e:
            print(f"Error: Encountered error while reading file {file_name}. Need to check manually.")
            print(e)
    
        tokenized_data = tokenizer.batch_encode_plus(
            df["input_text"].tolist(),
            truncation=True,   
            padding='max_length',
            return_tensors='pt',  	
            add_special_tokens=True, 
            max_length=512
        )
    
        input_ids_chunked = torch.split(tokenized_data["input_ids"], minibatch_size)
        attn_mask_chunked = torch.split(tokenized_data["attention_mask"], minibatch_size)
    
        emb_list = []
        pca_list = []
        for jdx, (input_ids, attn_mask) in enumerate(zip(input_ids_chunked, attn_mask_chunked)): 
            input_ids = input_ids.to(DEVICE)
            attn_mask = attn_mask.to(DEVICE)
        
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
            
            word_embeddings = outputs.hidden_states[-1]

            # At this point we can perform an avg_pool (procedure identical to SentenceTransformers) 
            # and/or we can run a PCA on the word embeddings (this might be helpful to identify "interesting" sequences).
            if include_pca:
                # This is the PCA code. Depending on what we want to do, there are some interesting resources available: 
                #     - PyTorch PCA: https://pytorch.org/docs/stable/generated/torch.pca_lowrank.html 
                #     - Discussion on an embedding approximation: https://stackoverflow.com/questions/75796047/how-to-evaluate-the-quality-of-pca-returned-by-torch-pca-lowrank
                #     - Batched processing: https://github.com/pytorch/pytorch/issues/99705
                U, S, V = torch.pca_lowrank(word_embeddings, q=pca_dim, center=True, niter=2)
                S = S.cpu()
                pca_list += [i for i in S]
                # TODO: We may have to assess the PCA quality at some point.
                
            # document_embedding = torch.nanmean(word_embeddings, dim=1)
            document_embeddings = mean_pooling(word_embeddings, attention_mask=attn_mask)
            document_embeddings = document_embeddings.cpu()
            
            emb_list += [i for i in document_embeddings]
            print(f"Completed minibatch #{jdx} for file #{idx} of {len(files)}.")
    
        df["input_text_modern_bert_embed"] = pd.Series(emb_list).astype(object)

        if include_pca: 
            df["input_text_modern_bert_pca_{pca_dim}_dims"] = pd.Series(pca_list).astype(object)

        try:
            df.to_parquet(f"{target_folder}/{file_name}")
            print(f"Completed file #{idx} of {len(files)}.")
        except pyarrow.lib.ArrowInvalid as e:
            print(f"Error: Encountered corrupt tensor during Arrow conversion for file {file_name}. Need to check manually.")
            print(e)


if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Embedding generator.')
    parser.add_argument('--source-folder', type=str, help='Path to folder that contains source files. Relative to script.')
    parser.add_argument('--target-folder', type=str, help='Path to output folder. Relative to script.')
    parser.add_argument('--include-pca', type=bool, default="store_false", help='Include PCA embedding.', required=False)
    parser.add_argument('--pca-dim', type=int, default=25, help='Dimensions for PCA.', required=False)
    # MBS = 64 requires about 20GB VRAM.
    parser.add_argument('--minibatch-size', type=int, default=64, help='Minibatch size to use with the Modern Bert Model.')
    args = parser.parse_args()

    main(
        source_folder=args.source_folder,
        target_folder=args.target_folder,
        include_pca=args.include_pca,
        pca_dim=args.pca_dim,
        minibatch_size=args.minibatch_size
    )

    print("Done.")
