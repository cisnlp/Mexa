import argparse
import os
import torch
import json
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Function to handle weighted embeddings
def weighted_embeddings(layer, attention_mask, device='cuda'):
    # Compute weights for non-padding tokens
    weights_for_non_padding = attention_mask * torch.arange(start=1, end=layer.shape[1] + 1, device=device).unsqueeze(0)
    sum_embeddings = torch.sum(layer * weights_for_non_padding.unsqueeze(-1), dim=1)
    num_of_non_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
    sentence_embeddings = sum_embeddings / num_of_non_padding_tokens
    sentence_embeddings = sentence_embeddings.squeeze().cpu().numpy()
    return sentence_embeddings

# Function to extract embeddings
def get_embedding_layers(text, model, tokenizer, device='cuda'):
    tokens = tokenizer(text, return_tensors='pt', padding=True).to(device)
    attention_mask = tokens.attention_mask.to(device)
    attention_mask_last = torch.zeros_like(attention_mask).to(device)
    attention_mask_last[:, -1] = 1

    sentence_embeddings_weighted = []
    sentence_embeddings_last_token = []
    
    with torch.no_grad():
        hidden_state_layers = model(**tokens, output_hidden_states=True)["hidden_states"]

        for layer in hidden_state_layers:
            embd_weighted = weighted_embeddings(layer, attention_mask, device)
            embd_last_token = weighted_embeddings(layer, attention_mask_last, device)

            sentence_embeddings_weighted.append(embd_weighted)
            sentence_embeddings_last_token.append(embd_last_token)

    return sentence_embeddings_weighted, sentence_embeddings_last_token

# Main function
def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from a model")

    # Add arguments for the parser
    parser.add_argument('--model_name', type=str, required=True, help='The model name from Hugging Face.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the parallel data directory.')
    parser.add_argument('--gpus', type=str, default='0', help='GPUs to use, e.g. "0".')
    parser.add_argument('--num_sents', type=int, default=100, help='Maximum number of sentences to process.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the embeddings.')
    parser.add_argument('--token', type=str, default=None, help='Hugging Face token (optional).')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Directory for caching the model (optional).')
    parser.add_argument('--file_ext', type=str, default='.txt', help='File extension for input files (optional, default: .txt).')

    # Parse the arguments
    args = parser.parse_args()

    # Set GPU environment
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # Define model name and token
    model_name = args.model_name
    token = args.token  # Optional token

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', cache_dir=args.cache_dir, use_auth_token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    tokenizer.pad_token = tokenizer.eos_token

    # Directory and number of sentences
    directory = args.data_path
    number_of_sents = args.num_sents

    # Initialize a dictionary to store embeddings
    result_dict = {}

    # Process the files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(args.file_ext):
            language = filename.split('.')[0]
            filepath = os.path.join(directory, filename)
            
            sentences = []
            with open(filepath, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                for idx, line in enumerate(lines):
                    if idx < number_of_sents:
                        sentence = line.strip()
                        sentences.append({'id': idx + 1, 'text': sentence})

            result_dict[language] = sentences

    # Prepare to save embeddings
    embeddings_dict = {}

    # Extract embeddings
    for language, texts in tqdm(result_dict.items()):
        embeddings_dict = {}

        for text in texts:
            embds_weighted, embds_last_token = get_embedding_layers(text['text'], model, tokenizer)

            for layer in range(len(embds_weighted)):
                if layer not in embeddings_dict:
                    embeddings_dict[layer] = []

                embeddings_dict[layer].append({
                    'id': text['id'],
                    'embd_weighted': embds_weighted[layer],
                    'embd_lasttoken': embds_last_token[layer]
                })

        # Save the embeddings as pickle
        save_filepath = os.path.join(args.save_path, f"{language}.pkl")
        os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
        with open(save_filepath, "wb") as pickle_file:
            pickle.dump(embeddings_dict, pickle_file)

if __name__ == "__main__":
    main()
