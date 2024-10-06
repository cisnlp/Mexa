import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
import argparse

def cosine_similarity(array1, array2):
    cosine_dist = cosine(array1, array2)
    cosine_similarity = 1 - cosine_dist
    return cosine_similarity

def mexa(matrix):
    n = len(matrix)  # size of the square matrix
    count = 0
    
    for i in range(n):
        # Get the diagonal element
        diag_element = matrix[i][i]
        
        # Get the row and column
        row = matrix[i]
        column = matrix[:,i]
        
        # Check if the diagonal element is strictly greater than all other elements in its row (excluding itself)
        if diag_element > max(np.delete(row, i)):
            # Check if the diagonal element is strictly greater than all other elements in its column (excluding itself)
            if diag_element > max(np.delete(column, i)):
                count += 1

    # Normalized count
    count_norm = count / n
    return count_norm

def compute_distance(lang, embedding_type='embd_weighted', num_sents=100):
    with open(os.path.join(embedding_path, f"{lang}.pkl"), "rb") as pickle_file:
        lang_embd = pickle.load(pickle_file)    

    similarities_dict = {}
    for layer in lang_embd.keys():
        pivot_embd_layer = pivot_embd[layer][:num_sents]
        lang_embd_layer = lang_embd[layer][:num_sents]
        
        # Initialize the similarities_dict matrix for each layer
        num_actual_sentences = min(len(pivot_embd_layer), len(lang_embd_layer))
        similarities_dict[layer] = np.zeros((num_actual_sentences, num_actual_sentences))
        
        # Compute similarities
        for p_id, pivot_single in enumerate(pivot_embd_layer):
            for l_id, lang_single in enumerate(lang_embd_layer):
                similarities_dict[layer][p_id, l_id] = cosine_similarity(pivot_single[embedding_type], lang_single[embedding_type])

    alignments = {}
    for layer in lang_embd.keys():
        alignments[layer] = mexa(similarities_dict[layer])
    
    return alignments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process embeddings and compute alignments.')
    
    parser.add_argument('--pivot', type=str, default='eng_Latn', help='Pivot language code (default: eng_Latn)')
    parser.add_argument('--file_ext', type=str, default='.pkl', help='File extension for embedding files (default: .pkl)')
    parser.add_argument('--embedding_path', type=str, required=True, help='Path to the directory containing embedding files.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the results.')
    parser.add_argument('--num_sents', type=int, default=100, help='Maximum number of sentences to process (default: 100)')
    parser.add_argument('--embedding_type', type=str, choices=['embd_weighted', 'embd_lasttoken'], default='embd_weighted', help='Type of embedding to use (default: embd_weighted)')

    args = parser.parse_args()

    # Set the global variables based on input arguments
    pivot = args.pivot
    file_ext = args.file_ext
    embedding_path = args.embedding_path
    save_path = args.save_path
    num_sents = args.num_sents
    embedding_type = args.embedding_type

    # Load the pivot embeddings
    with open(os.path.join(embedding_path, f"{pivot}{file_ext}"), "rb") as pickle_file:
        pivot_embd = pickle.load(pickle_file)

    languages = [filename[:-len(file_ext)] for filename in os.listdir(embedding_path) if filename.endswith(file_ext)]

    for lang in tqdm(languages):
        alignment_lang = compute_distance(lang, embedding_type=embedding_type, num_sents=num_sents)
        save_filepath = os.path.join(save_path, f"{lang}.json")
        os.makedirs(os.path.dirname(save_filepath), exist_ok=True)

        with open(save_filepath, "w") as json_file:
            json.dump(alignment_lang, json_file)
