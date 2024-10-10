# MEXA
__MEXA__ stands for **M**ultilingual **E**valuation via **Cross**-Lingual **A**lignment, [Paper, arXiv 2024](http://arxiv.org/abs/2410.05873)

We introduce MEXA, a method for assessing the multilingual capabilities of English-centric large language models (LLMs). MEXA builds on the observation that English-centric LLMs semantically use English as a kind of pivot language in their intermediate layers. MEXA computes the alignment between non-English languages and English using parallel sentences, estimating the transfer of language understanding capabilities from English to other languages through this alignment. This metric can be useful in estimating task performance, provided we know the English performance in the task and the alignment score between languages derived from a parallel dataset.

## Compute

Follow steps 1 to 3, prepare your data, and run 2 commands!

### 1) Preparing a Parallel Dataset

Save the parallel data in the following format:

Each language should have one text file named after the language (e.g., `eng_Latn.txt`), containing $n$ sentences. The line number corresponds to the sentence ID, meaning line $i$ in each file is parallel across languages.

- We use [FLORES-200](https://github.com/facebookresearch/flores/blob/main/flores200/README.md), available for download [here](https://tinyurl.com/flores200dataset). For our experiments, we use the first 100 sentences from the devtest folder.
- We also use the Bible dataset, which contains 103 sentences across 1,401 languages in the same format, accessible [here](https://huggingface.co/datasets/cis-lmu/sPBC).

### 2) Computing Embeddings with embed_extractor.py

The `embed_extractor.py` script allows you to extract embeddings from a specified model for a given dataset. 
It generates embeddings based on two methods: a __weighted average__ based on token positions and the __last token__.
Below are the instructions for using the script along with descriptions for each argument.

To run the script, use the following command:

```bash
python embed_extractor.py --model_name <MODEL_NAME> --data_path <DATA_PATH> --gpus <GPU_IDS> --num_sents <NUM_SENTENCES> --save_path <SAVE_PATH> --cache_dir <CACHE_DIR> --file_ext <FILE_EXTENSION> --token <HUGGING_FACE_TOKEN>
```

<details> <summary> <b> Click to Expand Arguments Description </b>  </summary>

- `--model_name` (str, required):  
  The name of the model to use for embedding extraction. It can be any compatible model from Hugging Face.  
  **Examples**:
  - `"google/gemma-2-9b"`
  - `"google/gemma-7b"`
  - `"meta-llama/Meta-Llama-3.1-70B"`
  - `"meta-llama/Llama-3.1-8B"`
  - `"meta-llama/Meta-Llama-3-8B"`
  - `"meta-llama/Llama-2-7b-hf"`
  - `"yahma/llama-7b-hf"` 
  - `"mistralai/Mistral-7B-v0.3"`
  - `"allenai/OLMo-1.7-7B-hf"`


- `--data_path` (str, required):  
  The path to the directory containing the parallel data files.

- `--gpus` (str, default='0'):  
  The GPU IDs to use for processing. You can specify a single GPU (e.g., `"0"`) or multiple GPUs separated by commas (e.g., `"0,1"`).

- `--num_sents` (int, default=100):  
  The maximum number of sentences to process from each input file. The default value is 100, but you can adjust it as needed.

- `--save_path` (str, required):  
  The path where the extracted embeddings will be saved. Ensure that the directory exists or the script has permission to create it.

- `--token` (str, optional, default=None):  
  Your Hugging Face token for authentication (if required). This is optional and can be omitted if the model does not require authentication.

- `--cache_dir` (str, optional, default='./cache'):  
  The directory where the model will be cached after downloading. This prevents re-downloading the model for future runs.

- `--file_ext` (str, optional, default='.txt'):  
  The file extension of the input files containing the parallel data. The default is `.txt`, but you can specify a different extension as needed (e.g., `.devtest`).

</details>

__Example Command:__

To extract embeddings using the `allenai/OLMo-1.7-7B-hf` model from the `./flores200_dataset/devtest` directory and save the results in `./embd_olmo`, processing the first 100 sentences of each file, use the following command:

```bash
python embed_extractor.py --model_name allenai/OLMo-1.7-7B-hf --data_path ./flores200_dataset/devtest --gpus '0' --num_sents 100 --save_path ./embd_olmo/ --cache_dir ./cache/ --file_ext .devtest
```


### 3) Computing MEXA Score with compute_mexa.py


The `compute_mexa.py` script computes mexa score between embeddings from a pivot language and multiple target languages. It uses cosine similarity to evaluate the embeddings and outputs the alignment scores as JSON files.

To execute the script, use the following command:

```bash
python compute_mexa.py --embedding_path <EMBEDDING_PATH> --save_path <SAVE_PATH> --num_sents <NUM_SENTENCES> --embedding_type <EMBEDDING_TYPE> --pivot <PIVOT_LANG> --file_ext <FILE_EXTENSION>
```

<details> <summary> <b> Click to Expand Arguments Description </b>  </summary>

- `--embedding_path` (str, required):  
  The path to the directory containing the embedding files. Ensure this directory exists and contains the required `.pkl` files.

- `--save_path` (str, required):  
  The path where the computed alignment results will be saved as JSON files. The directory should exist or the script should have permission to create it.

- `--num_sents` (int, optional, default=100):  
  The maximum number of sentences to process from each input file. The default value is 100, but you can adjust it as needed.

- `--embedding_type` (str, optional, default='embd_weighted'):  
  The type of embedding to use. Choose between:
  - `'embd_weighted'`: For weighted average embeddings based on token positions.
  - `'embd_lasttoken'`: For embeddings based on the last token.

- `--pivot` (str, optional, default='eng_Latn'):  
  The language code of the pivot language. This is the language against which other languages will be compared.

- `--file_ext` (str, optional, default='.pkl'):  
  The file extension for the embedding files. The default is `.pkl`, but you can specify a different extension if needed.

</details>

__Example Command:__

To compute alignments using the pivot language `eng_Latn`, processing the first 100 sentences from each embedding file located in `./embd_olmo/` and saving the results in `./mexa_olmo/`, use the following command:

```bash
python compute_mexa.py --embedding_path ./embd_olmo/ --save_path ./mexa_olmo/ --num_sents 100 --embedding_type embd_weighted --pivot eng_Latn --file_ext .pkl
```

## Language Coverage — Computed Scores

We host the estimated Mexa scores, which are calculated using mean and max pooling methods over layers and adjusted based on the models' performance in different tasks in English. These scores are available for popular state-of-the-art models based on FLORES and the Bible at https://huggingface.co/spaces/cis-lmu/Mexa.

## Citation

If you find our method, code and scores useful for your research, please cite:

```bash
@article{kargaran2024mexa,
title        = {{MEXA}: Multilingual Evaluation of {E}nglish-Centric {LLMs} via Cross-Lingual Alignment},
author       = {Kargaran, Amir Hossein and Modarressi, Ali and Nikeghbal, Nafiseh  and Diesner, Jana and Yvon, François and Schütze, Hinrich},
journal      = {arXiv preprint arXiv:2410.05873},
year         = {2024},
url          = {https://arxiv.org/abs/2410.05873}
}
```
