# Indic Wav2vec Visualization

## Setup

### Setting up IndicWav2Vec
```
git clone https://github.com/AI4Bharat/IndicWav2Vec
```
Follow the Setup instructions from here - https://github.com/AI4Bharat/IndicWav2Vec/blob/main/w2v_inference/README.md

This will install essential packages like pytorch and fairseq that are required for the visualization.

### Preparing Models
1. Create a folder for storing models - `mkdir models`
2. Download the pretrained base and large models from here - https://indicnlp.ai4bharat.org/indicwav2vec/#pretrained-models
2. Download the finetuned models from here - https://indicnlp.ai4bharat.org/indicwav2vec/#fine-tuned-models

### Preparing Data

1. Create a folder for storing data - `mkdir data`
2. Create separate subfolders for different languages. You can put your audio files in `wav` format in respective folders. The directory structure should look something like this:
```
data
├── assamese
│   ├── audio1.wav
│   ├── audio2.wav
│   └── audio3.wav
├── bengali
├── gujarati
├── hindi
...
├── santali
├── tamil
└── telugu
```

## t-SNE plot showing language representations

Usage -

```
CUDA_VISIBLE_DEVICES=0 python tools/lang_tsne.py \
    <data_directory> \
    <path_to_model.pt> \
    <output_path> \
    --arch_type <base/large>
```

Example script - 
```
CUDA_VISIBLE_DEVICES=0 python tools/lang_tsne.py \
    data/ 
    models/wav2vec_large.pt \
    outputs/language_plots/ai4b_large \
    --arch_type large
```

## Plot codebook distribution accross languages

Usage -

```
CUDA_VISIBLE_DEVICES=0 python tools/codebook_frequencies.py \
    <data_directory> \
    <path_to_model.pt> \
    <output_path> \
    --arch_type <base/large>
```

Example script - 
```
CUDA_VISIBLE_DEVICES=0 python python tools/codebook_frequencies.py \
    data/ 
    models/wav2vec_large.pt \
    outputs/codebooks/ai4b_large \
    --arch_type large
```

## Plot Attention Heads

1. Change the audio paths in `examples/example_outputs.txt` file.
2. Run the notebook - `viz_attention.ipynb`

###  Running on custom data

This step assumes that you have a finetuned ASR model using which you can generate CTC outputs. The expected input format for generating the plot is shown in - `examples/example_outputs.txt`
The file is a Tab-separated file containing the following items in each row -
1. Row Id
2. Absolute Path to the audio file
3. Transcription obtained from the model
4. CTC decoding predictions from the model in the form of python list containing tuples. Each tuple is of the form `[start_id, character]`
    1. start_id: The CTC decoder outputs a character at each timestep. In CTC, repeated characters are predicted over several timesteps. In start_id, we store only the starting index of each character.
    2. character: the character output from the CTC model.



