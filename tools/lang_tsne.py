'''
Example usage:
# python tools/codebook_frequencies.py data/samples_1h data/models/CLSRIL-23.pt data/outputs
'''
from __future__ import unicode_literals
import os
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import inference

class CodebookFrequenciesVisualizer:

    def __init__(self, dataset_path: str,
                checkpoint_path: str, arch_type: str='large', use_temp=False):
        self.dataset = self._prepare_dataset(dataset_path)
        if use_temp:
            logger.info("Skipped Model Loading step..")
            self.model = inference.Wav2VecInferenceModule(
                checkpoint_path, arch_type=arch_type)
        else:
            self.model = inference.Wav2VecInferenceModule(
                checkpoint_path, arch_type=arch_type)
    
    def get_lang_tsne(self, use_temp=False):
        
        if use_temp:
            # Avoid recalculating tsne vectors
            temp_file = 'outputs/all_lang_vectors_large.npy'
            all_lang_vectors = np.load(temp_file, allow_pickle=True)
            logger.info('Loaded precomputed distribution from {}.'.format(
                            temp_file))
            return all_lang_vectors
        else:
            pbar = tqdm(self.dataset.items())
            all_lang_vectors = []
            for (language, files) in pbar:
                lang_outputs = []
                for i in range(12):
                    lang_outputs.append([])
                pbar.set_description("Processing {}..".format(language))
                codebook_counter = Counter()
                for file in tqdm(files):
                    wav, sr = sf.read(file)
                    assert sr==16000, "Wav file is not 16000 Hz."
                    try:
                        transformer_outs = self.model.get_transformer_outputs(wav)
                        for i, layer_out in enumerate(transformer_outs):
                            sentence_mean = layer_out.mean(dim=0).cpu().numpy()
                            lang_outputs[i].append(sentence_mean)
                    except:
                        # Runtime error occurs if the input wav file is too small,
                        # i.e less than one codebook length (< 320)
                        # Assertion errror occurs for large models for small files
                        pass
                    # break
                lang_outputs = np.array(lang_outputs)
                lang_outputs = np.mean(lang_outputs, axis=1)
                all_lang_vectors.append(lang_outputs)
            
            all_lang_vectors = np.array(all_lang_vectors)
            print(all_lang_vectors.shape)
            
            Path('outputs/').mkdir(parents=True, exist_ok=True)
            temp_file = 'outputs/all_lang_vectors_large.npy'
            np.save(temp_file, all_lang_vectors)
            return all_lang_vectors

    def _prepare_dataset(self, dataset_path):

        logger.info("Preparing dataset..")
        language_list = sorted(os.listdir(dataset_path))
        dataset = {}
        for language in language_list:
            language_folder = os.path.join(dataset_path, language)
            files = os.listdir(language_folder)
            filepaths = list(map(lambda x: os.path.join(language_folder, x),
                            files))
            dataset[language] = filepaths
        logger.info("Dataset Prepared")
        return dataset
    
    def plot_tse(self, all_lang_vectors, output_folder: str):

        language_names = list(self.dataset.keys())
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        for layer in range(5,6):
            full_vectors = all_lang_vectors[:,layer,:]
            dimension_reducer = TSNE(n_components=2, verbose=1, random_state=42, n_jobs=-1, perplexity=3)
            # dimension_reducer = PCA(n_components=2, random_state=42)
            vectors_2d = dimension_reducer.fit_transform(full_vectors)

            output_destination = os.path.join(output_folder, "tsne_tlayer_{}.png".format(layer))

            lang_fams = []
            for language in language_names:
                if language in ['tamil', 'malayalam', 'telugu', 'kannada']:
                    lang_fams.append(1)
                elif language in ['indian_english']:
                    lang_fams.append(2)
                elif language in ['santali']:
                    lang_fams.append(3)
                else:
                    lang_fams.append(0)
            lang_fams = np.array(lang_fams)
            # print(lang_fams)
            # print(lang_fams == 0)

            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(111)
            ax.scatter(vectors_2d[:,0], vectors_2d[:,1])
            # ax.scatter(vectors_2d[(lang_fams==0),0], vectors_2d[(lang_fams==0),1], label='Indo-Aryan')
            # ax.scatter(vectors_2d[(lang_fams==1),0], vectors_2d[(lang_fams==1),1], label='Dravidian')
            # ax.scatter(vectors_2d[(lang_fams==2),0], vectors_2d[(lang_fams==2),1], label='Indian English')
            # ax.scatter(vectors_2d[(lang_fams==3),0], vectors_2d[(lang_fams==3),1], label=' Austro-Asiatic')
            for i, txt in enumerate(language_names):
                ax.annotate(txt.replace('_', ' ').title(), (vectors_2d[i,0]+5, vectors_2d[i,1]+5))
            plt.tight_layout()
            # plt.legend()
            plt.show()
            plt.savefig(output_destination, dpi=300)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('checkpoint_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--arch_type', type=str, default='large',
                        choices=['large', 'base'])
    parser.add_argument('--use_temp', action='store_true')

    args = parser.parse_args()
    
    visualizer = CodebookFrequenciesVisualizer(args.dataset_path,
            args.checkpoint_path, args.arch_type, args.use_temp)
    all_lang_vectors = visualizer.get_lang_tsne(args.use_temp)
    visualizer.plot_tse(all_lang_vectors, args.output_path)
