'''
Example usage:
# python tools/codebook_frequencies.py data/samples_1h data/models/CLSRIL-23.pt data/outputs
'''
from __future__ import unicode_literals
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from loguru import logger
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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
    
    def get_language_distribution(self, use_temp=False):
        
        if use_temp:
            temp_file = 'outputs/lang_dist.npy'
            language_frequencies = np.load(temp_file, allow_pickle=True).item()
            logger.info('Loaded precomputed distribution from {}.'.format(
                            temp_file))
            return language_frequencies
        else:
            language_frequencies = {}

            def preprocess(language):
                files = self.dataset[language]
                codebook_counter = Counter()
                for file in tqdm(files):
                    wav, sr = sf.read(file)
                    assert sr==16000, "Wav file is not 16000 Hz."
                    try:
                        codebook_sequence = self.model.get_codebook_sequence(wav)
                        codebook_sequence = list(codebook_sequence)
                        codebook_counter.update(codebook_sequence)
                    except RuntimeError:
                        # Runtime error occurs if the input wav file is too small,
                        # i.e less than one codebook length (< 320)
                        pass
                    # break
                frequency = np.zeros(102400)
                for key, value in dict(codebook_counter).items():
                    frequency[key] = value
                distribution = frequency # / frequency.max()
                language_frequencies[language] = distribution

            
            for lang in list(self.dataset.keys()):
                preprocess(lang)
                print(language_frequencies[lang])
                
            Path('outputs').mkdir(parents=True, exist_ok=True)
            temp_file = 'outputs/lang_dist.npy'
            np.save(temp_file, language_frequencies)
            return language_frequencies
    
    def get_language_codebook_usage(self, output_folder: str, language_dist: dict):

        logger.info("Generating Codebook usage per language..")
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        language_usage = {}
        for (language, dist) in language_dist.items():
            non_zeros = np.count_nonzero(dist)
            language_usage[language] = non_zeros
        for (language, counts) in language_usage.items():
            print("{}: {}".format(language, counts))
        df = pd.DataFrame.from_dict(language_usage, orient="index")
        output_destination = os.path.join(output_folder, "language_usage.csv")
        df.to_csv(output_destination, index=False, header=False)
        logger.info('Saved output in {}'.format(output_destination))

          
    def plot_codebook_frequencies(self, output_folder: str, language_dist: dict):
        logger.info("Generating language codebook frequencies..")
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        #############################

        ## For faster loading, use precomupted values
        # temp_file = 'outputs/codebook_1d.npy'
        # codebook_vectors_1d = np.load(temp_file, allow_pickle=True)

        codebook_vectors = self.model.model.quantizer.codebook().cpu().detach()

        # dim_reducer = PCA(n_components=1, random_state=42)
        # codebook_vectors_1d = dim_reducer.fit_transform(codebook_vectors)
        # logger.info('Loaded precomputed distribution from {}.'.format(
        #                 temp_file))
        
        ## T-SNE computation takes a long time. 
        ## Use the precomputed values for faster execution.
        dim_reducer = TSNE(n_components=1, verbose=1, random_state=42)
        codebook_vectors_1d = dim_reducer.fit_transform(codebook_vectors)
        Path('outputs').mkdir(parents=True, exist_ok=True)
        temp_file = 'outputs/codebook_1d.npy'
        np.save(temp_file, codebook_vectors_1d)

        #############################
        # hindi_freq = language_dist['hindi']
        # print(hindi_freq)
        # sorted_indices = np.argsort(-hindi_freq) # negative to sort in descending order
        # print(sorted_indices)
        ############################
        langs, frequencies = list(language_dist.keys()), list(language_dist.values())
        frequencies = np.array(frequencies)
        # frequencies = frequencies[:,sorted_indices]
        sum_rows = frequencies.sum(axis=0)
        frequencies = frequencies[:, (sum_rows != 0)]
        codebook_vectors_1d = codebook_vectors_1d[(sum_rows != 0)]
        frequencies = frequencies / frequencies.max(axis=0)

        ### Sorting by hindi
        hindi_idx = langs.index('hindi')
        sorted_indices = np.argsort(-frequencies[hindi_idx])
        # sorted_indices = np.argsort(-frequencies[20])
        ### Sorting by 1d projection
        # codebook_vectors_1d = np.squeeze(codebook_vectors_1d)
        # sorted_indices = np.argsort(codebook_vectors_1d)
        frequencies = frequencies[:,sorted_indices]

        # deciding_freq = frequencies[:,:10000] # 40 x 40000
        weighted_sum = frequencies * np.flip(np.arange(frequencies.shape[1]))
        correlations = weighted_sum.mean(axis=1)
        correlation_idx = np.argsort(-correlations)
        assert len(correlation_idx) == 40

        frequencies = frequencies[correlation_idx]
        langs = np.array(langs)
        langs = langs[correlation_idx]

        # Create grouping
        # hi_freqs = frequencies[6]
        # hi_freqs[hi_freqs <= 0.33] = 0
        # hi_freqs[hi_freqs > 0.66] = 1
        # hi_freqs[(hi_freqs > 0.33) & (hi_freqs <= 0.66)] = 0.5

        # frequencies[frequencies <= 0.2] = 0
        # frequencies[(frequencies > 0.2) & (frequencies <= 0.4)] = 0.2
        # frequencies[(frequencies > 0.4) & (frequencies <= 0.6)] = 0.4
        # frequencies[(frequencies > 0.6) & (frequencies <= 0.8)] = 0.6
        # # frequencies[(frequencies > 0.8) & (frequencies <= 0.4)] = 0.2
        # frequencies[frequencies > 0.8] = 0.8
        # frequencies[(frequencies > 0.33) & (frequencies <= 0.66)] = 0.5

        import matplotlib
        matplotlib.rc('ytick', labelsize=25)
        matplotlib.rc('xtick', labelsize=20)
        fig = plt.figure(figsize=(14,15))
        ax = fig.add_subplot(111)
        cax = ax.matshow(frequencies, aspect='auto', cmap='coolwarm',
                norm=colors.SymLogNorm(linthresh=0.15, linscale=0.15,
                vmin=0, vmax=1.0),)
        v1 = np.linspace(0, 1, 11, endpoint=True)
        cbar = fig.colorbar(cax, ax=ax)
        cbar.ax.set_yticklabels(['{:.0f}'.format(x) for x in np.arange(0, 10+1, 1)]) # , fontsize=16, weight='bold')
        langs = list(map(lambda x: x.replace('_', ' ').title(), langs))
        ax.set_yticks(np.arange(0, len(language_dist), 1.0))
        ax.set_yticklabels(list(langs))
        ###################

        output_destination = os.path.join(output_folder, "language_codebook_matrix.png")
        plt.tight_layout()
        plt.show()
        plt.savefig(output_destination)

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('checkpoint_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--arch_type', type=str, default='large',
                        choices=['large', 'base'])
    parser.add_argument('--use_temp', action='store_true')
    parser.add_argument('--no_plot', action='store_false')

    args = parser.parse_args()
    
    visualizer = CodebookFrequenciesVisualizer(args.dataset_path,
            args.checkpoint_path, args.arch_type, args.use_temp)
    language_dist = visualizer.get_language_distribution(args.use_temp)
    visualizer.plot_codebook_frequencies(args.output_path, language_dist)

