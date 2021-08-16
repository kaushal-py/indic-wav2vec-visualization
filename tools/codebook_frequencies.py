'''
Example usage:
# python tools/codebook_frequencies.py data/samples_1h data/models/CLSRIL-23.pt data/outputs
'''
import sys
import os
from collections import Counter
from pathlib import Path
from matplotlib import use

import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
import soundfile as sf
from loguru import logger
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import inference

class CodebookFrequenciesVisualizer:

    def __init__(self, dataset_path: str,
                checkpoint_path: str, arch_type: str='large', use_temp=False):
        self.dataset = self._prepare_dataset(dataset_path)
        if use_temp:
            logger.info("Skipped Model Loading step..")
        else:
            self.model = inference.Wav2VecInferenceModule(
                checkpoint_path, arch_type=arch_type)
    
    def get_language_distribution(self, use_temp=False):
        
        if use_temp:
            Path('/tmp/indic_wav2vec/').mkdir(parents=True, exist_ok=True)
            temp_file = '/tmp/indic_wav2vec/lang_dist.npy'
            language_frequencies = np.load(temp_file, allow_pickle=True).item()
            logger.info('Loaded precomputed distribution from {}.'.format(
                            temp_file))
            return language_frequencies
        else:
            language_frequencies = {}
            pbar = tqdm(self.dataset.items())
            for (language, files) in pbar:
                pbar.set_description("Processing {}..".format(language))
                codebook_counter = Counter()
                for file in files:
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
                distribution = frequency / frequency.sum()
                language_frequencies[language] = distribution
            
            Path('/tmp/indic_wav2vec/').mkdir(parents=True, exist_ok=True)
            temp_file = '/tmp/indic_wav2vec/lang_dist.npy'
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
    
    def _plot_affinity_matrix(self, affinity_matrix: np.ndarray, labels: list,
                                output_destination: str):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.matshow(affinity_matrix, interpolation='nearest', cmap='Blues')
        ax.set_xticklabels(['']+labels, rotation=90)
        ax.set_yticklabels(['']+labels)
        ax.set_xticks(np.arange(-1, len(labels)))
        ax.set_yticks(np.arange(-1, len(labels)))
        plt.show()
        plt.savefig(output_destination)

    
    def get_affinity_matrix(self, output_folder: str, language_dist: dict,
                                distance_metric: str = 'jensonshannon',
                                plot=True):

        logger.info("Generating affinity matrix..")
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        affinity_matrix = np.zeros((len(language_dist), len(language_dist)))
        distance_matrix = np.zeros((len(language_dist), len(language_dist)))
        for i, lang_a in enumerate(language_dist.values()):
            for j, lang_b in enumerate(language_dist.values()):
                if distance_metric == 'jensonshannon':
                    dist = distance.jensenshannon(lang_a, lang_b)
                elif distance_metric == 'wasserstein':
                    dist = wasserstein_distance(lang_a, lang_b)
                assert dist >= 0 and dist <= 1, dist
                affinity_matrix[i][j] = (1-dist)
                distance_matrix[i][j] = dist
        output_destination = os.path.join(output_folder, "affinity_matrix.png")
        if plot:
            self._plot_affinity_matrix(affinity_matrix, list(language_dist.keys()),
                                        output_destination)
            logger.info('Saved output in {}'.format(output_destination))
        return affinity_matrix, distance_matrix
    
    def cluster_and_visualize(self, language_dist, affinity_matrix, 
                                output_folder: str, num_clusters=5,
                                dim_reduce='TSNE'):
        logger.info("Performing spectral clustering with {} clusters..".format(num_clusters))
        # Pefrom clustring
        clustering = SpectralClustering(n_clusters=num_clusters,
            affinity='precomputed',
            assign_labels='discretize',
            random_state=0)
        labels = clustering.fit_predict(affinity_matrix)
        datapoints = np.array(list(language_dist.values()))
        language_names = list(language_dist.keys())
        # Dimensionality reduction
        logger.info("Performing dimensionality reduction using {}..".format(dim_reduce))
        dimension_reducer = getattr(
            sys.modules[__name__], dim_reduce)(n_components=2, verbose=1)
        pca_points = dimension_reducer.fit_transform(datapoints)
        print(pca_points.shape)
        output_destination = os.path.join(output_folder, "{}_clusters_{}.png".format(num_clusters, dim_reduce))
        # Plotting
        fig, ax = plt.subplots()
        fig.figsize = (20, 20)
        ax.scatter(pca_points[:,0], pca_points[:,1], c=labels)
        for i, txt in enumerate(language_names):
            ax.annotate(txt, (pca_points[i,0], pca_points[i,1]))
        fig.tight_layout()
        plt.show()
        plt.savefig(output_destination,  dpi=300, bbox_inches='tight')
        logger.info('Saved output in {}'.format(output_destination))

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
    parser.add_argument('--clusters', type=int, default=5)
    parser.add_argument('--distance_metric', type=str, default='jensonshannon',
                        choices=['jensonshannon', 'wasserstein'])
    parser.add_argument('--arch_type', type=str, default='large',
                        choices=['large', 'small'])
    parser.add_argument('--dim_reduce', type=str, default='TSNE',
                        choices=['TSNE', 'PCA'])
    parser.add_argument('--use_temp', action='store_true')
    parser.add_argument('--no_plot', action='store_false')
    args = parser.parse_args()
    
    visualizer = CodebookFrequenciesVisualizer(args.dataset_path,
            args.checkpoint_path, args.arch_type, args.use_temp)
    language_dist = visualizer.get_language_distribution(args.use_temp)
    affinity_matrix, _ = visualizer.get_affinity_matrix(
                args.output_path,
                distance_metric=args.distance_metric,
                language_dist=language_dist,
                plot=args.no_plot)
    visualizer.cluster_and_visualize(
                language_dist,
                affinity_matrix,
                args.output_path,
                dim_reduce=args.dim_reduce,
                num_clusters=args.clusters)
    # visualizer.get_language_codebook_usage(args.output_path)

