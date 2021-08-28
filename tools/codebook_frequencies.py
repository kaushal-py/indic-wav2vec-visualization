'''
Example usage:
# python tools/codebook_frequencies.py data/samples_1h data/models/CLSRIL-23.pt data/outputs
'''
from __future__ import unicode_literals
import os
from collections import Counter
from pathlib import Path
from joblib import dump, load

import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
import soundfile as sf
from loguru import logger
import argparse
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly; logger.info("Plotly version: {}".format(plotly.__version__))
import plotly.express as px
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, SpectralEmbedding
from indictrans import Transliterator

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
            Path('/tmp/indic_wav2vec/').mkdir(parents=True, exist_ok=True)
            temp_file = '/tmp/indic_wav2vec/lang_dist_10hr_23.npy'
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
                distribution = frequency # / frequency.max()
                language_frequencies[language] = distribution
            
            Path('/tmp/indic_wav2vec/').mkdir(parents=True, exist_ok=True)
            temp_file = '/tmp/indic_wav2vec/lang_dist_untrained.npy'
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
    
    def plot_weighted_tsne(self, output_folder: str, language_dist: dict, use_temp=False):

        Path(output_folder).mkdir(parents=True, exist_ok=True)
        embedding_path = os.path.join(output_folder, 'tsne_xlsr_small.npy')
        if use_temp:
            logger.info('Loading tsne-model..')
            codebook_vectors_2d = np.load(embedding_path)
            logger.info('Loaded precomputed distribution from {}.'.format(
                            embedding_path))
        else:
            logger.info("Generating weighted TSNE plot..")
            codebook_vectors = self.model.model.quantizer.codebook().cpu().detach()
            print(codebook_vectors.shape)
            dim_reducer = TSNE(n_components=2, verbose=1, random_state=42, n_jobs=-1)
            codebook_vectors_2d = dim_reducer.fit_transform(codebook_vectors)
            logger.info("Saving TSNE model {} ..".format(embedding_path))
            np.save(embedding_path, codebook_vectors_2d)
            logger.info("TSNE-Model Saved.")
            # codebook_vectors_2d = dim_reducer.transform(codebook_vectors)
            # return codebook_vectors_2d
        output_destination = os.path.join(output_folder, "tsne_lang_plot.png")
        # plt.figure(figsize=(25,25))
        # for i, (lang, freq) in enumerate(language_dist.items()):
        #     # if i==2:
        #     #     break
        #     alphas = freq /freq.max()
        #     print(alphas.shape, alphas.max(), alphas.mean())
        #     plt.scatter(codebook_vectors_2d[:, 0], codebook_vectors_2d[:, 1],
        #                 label=lang, alpha=alphas)
        weights = np.array(list(language_dist.values()))
        language_names = list(language_dist.keys())
        print(weights.shape, codebook_vectors_2d.shape)
        language_points = (weights @ codebook_vectors_2d) / 102400
        print(language_points.shape)
        fig, ax = plt.subplots()
        fig.figsize = (50, 50)
        # Codebook vectors
        # ax.scatter(codebook_vectors_2d[:,0], codebook_vectors_2d[:,1], alpha=0.3)
        families = self.language_to_family(language_names)
        ax.scatter(language_points[:,0], language_points[:,1], c=families)
        for i, txt in enumerate(language_names):
            ax.annotate(txt, (language_points[i,0], language_points[i,1]))
        fig.tight_layout()
        plt.show()
        plt.savefig(output_destination)
        # ,  dpi=300, bbox_inches='tight')
        # logger.debug("Rendering plot..")
        # plt.legend()
        # plt.show()
        # plt.savefig(output_destination)
          
    def plot_codebook_frequencies(self, output_folder: str, language_dist: dict):
        logger.info("Generating language codebook frequencies..")
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        #############################
        temp_file = 'data/tsne_1d_xlsr_small.npy'
        # codebook_vectors_1d = np.load(temp_file, allow_pickle=True)
        dim_reducer = PCA(n_components=1, random_state=42)
        codebook_vectors = self.model.model.quantizer.codebook().cpu().detach()
        codebook_vectors_1d = dim_reducer.fit_transform(codebook_vectors)
        logger.info('Loaded precomputed distribution from {}.'.format(
                        temp_file))
        # dim_reducer = TSNE(n_components=1, verbose=1, random_state=42)
        # codebook_vectors_1d = dim_reducer.fit_transform(codebook_vectors_2d)
        # Path('/tmp/indic_wav2vec/').mkdir(parents=True, exist_ok=True)
        # temp_file = '/tmp/indic_wav2vec/codebook_1d.npy'
        # np.save(temp_file, codebook_vectors_1d)
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
        print(codebook_vectors_1d.shape)
        codebook_vectors_1d = codebook_vectors_1d[(sum_rows != 0)]
        print(codebook_vectors_1d.shape)
        print("max:",frequencies.max(axis=0).shape)
        frequencies = frequencies / frequencies.max(axis=0)

        ### Sorting by hindi
        sorted_indices = np.argsort(-frequencies[4])
        # sorted_indices = np.argsort(-frequencies[20])
        ### Sorting by 1d projection
        # codebook_vectors_1d = np.squeeze(codebook_vectors_1d)
        # print(codebook_vectors_1d)
        # sorted_indices = np.argsort(codebook_vectors_1d)
        print(sorted_indices)
        # print(sorted_indices.shape)
        # print(sorted_indices)
        frequencies = frequencies[:,sorted_indices]

        # Create grouping
        hi_freqs = frequencies[6]
        # hi_freqs[hi_freqs <= 0.33] = 0
        # hi_freqs[hi_freqs > 0.66] = 1
        # hi_freqs[(hi_freqs > 0.33) & (hi_freqs <= 0.66)] = 0.5

        frequencies[frequencies <= 0.2] = 0
        frequencies[(frequencies > 0.2) & (frequencies <= 0.4)] = 0.2
        frequencies[(frequencies > 0.4) & (frequencies <= 0.6)] = 0.4
        frequencies[(frequencies > 0.6) & (frequencies <= 0.8)] = 0.6
        # frequencies[(frequencies > 0.8) & (frequencies <= 0.4)] = 0.2
        frequencies[frequencies > 0.8] = 0.8
        # frequencies[(frequencies > 0.33) & (frequencies <= 0.66)] = 0.5

        import matplotlib
        matplotlib.rc('ytick', labelsize=40)
        matplotlib.rc('xtick', labelsize=40)
        fig = plt.figure(figsize=(50,25))
        ax = fig.add_subplot(111)
        cax = ax.matshow(frequencies, aspect='auto', cmap='coolwarm')
        ax.set_yticks(np.arange(0, len(language_dist), 1.0))
        ax.set_yticklabels(list(langs))
        ###################

        # fig = plt.figure(figsize=(50,25))
        # ax = fig.add_subplot(111)
        # df = pd.DataFrame(frequencies.T, columns=list(langs))
        # pd.plotting.parallel_coordinates(
        #     df, 'hindi', ax=ax, # color=('#556270', '#4ECDC4', '#C7F464')
        # )

        # output_destination = os.path.join(output_folder, "parallelplot_hin_sorted_grouped.png")
        output_destination = os.path.join(output_folder, "language_codebook_matrix_grouped_en_sorted_5.png")
        plt.show()
        plt.savefig(output_destination)
    
    def _plot_affinity_matrix(self, affinity_matrix: np.ndarray,
                                output_destination: str,
                                labels_x: list = None, labels_y: list = None,
                                title_x = None, title_y = None):

        # # hindi_font = FontProperties(fname = 'KRDEV011.ttf')
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.matshow(affinity_matrix, interpolation='nearest', cmap='Blues')
        if title_x is not None:
            ax.set_xlabel(title_x)
        if title_y is not None:
            ax.set_ylabel(title_y)
        if labels_x is not None:
            ax.set_xticklabels(['']+labels_x, rotation=90)
            ax.set_xticks(np.arange(-1, len(labels_x)))
        if labels_y is not None:
            ax.set_yticklabels(['']+labels_y)
            ax.set_yticks(np.arange(-1, len(labels_y)))
        plt.show()
        plt.savefig(output_destination)
        #####################################
        # fig = px.imshow(affinity_matrix,
        #         # labels=dict(x=title_x, y=title_y, color="Productivity"),
        #         # x=labels_x,
        #         # y=labels_y
        #        )
        # fig.write_image(output_destination)

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
        # print(affinity_matrix)
        min_val = affinity_matrix.min()
        max_val = affinity_matrix.max()
        np.fill_diagonal(affinity_matrix, min_val)
        # affinity_matrix = (affinity_matrix - min_val)/(max_val - min_val)
        # print(affinity_matrix)
        output_destination = os.path.join(output_folder, "affinity_matrix_nodiag.png")
        if plot:
            self._plot_affinity_matrix(affinity_matrix, output_destination,
                    list(language_dist.keys()), list(language_dist.keys()))
            logger.info('Saved output in {}'.format(output_destination))
        return affinity_matrix, distance_matrix
    
    def cluster_and_visualize(self, language_dist, affinity_matrix, 
                                output_folder: str, num_clusters=5,
                                dim_reduce='TSNE'):
        logger.info("Performing spectral clustering with {} clusters..".format(num_clusters))

        datapoints = np.array(list(language_dist.values()))
        language_names = list(language_dist.keys())
        # Dimensionality reduction
        logger.info("Performing dimensionality reduction using {}..".format(dim_reduce))
        dimension_reducer = TSNE(n_components=2, perplexity=1,
                                learning_rate=10, verbose=1, metric='precomputed',
                                random_state=42)
        # dimension_reducer = SpectralEmbedding(
        #                 n_components=2, affinity='precomputed', n_neighbors=10)
        # dimension_reducer = PCA(n_components=2)
        pca_points = dimension_reducer.fit_transform((1-affinity_matrix))
        # pca_points = dimension_reducer.fit_transform(affinity_matrix)
        # pca_points = dimension_reducer.fit_transform((datapoints))
        print(pca_points.shape)

        for num_clusters in [5]:
            # Pefrom clustring
            clustering = SpectralClustering(n_clusters=num_clusters,
                affinity='precomputed',
                assign_labels='discretize',
                random_state=42)
            labels = clustering.fit_predict(affinity_matrix)
            
            # Plotting
            output_destination = os.path.join(output_folder, "{}_clusters_{}.png".format(num_clusters, dim_reduce))
            fig, ax = plt.subplots()
            fig.figsize = (20, 20)
            ax.scatter(pca_points[:,0], pca_points[:,1], c=labels)
            for i, txt in enumerate(language_names):
                ax.annotate(txt, (pca_points[i,0], pca_points[i,1]))
            fig.tight_layout()
            plt.show()
            plt.savefig(output_destination,  dpi=300, bbox_inches='tight')
            logger.info('Saved output in {}'.format(output_destination))
    
    def _load_phoneme_prob_matrix(self, language):
        model_path = "data/phoneme_model/CLSRIL_{}_10h_trimmed.npy".format(language)
        model = np.load(model_path)
        sums = model.sum(axis=1)
        sums[sums == 0] = 1
        model = (model.T/sums).T
        return model
    
    def _get_phoneme_list(self, language):
        lang_dict_path = "data/phoneme_model/{}_dict.txt".format(language)
        df = pd.read_csv(lang_dict_path, sep=' ', header=None)
        chars = df[0].tolist()
        chars = chars[1:]
        two2three = {
            'hi': 'hin',
            'en': 'eng',
            'te': 'tel',
            'ta': 'tam',
            'be': 'ben',
            'gu': 'guj',
            'kn': 'kan',
        }
        print(chars)
        # try:
        #     trn = Transliterator(source=two2three[language], target='eng', build_lookup=True)
        #     chars = list(map(lambda x: trn.transform(x), chars))
        # except:
        #     pass
        return chars
    
    def phoneme_tsne_visualise(self, lang_1, lang_2, output_folder):
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        temp_file = 'data/tsne_xlsr_small.npy'
        codebook_vectors_2d = np.load(temp_file, allow_pickle=True)
        logger.info('Loaded precomputed distribution from {}.'.format(
                        temp_file))
        model_1 = self._load_phoneme_prob_matrix(lang_1)
        model_2 = self._load_phoneme_prob_matrix(lang_2)
        lang_1_list = self._get_phoneme_list(lang_1)
        lang_2_list = self._get_phoneme_list(lang_2)

        output_destination = os.path.join(output_folder, "tsne_plot_phonemes_pair_{}_{}.png".format(lang_1, lang_2))
        
        # lang 1
        language_points = (model_1 @ codebook_vectors_2d) / 102400
        print(language_points.shape)
        fig, ax = plt.subplots(figsize=(10,10))
        # fig.figsize = (50, 50)
        # Codebook vectors
        # ax.scatter(codebook_vectors_2d[:,0], codebook_vectors_2d[:,1], alpha=0.3)
        ax.scatter(language_points[:,0], language_points[:,1])
        for i, txt in enumerate(lang_1_list):
            ax.annotate(txt, (language_points[i,0], language_points[i,1]))

        # lang 2
        language_points = (model_2 @ codebook_vectors_2d) / 102400
        print(language_points.shape)
        # Codebook vectors
        # ax.scatter(codebook_vectors_2d[:,0], codebook_vectors_2d[:,1], alpha=0.3)
        ax.scatter(language_points[:,0], language_points[:,1])
        for i, txt in enumerate(lang_2_list):
            ax.annotate(txt, (language_points[i,0], language_points[i,1]))
        # fig.tight_layout()
        plt.show()
        plt.savefig(output_destination, dpi=300)

    def phoneme_visualise(self, lang_1, lang_2, output_folder,
                            plot=True):
        logger.info("Generating phoneme matrix..")
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        model_1 = self._load_phoneme_prob_matrix(lang_1)
        model_2 = self._load_phoneme_prob_matrix(lang_2)
        lang_1_list = self._get_phoneme_list(lang_1)
        lang_2_list = self._get_phoneme_list(lang_2)
        print(model_1.shape, model_2.shape)
        affinity_matrix = np.zeros((len(model_1), len(model_2)))
        for i, phoneme_1 in enumerate(model_1):
            for j, phoneme_2 in enumerate(model_2):
                dist = distance.jensenshannon(phoneme_1, phoneme_2)
                # dist = wasserstein_distance(phoneme_1, phoneme_2)
                if np.isnan(dist):
                    dist = 1
                assert dist >= 0 and dist <= 1, dist
                affinity_matrix[i][j] = (1-dist)
        # print(affinity_matrix)
        # make row-stochastic matrix
        affinity_matrix = (affinity_matrix.T / affinity_matrix.sum(axis=1)).T
        output_destination = os.path.join(output_folder,
                    "phoneme_matrix_{}_{}_10h_trimmed.png".format(lang_1, lang_2))
        if plot:
            self._plot_affinity_matrix(affinity_matrix, output_destination,
                                        lang_2_list, lang_1_list, lang_2, lang_1)
            logger.info('Saved output in {}'.format(output_destination))
        return affinity_matrix

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
    
    def language_to_family(self, language_list: list):

        family_ids = {
            'indo_aryan': 0,
            'dravidian':1,
            'others':2,
        }
        language_family = {
            'assamese': 'indo_aryan',
            'bengali': 'indo_aryan',
            'gujarati': 'indo_aryan',
            'hindi': 'indo_aryan',
            'konkani': 'indo_aryan',
            'bodo': 'indo_aryan',
            'dogri': 'indo_aryan',
            'kashmiri': 'indo_aryan',
            'maithili': 'indo_aryan',
            'manipuri': 'indo_aryan',
            'marathi': 'indo_aryan',
            'nepali': 'indo_aryan',
            'odia': 'indo_aryan',
            'punjabi': 'indo_aryan',
            'sanskrit': 'indo_aryan',
            'sindhi': 'indo_aryan',
            'urdu': 'indo_aryan',
            'kannada': 'dravidian',
            'malayalam': 'dravidian',
            'tamil': 'dravidian',
            'telugu': 'dravidian',
            'english': 'others',
            'santali': 'others',
        }
        family_list = []
        for language in language_list:
            fam_id = family_ids[language_family[language]]
            family_list.append(fam_id)
        return family_list

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
    visualizer.plot_codebook_frequencies(args.output_path, language_dist)
    # visualizer.plot_weighted_tsne(args.output_path, language_dist)
    # affinity_matrix, _ = visualizer.get_affinity_matrix(
    #             args.output_path,
    #             distance_metric=args.distance_metric,
    #             language_dist=language_dist,
    #             plot=args.no_plot)
    # visualizer.cluster_and_visualize(
    #             language_dist,
    #             affinity_matrix,
    #             args.output_path,
    #             dim_reduce=args.dim_reduce,
    #             num_clusters=args.clusters)
    # visualizer.phoneme_visualise(
    #     "te",
    #     "kn",
    #     args.output_path,
    #     )
    # visualizer.phoneme_tsne_visualise(
    #     "te",
    #     "kn",
    #     args.output_path,
    #     )

    # visualizer.get_language_codebook_usage(args.output_path)

