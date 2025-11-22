import os
import sys
import faiss
import argparse

import numpy as np

from multiprocessing import cpu_count
from sklearn.cluster import MiniBatchKMeans

sys.path.append(os.getcwd())

from main.app.variables import logger, translations, configs

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--create_index", action='store_true')
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--rvc_version", type=str, default="v2")
    parser.add_argument("--index_algorithm", type=str, default="Auto")

    return parser.parse_args()

def main():
    args = parse_arguments()
    exp_dir = os.path.join(configs["logs_path"], args.model_name)
    version, index_algorithm = args.rvc_version, args.index_algorithm

    log_data = {translations['modelname']: args.model_name, translations['model_path']: exp_dir, translations['training_version']: version, translations['index_algorithm_info']: index_algorithm}
    for key, value in log_data.items():
        logger.debug(f"{key}: {value}")

    try:
        npys = []
        feature_dir = os.path.join(exp_dir, f"{version}_extracted")
        model_name = os.path.basename(exp_dir)

        for name in sorted(os.listdir(feature_dir)):
            npys.append(np.load(os.path.join(feature_dir, name)))

        big_npy = np.concatenate(npys, axis=0)
        big_npy_idx = np.arange(big_npy.shape[0])
        np.random.shuffle(big_npy_idx)
        big_npy = big_npy[big_npy_idx]

        if big_npy.shape[0] > 2e5 and (index_algorithm == "Auto" or index_algorithm == "KMeans"): big_npy = (MiniBatchKMeans(n_clusters=10000, verbose=True, batch_size=256 * cpu_count(), compute_labels=False, init="random").fit(big_npy).cluster_centers_)
        np.save(os.path.join(exp_dir, "total_fea.npy"), big_npy)

        n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
        index_trained = faiss.index_factory(256 if version == "v1" else 768, f"IVF{n_ivf},Flat")
        index_ivf_trained = faiss.extract_index_ivf(index_trained)
        index_ivf_trained.nprobe = 1
        index_trained.train(big_npy)
        faiss.write_index(index_trained, os.path.join(exp_dir, f"trained_IVF{n_ivf}_Flat_nprobe_{index_ivf_trained.nprobe}_{model_name}_{version}.index"))

        index_added = faiss.index_factory(256 if version == "v1" else 768, f"IVF{n_ivf},Flat")
        index_ivf_added = faiss.extract_index_ivf(index_added)
        index_ivf_added.nprobe = 1
        index_added.train(big_npy)
        batch_size_add = 8192
    
        for i in range(0, big_npy.shape[0], batch_size_add):
            index_added.add(big_npy[i : i + batch_size_add])

        index_filepath_added = os.path.join(exp_dir, f"added_IVF{n_ivf}_Flat_nprobe_{index_ivf_added.nprobe}_{model_name}_{version}.index")
        faiss.write_index(index_added, index_filepath_added)
        logger.info(f"{translations['save_index']} '{index_filepath_added}'")
    except Exception as e:
        logger.error(f"{translations['create_index_error']}: {e}")
        import traceback
        logger.debug(traceback.format_exc())

if __name__ == "__main__": main()