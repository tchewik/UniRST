import os
from glob import glob

import json
import fire
import numpy as np

from src.evaluation.eval_universal import EvalUniversal


class EvalDMRSTTransfer:
    """
    Evaluates transfer learning performance for DMRST models.

    Args:
        models_dir (str): Directory containing model checkpoints.
        nfolds (int): Number of cross-validation folds.
        corpus (str): Corpus name (e.g., 'GUM').
        cuda_device (int, optional): CUDA device ID (default is 0).

    Attributes:
        models_dir (str): Directory containing model checkpoints.
        nfolds (int): Number of cross-validation folds.
        corpus (str): Corpus name.
        cuda_device (int): CUDA device ID.

    Methods:
        evaluate(): Computes evaluation metrics for each genre and overall.

    Example:
        To evaluate transfer performance:
        ```
        python src/evaluation/eval_universal_transfer.py --models_dir saves/eng.rst.rstdt --corpus 'eng.erst.gentle' --nfolds 3 evaluate
        ```

    """

    def __init__(self, models_dir: str, nfolds: int, corpus: str, rel_idx=0, cuda_device=0):
        self.models_dir = models_dir
        self.nfolds = nfolds
        self.corpus = corpus
        self.rel_idx = rel_idx
        self.cuda_device = cuda_device

    def evaluate(self):
        """
        Computes evaluation metrics for each genre and overall.

        Returns:
            dict: Dictionary containing genre-wise and overall metrics.
        """

        all_metrics = []
        for path in sorted(glob(self.models_dir + '_4*'))[:self.nfolds]:
            print(f'{path = }')
            evaluator = EvalUniversal(path, self.corpus, relinventory_idx=self.rel_idx, cuda_device=self.cuda_device)
            # metrics = evaluator.by_genre()
            metrics = {'full': evaluator.full()}
            all_metrics.append(metrics)

        metrics_stats = dict()
        for genre in all_metrics[0].keys():
            metrics_stats[genre] = dict()
            for key in all_metrics[0][genre].keys():
                metrics_stats[genre][key] = (np.mean([metric[genre][key] for metric in all_metrics]),
                                             np.std([metric[genre][key] for metric in all_metrics]))


        with open(self.models_dir + f'evaluation_{self.corpus}.json', 'w') as f:
            json.dump(metrics_stats, f)

        return metrics_stats


if __name__ == '__main__':
    fire.Fire(EvalDMRSTTransfer)
