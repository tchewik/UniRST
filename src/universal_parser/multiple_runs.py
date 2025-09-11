"""
Script for multiple runs of experiments.

    # Train
    python universal_parser/multiple_runs.py --corpora '["corp.1","corp.2"]' --model_type "$TYPE" train
    # Evaluation
    python universal_parser/multiple_runs.py --corpora '["corp.1","corp.2"]' --model_type "$TYPE" evaluate
"""

import json
import os
import subprocess
import sys
from glob import glob
import queue
import time

import fire
import nltk

nltk.download('punkt_tab')


class MultipleRunnerGeneral:
    def __init__(self,
                 corpora: list,
                 data_aug: bool = False,
                 model_type: str = '+tony+trainable_edus',
                 transformer_name: str = 'xlm-roberta-large',
                 masked_union: bool = False,
                 segmenter_separated: bool = True,
                 emb_size: int = 1024,
                 freeze_first_n: int = 0,
                 window_size: int = 450,
                 window_padding: int = 30,
                 cuda_device: int = 0,
                 cuda_devices: list | None = None,
                 resume_training: bool = False,
                 n_runs: int = 3,
                 save_path: str = 'saves/',
                 ):
        """
        :param corpus: (str)  - 'GUM' or 'RST-DT'
        :param lang: (str)  - 'en' or 'ru'
        :param model_type: (str)  - one of {'default', '+tony', '+tony+trainable_edus', '+tony+trainable_edus+bimpm'}
        :param transformer_name: (str)  - model name or path to the pretrained LM
        :param emb_size: (int)  - LM encodings size
        :param cuda_device: (int)  - number of cuda device
        :param cuda_devices: (list[int], optional) - ids of several cuda devices for parallel runs
        :param resume_training: (bool)  - whether to rewrite previous saves
        """
        self.corpora = corpora
        self.data_aug = data_aug
        self.model_type = model_type
        self.masked_union = masked_union
        self.segmenter_separated = segmenter_separated
        self.transformer_name = transformer_name
        self.emb_size = emb_size
        self.freeze_first_n = freeze_first_n
        self.window_size = window_size
        self.window_padding = window_padding
        self.cuda_device = cuda_device
        self.cuda_devices = cuda_devices
        self.resume_training = resume_training
        self.n_runs = n_runs
        self.save_path = save_path

    def _general_parameters(self):
        overrides = {
            'corpora': self.corpora,
            'data_augmentation': str(self.data_aug).lower(),
            'second_lang_fold': 0,
            'second_lang_fraction': 0,
            'transformer_name': self.transformer_name,  # LM name
            'emb_size': self.emb_size,  # LM embedding size
            'freeze_first_n': self.freeze_first_n,  # LM fine-tuning configuration
            'window_size': self.window_size,
            'window_padding': self.window_padding,
            'transformer_normalize': 'true',
            'hidden_size': 512,
            'use_crf': 'true',  # ToNy (LSTM-CRF)
            'use_log_crf': 'false',  # [Optional] Logits restriction for ToNy
            'token_bilstm_hidden': 0,  # BiMPM representation hidden size
            'use_union_relations': str(self.masked_union).lower(),
            'batch_size': 2,
            'dwa_bs': 12,  # Batch size for DWA computation
            'grad_clipping_value': 10.0,
            'combine_batches': 'false',  # [Optional] Combine batches w/smallest trees (for normalization when bs=1)
            'lr': 0.0001,
            'cuda_device': self.cuda_device,
            'save_path': self.save_path,
            'epochs': 30,
            'patience': 5,
        }

        # Default parameters
        overrides.update({
            'segmenter_type': 'linear',
            'segmenter_hidden_dim': overrides['hidden_size'],
            'segmenter_dropout': 0.4,
            'lstm_bidirectional': 'true',
            'if_edu_start_loss': 'true',
            'edu_encoding_kind': 'avg',
            'du_encoding_kind': 'avg',
            'rel_classification_kind': 'default',
            'use_discriminator': 'false',
            'discriminator_warmup': 0,
            'segmenter_separated': str(self.segmenter_separated).lower(),
        })

        if self.corpora[0] == 'CONCAT' or len(self.corpora) > 10:
            overrides['batch_size'] = 4
            overrides['dwa_bs'] = 32
            overrides['patience'] = 3

        if self.corpora[0] == 'CONCAT' and self.masked_union:
            overrides['use_union_relations'] = 'true'

        if self.model_type != 'default':
            types = self.model_type.split('+')

            if 'tony' in types:
                overrides['segmenter_type'] = 'tony'
                overrides['if_edu_start_loss'] = 'false'
                overrides['segmenter_hidden_dim'] = 200

            if 'no_crf' in types:
                overrides['use_crf'] = 'false'

            if 'trainable_edus' in types:
                overrides['edu_encoding_kind'] = 'trainable'

            if 'gru_edus' in types:
                overrides['edu_encoding_kind'] = 'gru'

            if 'bigru_edus' in types:
                overrides['edu_encoding_kind'] = 'bigru'

            if 'bilstm_edus' in types:
                overrides['edu_encoding_kind'] = 'bilstm'

            if 'trainable_dus' in types:
                overrides['du_encoding_kind'] = 'trainable'

            if 'bimpm' in types:
                overrides['rel_classification_kind'] = 'with_bimpm'

            if 'al' in types:
                overrides['use_discriminator'] = 'true'
                overrides['discriminator_warmup'] = 3

        if self.corpora in (['spa.rst.sctb'], ['zho.rst.sctb'], ['ces.rst.crdt']):
            overrides['hidden_size'] = 256

        if self.corpora == ['rus.rst.rrt']:
            overrides['batch_size'] = 6
            overrides['dwa_bs'] = 24
            overrides['hidden_size'] = 768

        return overrides

    def _get_variants(self):
        return range(40, 40 + self.n_runs)  # There is a fixed split, we just change the nn random seed

    def train(self):
        general_parameters = self._general_parameters()
        devices = self.cuda_devices or [self.cuda_device]

        device_queue = queue.Queue()
        for dev in devices:
            device_queue.put(dev)
        active_processes = []

        for run in self._get_variants():
            general_parameters['foldnum'] = 0
            general_parameters['seed'] = run

            aug_param = "_aug" if self.data_aug else ""
            segsep_param = "_segnosep" if not self.segmenter_separated else ""
            masked_union_param = "_MU" if self.masked_union else ""
            if len(self.corpora) > 10:
                general_parameters['run_name'] = f'ALL{masked_union_param}{aug_param}{segsep_param}_{run}'
            else:
                general_parameters['run_name'] = f'{"+".join(self.corpora)}{masked_union_param}{aug_param}{segsep_param}_{self.model_type}_{run}'

            while device_queue.empty():
                for proc, dev in active_processes[:]:
                    if proc.poll() is not None:
                        active_processes.remove((proc, dev))
                        device_queue.put(dev)
                if device_queue.empty():
                    time.sleep(1)

            device = device_queue.get()
            general_parameters['cuda_device'] = device
            for key, value in general_parameters.items():
                general_parameters[key] = str(value)

            if self.resume_training:
                if os.path.isfile(os.path.join('saves', general_parameters['run_name'], 'best_metrics.json')):
                    device_queue.put(device)
                    continue

            p = subprocess.Popen(
                ['python', 'src/universal_parser/trainer.py',
                 'configs/general_uni_config.jsonnet', json.dumps(general_parameters)],
                stdout=sys.stdout, stderr=sys.stderr
            )
            active_processes.append((p, device))

        for proc, _ in active_processes:
            proc.wait()

    def evaluate(self):
        results = {
            'e2e_test_f1_full': [],
            'e2e_test_f1_nuc': [],
            'e2e_test_f1_rel': [],
            'e2e_test_f1_seg': [],
            'e2e_test_f1_span': [],
            'gs_test_f1_full': [],
            'gs_test_f1_nuc': [],
            'gs_test_f1_rel': [],
            'gs_test_f1_span': []
        }
        for run in self._get_variants():
            aug_param = "_aug" if self.data_aug else ""
            segsep_param = "_segnosep" if not self.segmenter_separated else ""
            if len(self.corpora) > 10:
                run_name = f'ALL{aug_param}{segsep_param}_{run}'
            else:
                masked_union_param = "_MU" if self.masked_union else ""
                run_name = f'{"+".join(self.corpora)}{masked_union_param}{aug_param}{segsep_param}_{self.model_type}_{run}'

            run_path = os.path.join(self.save_path, run_name)
            try:
                all_metrics = glob(os.path.join(run_path, 'metrics_epoch_*.json'))
                best_epoch = sorted([int(os.path.basename(metrics)[14:-5]) for metrics in all_metrics])[-1]
                best_dev_metrics = json.load(open(os.path.join(run_path, f'metrics_epoch_{best_epoch}.json')))
                for key in results:
                    results[key].append(best_dev_metrics[key])
            except:
                print(f'Run {run} is missing.')

        aug_param = "_aug" if self.data_aug else ""
        segsep_param = "_segnosep" if not self.segmenter_separated else ""
        masked_union_param = "_MU" if self.masked_union else ""
        if len(self.corpora) > 10:
            filename = f'ALL{masked_union_param}{aug_param}{segsep_param}_{run}'
        else:
            run_name = f'{"+".join(self.corpora)}{masked_union_param}{aug_param}{segsep_param}_{self.model_type}_{run}'

            filename = run_name + '_all_res.json'

        with open(filename, 'w') as f:
            json.dump(results, f)


if __name__ == '__main__':
    fire.Fire(MultipleRunnerGeneral)
