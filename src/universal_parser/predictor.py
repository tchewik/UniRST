import ast
import json
import os

import razdel
import torch
from isanlp.annotation import Token
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from .data_manager import DataManager
from .du_converter import DUConverter
from .src.parser.data import Data
from .src.parser.parsing_net import ParsingNet
from .src.parser.parsing_net_bottom_up import ParsingNetBottomUp
from .trainer import Trainer


def str2bool(value):
    if type(value) == bool:
        return value

    if type(value) == str:
        return value.lower() == 'true'


class Predictor:
    def __init__(self, model_dir: str, cuda_device: int, relinventory_idx: int = 0):

        self.model_path = os.path.join(model_dir, 'best_weights.pt')
        self.config_path = os.path.join(model_dir, 'config.json')
        self.config = json.load(open(self.config_path))
        self.relinventory_idx = relinventory_idx

        self.data_managers = []
        for corpus_name in ast.literal_eval(self.config['data']['corpora']):
            data_manager_path = f'data/dms/data_manager_{corpus_name.lower()}.pickle'
            dp = DataManager(corpus=corpus_name).from_pickle(data_manager_path)
            self.data_managers.append(dp)

        self._cuda_device = torch.device('cpu' if cuda_device == -1 else f'cuda:{cuda_device}')

        self._load_model()

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['transformer']['model_name'], use_fast=True)
        transformer = AutoModel.from_pretrained(self.config['model']['transformer']['model_name']).to(self._cuda_device)

        self.tokenizer.add_tokens(['<P>'])
        transformer.resize_token_embeddings(len(self.tokenizer))

        rel_tables = [dm.relation_table for dm in self.data_managers]
        use_union = self.config['model'].get('use_union_relations', False) and len(rel_tables) > 1

        if use_union:
            union_table = []
            label2id = {}
            for table in rel_tables:
                for lbl in table:
                    if lbl not in label2id:
                        label2id[lbl] = len(union_table)
                        union_table.append(lbl)

            dataset_masks = []
            label_maps = []
            for table in rel_tables:
                mask = [False] * len(union_table)
                mapping_tbl = []
                for lbl in table:
                    uid = label2id[lbl]
                    mask[uid] = True
                    mapping_tbl.append(uid)
                dataset_masks.append(mask)
                label_maps.append(mapping_tbl)

            self.label_maps = label_maps

            model_config = {
                'relation_tables': rel_tables,
                'relation_vocab': union_table,
                'dataset_masks': dataset_masks,
                'classes_numbers': [len(union_table)],
                'dataset2classifier': list(range(len(rel_tables))),
            }
        else:
            unique_tables = []
            mapping = []
            for table in rel_tables:
                for idx, ut in enumerate(unique_tables):
                    if table == ut:
                        mapping.append(idx)
                        break
                else:
                    mapping.append(len(unique_tables))
                    unique_tables.append(table)

            self.label_maps = None

            model_config = {
                'relation_tables': unique_tables,
                'classes_numbers': [len(t) for t in unique_tables],
                'dataset2classifier': mapping,
            }

        model_config.update({
            'transformer': transformer,
            'emb_dim': int(self.config['model']['transformer']['emb_size']),
            'cuda_device': self._cuda_device
        })

        model_config.update(self._get_model_configs())
        model_cls = ParsingNet if self.config['model'].get('parser_type',
                                                           'top-down') == 'top-down' else ParsingNetBottomUp
        self.model = model_cls(**model_config).to(self._cuda_device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self._cuda_device))
        self.model.eval()

    def _get_model_configs(self):
        config = {}

        if 'normalize' in self.config['model']['transformer']:
            config['normalize_embeddings'] = self.config['model']['transformer'].get('normalize')

        if 'hidden_size' in self.config['model']:
            hidden_size = int(self.config['model'].get('hidden_size'))
            config['hidden_size'] = hidden_size
            config['decoder_input_size'] = hidden_size
            config['classifier_input_size'] = hidden_size
            config['classifier_hidden_size'] = hidden_size

        if 'type' in self.config['model']['segmenter']:
            config['segmenter_type'] = self.config['model']['segmenter'].get('type')

        if 'hidden_dim' in self.config['model']['segmenter']:
            config['segmenter_hidden_dim'] = int(self.config['model']['segmenter'].get('hidden_dim'))

        if 'lstm_num_layers' in self.config['model']['segmenter']:
            config['segmenter_lstm_num_layers'] = self.config['model']['segmenter'].get('lstm_num_layers')

        if 'lstm_dropout' in self.config['model']['segmenter']:
            config['segmenter_lstm_dropout'] = self.config['model']['segmenter'].get('lstm_dropout')

        if 'lstm_bidirectional' in self.config['model']['segmenter']:
            config['segmenter_lstm_bidirectional'] = str2bool(
                self.config['model']['segmenter'].get('lstm_bidirectional'))

        if 'use_crf' in self.config['model']['segmenter']:
            config['segmenter_use_crf'] = str2bool(self.config['model']['segmenter'].get('use_crf'))

        if 'use_log_crf' in self.config['model']['segmenter']:
            config['segmenter_use_log_crf'] = str2bool(self.config['model']['segmenter'].get('use_log_crf'))

        if 'if_edu_start_loss' in self.config['model']['segmenter']:
            config['segmenter_if_edu_start_loss'] = str2bool(self.config['model']['segmenter'].get('if_edu_start_loss'))

        if 'edu_encoding_kind' in self.config['model']:
            config['edu_encoding_kind'] = self.config['model'].get('edu_encoding_kind')

        if 'du_encoding_kind' in self.config['model']:
            config['du_encoding_kind'] = self.config['model'].get('du_encoding_kind')

        if 'rel_classification_kind' in self.config['model']:
            config['rel_classification_kind'] = self.config['model'].get('rel_classification_kind')

        if 'token_bilstm_hidden' in self.config['model']:
            config['token_bilstm_hidden'] = int(self.config['model'].get('token_bilstm_hidden'))

        return config

    def tokenize(self, data):
        """ Takes data with word level tokenization, run current transformer tokenizer and recount EDU boundaries."""

        def get_offset_mappings(input_ids):
            subwords_str = self.tokenizer.convert_ids_to_tokens(input_ids)

            start, end = 0, 0
            result = []
            for subword in subwords_str:
                if subword.startswith('▁'):
                    if subword != '▁':
                        start += 1

                if subword == '<P>' and start > 0:
                    start += 1
                    end += 1

                end += len(subword)
                result.append((start, end))
                start = end
            return result

        # (word_start_char, word_end_char+1) for each token
        word_offsets = []
        for document in data.input_sentences:
            doc_word_offsets = []
            cur_char = 0
            for word in document:
                doc_word_offsets.append((cur_char, cur_char + len(word)))
                cur_char += len(word) + 1
            word_offsets.append(doc_word_offsets)

        texts = [' '.join(line).strip() for line in data.input_sentences]
        tokens = self.tokenizer(texts, add_special_tokens=False, return_offsets_mapping=True)
        tokens['entity_ids'] = None
        tokens['entity_position_ids'] = None

        # recount edu_breaks for subwords
        subword_edu_breaks = []
        for doc_word_offsets, doc_subword_offsets, edu_breaks in zip(
                word_offsets, tokens['offset_mapping'], data.edu_breaks):
            subword_edu_breaks.append(Trainer.recount_spans(doc_word_offsets, doc_subword_offsets, edu_breaks))

        if self.label_maps:
            mapping = self.label_maps[self.relinventory_idx]
            try:
                remapped = [[mapping[idx] for idx in doc] for doc in data.relation_label]
            except IndexError:
                if hasattr(data, 'relation_table'):
                    label2id = {lbl.lower(): i for i, lbl in enumerate(self.model.relation_vocab)}
                    mapping = []
                    for lbl in data.relation_table:
                        if lbl.lower() not in label2id:
                            raise ValueError(f"Label '{lbl}' not found in model relation inventory")
                        mapping.append(label2id[lbl.lower()])
                    remapped = [[mapping[idx] for idx in doc] for doc in data.relation_label]
                else:
                    raise
        else:
            remapped = data.relation_label

        return Data(
            input_sentences=tokens['input_ids'],
            entity_ids=tokens['entity_ids'],
            entity_position_ids=tokens['entity_position_ids'],
            sent_breaks=None,
            edu_breaks=subword_edu_breaks,
            decoder_input=data.decoder_input,
            relation_label=remapped,
            parsing_breaks=data.parsing_breaks,
            golden_metric=data.golden_metric,
            parents_index=data.parents_index,
            sibling=data.sibling,
            dataset_index=[self.relinventory_idx for _ in range(len(data.input_sentences))]
        )

    @staticmethod
    def divide_chunks(_list, n):
        if _list:
            for i in range(0, len(_list), n):
                yield _list[i:min(i + n, len(_list))]
        else:
            yield _list

    def get_batches(self, data: Data, size: int):
        """ Splits a batch into multiple smaller with given size. """

        if len(data.input_sentences) < size:
            return [data]

        _input_sentences = list(self.divide_chunks(data.input_sentences, size))
        _edu_breaks = list(self.divide_chunks(data.edu_breaks, size))
        _decoder_input = list(self.divide_chunks(data.decoder_input, size))
        _relation_label = list(self.divide_chunks(data.relation_label, size))
        _parsing_breaks = list(self.divide_chunks(data.parsing_breaks, size))
        _golden_metric = list(self.divide_chunks(data.golden_metric, size))
        _dataset_index = list(self.divide_chunks(data.dataset_index, size))

        batches = []
        for (input_sentences, edu_breaks, decoder_input,
             relation_label, parsing_breaks, golden_metric, dataset_index
             ) in tqdm(zip(_input_sentences, _edu_breaks, _decoder_input,
                           _relation_label, _parsing_breaks, _golden_metric, _dataset_index),
                       total=len(_input_sentences)):
            batches.append(
                Data(
                    input_sentences=input_sentences,
                    entity_ids=None,
                    entity_position_ids=None,
                    sent_breaks=None,
                    edu_breaks=edu_breaks,
                    decoder_input=decoder_input,
                    relation_label=relation_label,
                    parsing_breaks=parsing_breaks,
                    golden_metric=golden_metric,
                    parents_index=None,
                    sibling=None,
                    dataset_index=dataset_index
                )
            )

        return batches

    def _collect_tokens(self, tree):
        tokens = []
        begin = 0
        for token in tree.text.split(' '):
            tokens.append(Token(text=token, begin=begin, end=begin + len(token)))
            begin += len(token) + 1

        return tokens

    def parse_rst(self, text: str):
        """
        Parses the given text to generate a tree of rhetorical structure.

        Args:
            text (str): The input text to be parsed.

        Returns:
            dict: Tokens and a tree representing the rhetorical structure based on the input text.
        """

        # Preprocess the text
        _text = text.replace('-', ' - ').replace('—', ' — ').replace('  ', ' ')
        _text = _text.replace('...', '…').replace('_', ' ')

        # Prepare the input data
        tokenized_text = [token.text for token in razdel.tokenize(_text)]
        data = {
            'input_sentences': [tokenized_text],
            'edu_breaks': [[]],
            'decoder_input': [[]],
            'relation_label': [[]],
            'parsing_breaks': [[]],
            'golden_metric': [[]],
        }

        if len(tokenized_text) < 3:
            tree = DUConverter.dummy_tree(tokenized_text)

            return {
                'tokens': self._collect_tokens(tree),
                'rst': [tree]
            }

        # Initialize predictions dictionary
        input_data = Data(**data)

        predictions = {
            'tokens': [],
            'spans': [],
            'edu_breaks': [],
            'true_spans': [],
            'true_edu_breaks': []
        }

        # Tokenize the input for the transformer
        batch = self.tokenize(input_data)

        # Perform forward pass
        with torch.no_grad():
            loss_tree_batch, loss_label_batch, \
                span_batch, label_tuple_batch, predict_edu_breaks = self.model.testing_loss(
                batch.input_sentences, batch.sent_breaks, batch.entity_ids, batch.entity_position_ids,
                batch.edu_breaks, batch.relation_label, batch.parsing_breaks,
                generate_tree=True, use_pred_segmentation=True, dataset_index=batch.dataset_index)

        # Update predictions dictionary
        predictions['tokens'] += [self.tokenizer.convert_ids_to_tokens(text) for text in
                                  batch.input_sentences]
        predictions['spans'] += span_batch
        predictions['edu_breaks'] += predict_edu_breaks
        predictions['true_spans'] += batch.golden_metric
        predictions['true_edu_breaks'] += batch.edu_breaks

        # Convert predictions to a tree structure
        duc = DUConverter(predictions, tokenization_type='default')
        tree = duc.collect(tokens=data['input_sentences'])[0]

        return {
            'tokens': self._collect_tokens(tree),
            'rst': [tree]
        }
