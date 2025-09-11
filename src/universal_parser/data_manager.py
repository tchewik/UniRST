import glob
import os
import pickle
import random
from pathlib import Path

import fire
import spacy
from spacy.tokens import Doc
from tqdm import tqdm

from src.universal_parser.src.corpus import relation_set
from src.universal_parser.src.corpus.binary_tree import BinaryTree
from src.universal_parser.src.corpus.data import Rs3Document
from src.universal_parser.src.parser import reltables
from src.universal_parser.src.parser.data import Data

random.seed(42)


class ParserInput:
    def __init__(self):
        self.sentences = []
        self.edu_breaks = []
        self.label_for_metrics_list = []
        self.label_for_metrics = ''
        self.parsing_index = []
        self.relation = []
        self.decoder_inputs = []
        self.parents = []
        self.siblings = []
        self.sentence_span = []


class DataManager:
    data_dir = 'data'
    prep_dir = 'data/prepared'

    def __init__(self, corpus: str, aug: bool = False):
        """Corpus reader for predefined train/dev/test splits."""
        self.rs3_dir = os.path.join(self.data_dir, 'rs3')
        available_corpora = [os.path.basename(dir) for dir in glob.glob(os.path.join(self.rs3_dir, '*'))]
        available_corpora.append('CONCAT')
        print(f"{available_corpora = }")
        print(f'{corpus = }')

        assert corpus in available_corpora
        self.aug = aug
        self.corpus_name = corpus
        self.input_path = os.path.join(self.rs3_dir, self.corpus_name)
        self.output_path = Path(os.path.join(self.prep_dir, self.corpus_name))
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.corpus = {'train': [], 'dev': [], 'test': []}
        self.langs = []
        self.relation_fixer = dict()  # In case of relation_nuclearity mislabelings/rare classes

        if self.corpus_name == 'CONCAT':
            self._init_concat_corpus()

        elif self.corpus_name == 'ces.rst.crdt':
            self._init_crdt_corpus()

        elif self.corpus_name == 'deu.rst.pcc':
            self._init_pcc_corpus()

        elif self.corpus_name in ('eng.rst.gum', 'eng.erst.gum',
                                  'eng.rst.gentle', 'eng.erst.gentle',
                                  'rus.rst.rrg', 'zho.rst.gcdt'):
            self._init_gum_corpus()

        elif self.corpus_name == 'eng.rst.rstdt':
            self._init_rstdt_corpus()

        elif self.corpus_name in ('eng.rst.sts', 'eng.rst.oll'):
            self._init_sts_corpus()

        elif self.corpus_name == 'eng.rst.umuc':
            self._init_umuc_corpus()

        elif self.corpus_name == 'eus.rst.ert':
            self._init_ert_corpus()

        elif self.corpus_name == 'fas.rst.prstc':
            self._init_prstc_corpus()

        elif self.corpus_name == 'fra.sdrt.annodis':
            self._init_annodis_corpus()

        elif self.corpus_name == 'nld.rst.nldt':
            self._init_nldt_corpus()

        elif self.corpus_name == 'por.rst.cstn':
            self._init_cstn_corpus()

        elif self.corpus_name == 'spa.rst.rststb':
            self._init_rststb_corpus()

        elif self.corpus_name == 'rus.rst.rrt':
            self._init_rurstb_corpus()

        elif self.corpus_name in ('spa.rst.sctb', 'zho.rst.sctb'):
            self._init_sctb_corpus()

        # Initialize a simple spaCy pipeline for sentence segmentation
        self._nlp = spacy.blank("xx")
        self._nlp.add_pipe("sentencizer")

    def _init_annodis_corpus(self):
        self.rel2class = relation_set.annodis_labels
        self.relation_table = reltables.RelationTableAnnodis
        self.relation_dic = {word.lower(): i for i, word in enumerate(self.relation_table)}
        self.relation_fixer = {
            'conjunction_ns': 'conjunction_nn',
            'sequence_ns': 'sequence_nn',
            'disjunction_ns': 'disjunction_nn',
            'disjunction_sn': 'disjunction_nn',
            'contrast_ns': 'contrast_nn',
            'e-elaboration_sn': 'elaboration_ns',  # 5 times
            'elaboration_sn': 'elaboration_ns',  # 2 times
            'contrast_sn': 'contrast_nn',  # 4 times
            'result_sn': 'result_ns',  # 1 time
            'flashback_sn': 'flashback_ns',  # 1 time
            'frame_sn': 'frame_ns',  # x6
            'explanation_sn': 'background_sn',  # x6
            'frame_sn': 'frame_ns',  # x6
            'topic-comment_sn': 'topic-comment_ns',  # x6,
            'condition_sn': 'condition_ns',  # x7
            'purpose_sn': 'purpose_ns'  # x7
        }

    def _init_cstn_corpus(self):
        self.relation_table = reltables.RelationTableCSTN
        self.relation_dic = {word.lower(): i for i, word in enumerate(self.relation_table)}
        self.rel2class = relation_set.cstn_rel2class
        self.relation_fixer = {
            'parenthetical_sn': 'elaboration_sn',
            'solutionhood_sn': 'background_sn',  # x2
            'evidence_sn': 'explanation_sn',  # x2
            'antithesis_sn': 'contrast_nn',  # x5
            'antithesis_ns': 'contrast_nn',  # x3
        }

    def _init_crdt_corpus(self):
        self.relation_table = reltables.RelationTableCRDT
        self.relation_dic = {word.lower(): i for i, word in enumerate(self.relation_table)}
        self.rel2class = relation_set.crdt_rel2class
        self.relation_fixer = {
            'e-elaboration_ns': 'e-elaboration_sn',  # 1 time in ln94207_54:38-41~42-47
            'solutionhood_ns': 'solutionhood_sn',  # 1 time
            'means_sn': 'means_ns',  # 1 time
            'restatement_sn': 'restatement_nn',
            'gradation_ns': 'elaboration_ns',
            'contrast_ns': 'contrast_nn',
        }

    def _init_ert_corpus(self):
        self.relation_table = reltables.RelationTableERT
        self.relation_dic = {word.lower(): i for i, word in enumerate(self.relation_table)}
        self.rel2class = relation_set.basque_labels
        self.relation_fixer = {
            'result_sn': 'background_sn',  # x1
            'otherwise_ns': 'condition_ns',  # x1
            'preparation_ns': 'elaboration_ns',  # x2
            'evaluation_sn': 'preparation_sn',  # x2
            'elaboration_sn': 'preparation_sn',  # x3
            'restatement_sn': 'preparation_sn',  # x3
            'evidence_sn': 'justify_sn',  # x3
            'solutionhood_ns': 'elaboration_ns',  # x4
            'antithesis_ns': 'concession_ns',  # x6
            'interpretation_sn': 'preparation_sn',  # x6
        }

    def _init_gum_corpus(self):
        self.relation_table = reltables.RelationTableGUM

        if self.corpus_name == 'zho.rst.gcdt':
            self.relation_table.append('Elaboration_SN')

        self.relation_dic = {word.lower(): i for i, word in enumerate(self.relation_table)}
        self.relation_fixer = {
            'contingency_ns': 'condition_ns',
            'contingency_sn': 'condition_sn',
            'topic_ns': 'condition_ns',  # One example of this type in GUM v9.1
            'restatement_sn': 'restatement_ns'  # 4 examples in GUM_conversation_gossip
        }

    def _init_nldt_corpus(self):
        self.rel2class = relation_set.nldt_labels

        self.relation_table = reltables.RelationTableNLDT
        self.relation_dic = {word.lower(): i for i, word in enumerate(self.relation_table)}
        self.relation_fixer = {
            'sequence_sn': 'sequence_nn',
            'interpretation_sn': 'interpretation_ns',
            'contrast_sn': 'contrast_nn',
            'restatement_sn': 'restatement_ns',
        }

    def _init_pcc_corpus(self):
        self.rel2class = relation_set.germanPcc_labels
        self.relation_table = reltables.RelationTablePCC
        self.relation_dic = {word.lower(): i for i, word in enumerate(self.relation_table)}
        self.relation_fixer = {
            'background_ns': 'circumstance_ns',
            'solutionhood_ns': 'elaboration_ns'
        }

    def _init_prstc_corpus(self):
        # fas.rst.prstc

        self.rel2class = relation_set.fas_labels
        self.relation_table = reltables.RelationTablePRSTC
        self.relation_dic = {word.lower(): i for i, word in enumerate(self.relation_table)}
        self.relation_fixer = {
            'evaluation_sn': 'evaluation_nn',  # x1
            'evaluation_ns': 'evaluation_nn',  # x1
            'elaboration_nn': 'elaboration_ns',  # x1
            'contrast_ns': 'contrast_nn',  # x1
            'causal_ns': 'causal_nn',  # x1
            'causal_sn': 'causal_nn',  # x1
            'temporal_sn': 'temporal_nn',  # x2
        }

    def _init_rstdt_corpus(self):
        self.relation_table = reltables.RelationTableRSTDT
        self.rel2class = relation_set.rstdt_rel2class
        self.relation_dic = {word.lower(): i for i, word in enumerate(self.relation_table)}

    def _init_rurstb_corpus(self):
        # The corpus is split into separate trees (docname_part*.rs3)
        # "##### " are replaced with <P> tag as in the rst-dt
        # (although it still marks the beginning of a paragraph here, not the ending)
        # Also the corpus converted from rs3 -> isanlp -> rs3 to fix empty spans

        self.rel2class = relation_set.rrt_rel2class

        self.relation_table = reltables.RelationTableRuRSTB
        self.relation_dic = {word.lower(): i for i, word in enumerate(self.relation_table)}
        self.relation_fixer = {
            'restatement_sn': 'condition_sn',
            'restatement_ns': 'elaboration_ns',
            'solutionhood_ns': 'solutionhood_sn',
            'preparation_ns': 'elaboration_ns',
            'elaboration_sn': 'preparation_sn',
            'background_ns': 'elaboration_ns',
        }

    def _init_rststb_corpus(self):
        self.rel2class = relation_set.spanish_labels

        self.relation_table = reltables.RelationTableRSTSTB
        self.relation_dic = {word.lower(): i for i, word in enumerate(self.relation_table)}
        self.relation_fixer = {
            'enablement_sn': 'purpose_sn',
            'result_sn': 'cause_sn',
            'background_ns': 'circumstance_ns',
            'unless_ns': 'condition_ns',
            'alternative_ns': 'condition_ns',
            'antithesis_sn': 'concession_sn',
            'evidence_sn': 'motivation_sn',
            'enablement_ns': 'purpose_ns',
            'means_sn': 'means_ns',
            'summary_ns': 'elaboration_ns',
            'disjunction_nn': 'list_nn'
        }

    def _init_sctb_corpus(self):
        self.rel2class = relation_set.sctb_rel2class

        self.relation_table = reltables.RelationTableSCTB
        self.relation_dic = {word.lower(): i for i, word in enumerate(self.relation_table)}
        self.relation_fixer = {
            'means_sn': 'preparation_sn',  # x1
            'attribution_sn': 'preparation_sn',  # x1
            'attribution_ns': 'elaboration_ns',  # x1
            'evaluation_sn': 'preparation_sn',  # x1
            'restatement_ns': 'elaboration_ns',  # x1
            'background_ns': 'elaboration_ns',  # x2
        }

    def _init_sts_corpus(self):
        self.rel2class = relation_set.sts_rel2class

        self.relation_table = reltables.RelationTableSTS
        self.relation_dic = {word.lower(): i for i, word in enumerate(self.relation_table)}

        self.relation_fixer = {
            'antithesis_nn': 'antithesis_ns'  # Appears 1 time in OLL corpus
        }

    def _init_oll_corpus(self):
        self.rel2class = relation_set.sts_rel2class

        self.relation_table = reltables.RelationTableOLL
        self.relation_dic = {word.lower(): i for i, word in enumerate(self.relation_table)}

        self.relation_fixer = {
            'antithesis_nn': 'antithesis_ns'  # Appears 1 time in OLL corpus
        }

    def _init_umuc_corpus(self):
        self.rel2class = relation_set.umuc_rel2class
        self.relation_table = reltables.RelationTableUMUC
        # self.relation_table = reltables.RelationTableRSTDT + reltables.RelationTableCRDT + reltables.RelationTableAnnodis + reltables.RelationTableGUM + [
        #     'Solutionhood_NS', 'Circumstance_NS', 'Motivation_NS', 'Otherwise_NS', 'Otherwise_SN', 'Circumstance_SN', 'Interpretation_SN'
        # ] + reltables.RelationTableCSTN + reltables.RelationTableNLDT + reltables.RelationTableRSTSTB + reltables.RelationTableSCTB + reltables.RelationTableSTS
        self.relation_dic = {word.lower(): i for i, word in enumerate(self.relation_table)}

        self.relation_fixer = {
            'e-elaboration_sn': 'elaboration_sn',  # x2
            'restatement_sn': 'preparation_sn',  # x1
        }

    def _init_concat_corpus(self):
        """Initialise concatenated corpus settings."""
        self.concat_corpora = [
            'ces.rst.crdt', 'deu.rst.pcc', 'eng.erst.gum', 'eng.rst.oll',
            'eng.rst.rstdt', 'eng.rst.sts', 'eng.rst.umuc', 'eus.rst.ert',
            'fas.rst.prstc', 'fra.sdrt.annodis', 'nld.rst.nldt', 'por.rst.cstn',
            'rus.rst.rrg', 'rus.rst.rrt', 'spa.rst.rststb', 'spa.rst.sctb',
            'zho.rst.gcdt', 'zho.rst.sctb'
        ]

        corpus_tables = {
            'ces.rst.crdt': reltables.RelationTableCRDT,
            'deu.rst.pcc': reltables.RelationTablePCC,
            'eng.erst.gum': reltables.RelationTableGUM,
            'eng.rst.oll': reltables.RelationTableOLL,
            'eng.rst.rstdt': reltables.RelationTableRSTDT,
            'eng.rst.sts': reltables.RelationTableSTS,
            'eng.rst.umuc': reltables.RelationTableUMUC,
            'eus.rst.ert': reltables.RelationTableERT,
            'fas.rst.prstc': reltables.RelationTablePRSTC,
            'fra.sdrt.annodis': reltables.RelationTableAnnodis,
            'nld.rst.nldt': reltables.RelationTableNLDT,
            'por.rst.cstn': reltables.RelationTableCSTN,
            'rus.rst.rrg': reltables.RelationTableGUM,
            'rus.rst.rrt': reltables.RelationTableRuRSTB,
            'spa.rst.rststb': reltables.RelationTableRSTSTB,
            'spa.rst.sctb': reltables.RelationTableSCTB,
            'zho.rst.gcdt': reltables.RelationTableGUM + ['Elaboration_SN'],
            'zho.rst.sctb': reltables.RelationTableSCTB,
        }

        union = set()
        for corp in self.concat_corpora:
            union.update(corpus_tables[corp])
        self.relation_table = sorted(union)
        self.relation_dic = {w.lower(): i for i, w in enumerate(self.relation_table)}
        self.rel2class = {}
        self.relation_fixer = {}
        self.corpus = {'train': [], 'dev': [], 'test': []}

    def from_rs3(self):
        # Collect all *.edus, *.lisp in the same     directory
        self.prepare_lisp_format()

        # Collect pickled binaries for each document
        self.prepare_parser_format()

        self.construct_corpus()

    def from_pickle(self, filename):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        if hasattr(self, 'aug'):
            obj.aug = self.aug
        return obj

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def prepare_lisp_format(self):
        rs3_files = Path(self.input_path).glob('**/*.rs3')
        rs4_files = Path(self.input_path).glob('**/*.rs4')
        for rs3_file in sorted(list(rs3_files) + list(rs4_files)):
            try:
                self.convert_doc(filename=rs3_file.name,
                                 input_dir=str(rs3_file.parent),
                                 output_dir=self.output_path)
            except Exception as e:
                print(rs3_file)
                raise e

    def prepare_parser_format(self):
        files = list(self.output_path.glob('*.edus'))
        for edu_path in tqdm(files, desc='Reading *.lisp files'):
            lisp_path = edu_path.parent.joinpath(edu_path.name[:-5] + '.lisp')
            try:
                parser_input = self.generate_input(lisp_path, edu_path, edu_path)
            except Exception as e:
                print('Exception is evoked by:', edu_path)
                raise e
            with open(edu_path.parent.joinpath(edu_path.name[:-5] + '.pkl'), 'wb') as f:
                pickle.dump(parser_input, f)

    def get_data(self):
        """Return parsed data for the fixed split."""
        corpus = {k: v[:] for k, v in self.corpus.items()}

        if self.corpus_name != 'CONCAT':
            result = {'train': None, 'dev': None, 'test': None}
            for key in result.keys():
                docs = []
                for docname in corpus[key]:  # [:min(len(corpus[key]), 100)]:
                    filename = self.output_path.joinpath(docname + '.pkl')
                    try:
                        docs.append(pickle.load(open(filename, 'rb')))
                    except FileNotFoundError:
                        print('No such file in the corpus:', filename)

                if self.aug and key == 'train':
                    for docname in corpus[key]:
                        docs.extend(self._augment_doc(docname))

                input_sentences = [doc.sentences for doc in docs]
                edu_breaks = [doc.edu_breaks for doc in docs]
                decoder_input = [doc.decoder_inputs for doc in docs]
                relation_label = [doc.relation for doc in docs]
                parsing_breaks = [doc.parsing_index for doc in docs]
                golden_metric = [' '.join(doc.label_for_metrics_list) for doc in docs]
                parents_index = [doc.parents for doc in docs]
                sibling = [doc.siblings for doc in docs]
                result[key] = Data(input_sentences=input_sentences,
                                   edu_breaks=edu_breaks,
                                   decoder_input=decoder_input,
                                   relation_label=relation_label,
                                   parsing_breaks=parsing_breaks,
                                   golden_metric=golden_metric,
                                   parents_index=parents_index,
                                   sibling=sibling)

            return result['train'], result['dev'], result['test']

        # CONCAT corpus
        train_parts, dev_parts, test_parts = [], [], []
        for corp in self.concat_corpora:
            dm_path = f'data/dms/data_manager_{corp.lower()}.pickle'

            # aug = False
            # if self.aug:
            #     if corp not in ('eng.erst.gum', 'eng.rst.sts', 'eus.rst.ert', 'fra.srdt.annodis', 'rus.rst.rrg'):
            #         aug = True

            dm = DataManager(corpus=corp, aug=self.aug).from_pickle(dm_path)
            tr, dv, te = dm.get_data()
            train_parts.append(self._remap_data(tr, dm))
            dev_parts.append(self._remap_data(dv, dm))
            test_parts.append(self._remap_data(te, dm))

        return (self._merge_data(train_parts),
                self._merge_data(dev_parts),
                self._merge_data(test_parts))

    def _remap_data(self, data, dm):
        """Map relation indices of a Data object to the CONCAT table."""
        mapped = []
        for doc in data.relation_label:
            mapped.append([self.relation_dic[dm.relation_table[idx].lower()] for idx in doc])
        return Data(input_sentences=data.input_sentences,
                    edu_breaks=data.edu_breaks,
                    decoder_input=data.decoder_input,
                    relation_label=mapped,
                    parsing_breaks=data.parsing_breaks,
                    golden_metric=data.golden_metric,
                    parents_index=data.parents_index,
                    sibling=data.sibling)

    def _merge_data(self, data_list):
        """Concatenate several Data objects."""
        input_sentences = []
        edu_breaks = []
        decoder_input = []
        relation_label = []
        parsing_breaks = []
        golden_metric = []
        parents_index = []
        sibling = []
        for d in data_list:
            input_sentences.extend(d.input_sentences)
            edu_breaks.extend(d.edu_breaks)
            decoder_input.extend(d.decoder_input)
            relation_label.extend(d.relation_label)
            parsing_breaks.extend(d.parsing_breaks)
            golden_metric.extend(d.golden_metric)
            parents_index.extend(d.parents_index)
            sibling.extend(d.sibling)

        return Data(input_sentences=input_sentences,
                    edu_breaks=edu_breaks,
                    decoder_input=decoder_input,
                    relation_label=relation_label,
                    parsing_breaks=parsing_breaks,
                    golden_metric=golden_metric,
                    parents_index=parents_index,
                    sibling=sibling)

    def construct_corpus(self):
        # if self.corpus_name == 'GUM':
        for part in ('train', 'dev', 'test'):
            files = glob.glob(os.path.join(self.input_path, part, '*.rs3'))
            files += glob.glob(os.path.join(self.input_path, part, '*.rs4'))
            self.corpus[part] = [os.path.basename(f)[:-4] for f in files]

    def generate_input(self, lisp_path, text_path, edus_path, is_depth_manner=True):
        tree = BinaryTree(lisp_path, text_path, edus_path)
        edus_list = [edu.split() for edu in open(edus_path, 'r').read().splitlines()]  # GUM is pre-tokenized
        return self.find_document_span(tree.root, edus_list, is_depth_manner, tree.sentence_span)

    def find_document_span(self, node, edus_list, is_depth_manner, sentence_span_dic):
        parser_input = self.parse_sentence(node, edus_list, is_depth_manner)
        parser_input.sentence_span = self.get_sentence_span_list(sentence_span_dic, edus_list)
        return parser_input

    def get_sentence_span_list(self, sentence_span_dic, edus_list=None):
        """Return sentence spans as EDU ranges.

        If ``sentence_span_dic`` already contains span information, it is
        converted to a sorted list. Otherwise the spans are computed using
        spaCy sentence segmentation over the provided ``edus_list``.
        """

        if sentence_span_dic:
            sentence_list = []
            for key in sentence_span_dic:
                left, right = [int(x) for x in key.strip("[]").split(",")]
                sentence_list.append([left, right])
            sentence_list.sort(key=lambda x: x[0])
            return sentence_list

        if not edus_list:
            return []

        tokens = [tok for edu in edus_list for tok in edu]
        doc = Doc(self._nlp.vocab, words=tokens)
        for name, proc in self._nlp.pipeline:
            doc = proc(doc)

        token2edu = []
        for idx, edu in enumerate(edus_list, 1):
            token2edu.extend([idx] * len(edu))

        spans = []
        for sent in doc.sents:
            start_edu = token2edu[sent.start]
            end_edu = token2edu[sent.end - 1]
            spans.append([start_edu, end_edu])

        return spans

    def parse_sentence(self, root_node, edus_list, is_depth_manner, coarse=True):
        def get_depth_manner_node_list(root):
            node_list = []
            stack = []
            stack.append(root)
            while len(stack) > 0:
                node = stack.pop()
                node_list.append(node)
                if node.right is not None:
                    stack.append(node.right)
                if node.left is not None:
                    stack.append(node.left)
            return node_list

        def get_width_manner_node_list(root):
            node_list = []
            queue = []
            if root is not None:
                queue.append(root)
            while len(queue) != 0:
                node = queue.pop(0)
                node_list.append(node)
                if node.left is not None:
                    queue.append(node.left)
                if node.right is not None:
                    queue.append(node.right)
            return node_list

        root_node.parent = None
        parser_input = ParserInput()
        if is_depth_manner:
            node_list = get_depth_manner_node_list(root_node)
        else:
            node_list = get_width_manner_node_list(root_node)

        sentences_list = []

        edu_start = root_node.span[0]
        for node in node_list:
            if node.edu_id is not None:
                sentences_list.append([node.edu_id, edus_list[node.edu_id - 1]])
            else:
                parser_input.parsing_index.append(node.left.span[1] - edu_start)
                parser_input.decoder_inputs.append(node.span[0] - edu_start)

                parent_index = node.parent.span[1] - edu_start if node.parent is not None else 0
                parser_input.parents.append(parent_index)

                if node.parent is None:
                    sibling_index = 99
                else:
                    if node == node.parent.left:
                        sibling_index = 99
                    else:
                        sibling_index = node.parent.left.span[1] - edu_start

                parser_input.siblings.append(sibling_index)

                #   LabelforMetric:
                left_child_span = node.left.span
                right_child_span = node.right.span
                nuclearity = node.relation[:2]
                relation = node.relation[3:]

                # Label to Class
                if self.corpus_name in ('eng.rst.gum', 'eng.erst.gum', 'eng.rst.gentle', 'eng.erst.gentle',
                                        'rus.rst.rrg', 'zho.rst.gcdt'):
                    if coarse and relation != 'same-unit':
                        relation = relation.split('-')[0]
                else:
                    relation = self.rel2class.get(relation.lower(), relation.lower())

                # Relation:
                lookup_relation = (relation + '_' + nuclearity).lower()
                if lookup_relation in self.relation_fixer:
                    lookup_relation = self.relation_fixer.get(lookup_relation)
                    relation, nuclearity = lookup_relation.split('_')
                    nuclearity = nuclearity.upper()
                    if relation != 'same-unit':
                        relation = relation[0].upper() + relation[1:]

                parser_input.relation.append(self.relation_dic[lookup_relation])
                left_nuclearity = 'Nucleus' if nuclearity[0] == 'N' else 'Satellite'
                right_nuclearity = 'Nucleus' if nuclearity[1] == 'N' else 'Satellite'
                if nuclearity == 'NS' or nuclearity == 'SN':
                    if nuclearity == 'NS':
                        left_relation = 'span'
                        right_relation = relation
                    else:
                        left_relation = relation
                        right_relation = 'span'
                else:
                    left_relation = relation
                    right_relation = relation
                label_string = '(' + str(
                    left_child_span[0] - edu_start + 1) + ':' + left_nuclearity + '=' + left_relation + ':' + str(
                    left_child_span[1] - edu_start + 1) + ',' + str(
                    right_child_span[0] - edu_start + 1) + ':' + right_nuclearity + '=' + right_relation + ':' + str(
                    right_child_span[1] - edu_start + 1) + ')'
                parser_input.label_for_metrics_list.append(label_string)

        # parser_input.LabelforMetric = [' '.join(parser_input.label_for_metrics_list)]

        Sentences_list = sorted(sentences_list, key=lambda x: x[0])
        for i in range(len(Sentences_list)):
            parser_input.sentences += Sentences_list[i][1]
            parser_input.edu_breaks.append(len(parser_input.sentences) - 1)

        return parser_input

    def _is_valid_subtree(self, node, sentence_spans):
        start, end = node.span
        sent_starts = [s[0] for s in sentence_spans]
        sent_ends = [s[1] for s in sentence_spans]
        if start not in sent_starts or end not in sent_ends:
            return False
        inside = [s for s in sentence_spans if s[0] >= start and s[1] <= end]
        if len(inside) < 2:
            return False
        return inside[0][0] == start and inside[-1][1] == end

    def _get_valid_subtrees(self, root, sentence_spans):
        nodes = []
        queue = [root]
        while queue:
            node = queue.pop(0)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
            if node is not root and self._is_valid_subtree(node, sentence_spans):
                nodes.append(node)
        return nodes

    def _augment_doc(self, docname, min_rels=3, perc=0.25):
        lisp_path = self.output_path.joinpath(docname + '.lisp')
        edus_path = self.output_path.joinpath(docname + '.edus')
        if not lisp_path.exists() or not edus_path.exists():
            return []
        tree = BinaryTree(lisp_path, edus_path, edus_path)
        edus_list = [edu.split() for edu in open(edus_path, 'r').read().splitlines()]
        sentence_spans = self.get_sentence_span_list(tree.sentence_span, edus_list)
        nodes = self._get_valid_subtrees(tree.root, sentence_spans)
        inputs = []
        for node in nodes:
            sub_spans = [
                [s[0] - node.span[0] + 1, s[1] - node.span[0] + 1]
                for s in sentence_spans
                if s[0] >= node.span[0] and s[1] <= node.span[1]
            ]
            span_dic = {str(span): 1 for span in sub_spans}
            inputs.append(self.find_document_span(node, edus_list, True, span_dic))

        filtered = [
            inp for inp in inputs
            if min_rels < len(inp.decoder_inputs) < len(edus_list) - 2
        ]

        # Randomly sample according to percentage
        sample_size = max(1, int(len(filtered) * perc))
        filtered_sampled = random.sample(filtered, sample_size) if len(filtered) > sample_size else filtered

        return filtered_sampled

    def convert_doc(self, filename, input_dir, output_dir):
        """ Take all rs3/rs4 documents and save them in the same directory
            as *.edus and *.lisp files ready for processing. """
        rs3 = Rs3Document(os.path.join(input_dir, filename))
        rs3.read()
        rs3.writeEdu(output_dir)
        out_ext = '.lisp'
        rs3.writeTree(output_dir, out_ext)
