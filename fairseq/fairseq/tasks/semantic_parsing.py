# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import json
import itertools
import logging
import os

import numpy as np

from fairseq import metrics, options, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    data_utils,
    encoders,
    indexed_dataset,
    FairseqDataset,
    LanguagePairDataset,
    LanguageTripleDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
)
from fairseq.data import iterators, Dictionary


from fairseq.tasks import FairseqTask, register_task

EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


def load_langpair_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions, prepend_bos=False, load_alignments=False,
    truncate_source=False, append_source_id=False,
    num_buckets=0,
    shuffle=True,
    dis_data=False
):



    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        print('filename : '+str(filename))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        if os.path.exists(prefix+"filter"):
            filter_indice = set(json.load(open(prefix+"filter")))
            print("filtered example number : "+str(len(filter_indice)))
        else:
            filter_indice = None


        src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl, append_eos = False)
        if filter_indice is not None:
            src_dataset.lines = [item for i, item in enumerate(src_dataset.lines) if i not in filter_indice]
            src_dataset.tokens_list = [item for i, item in enumerate(src_dataset.tokens_list) if i not in filter_indice]
            src_dataset.sizes = np.array([item for i, item in enumerate(src_dataset.sizes) if i not in filter_indice])
            src_dataset.size = len(src_dataset.tokens_list)
        print("src data number : "+str(len(src_dataset)))
        print("src data example : "+str(src_dataset[0]))
        print("max src example size : "+str(src_dataset.sizes.max()))
        if truncate_source:
            src_dataset = TruncateDataset(
                            src_dataset,
                            max_source_positions - 1,
                            )
        print("truncated max src example size : "+str(src_dataset.sizes.max()))
        src_datasets.append(src_dataset)


        tgt_dataset = data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)
        if filter_indice is not None:
            tgt_dataset.lines = [item for i, item in enumerate(tgt_dataset.lines) if i not in filter_indice]
            tgt_dataset.tokens_list = [item for i, item in enumerate(tgt_dataset.tokens_list) if i not in filter_indice]
            tgt_dataset.sizes = np.array([item for i, item in enumerate(tgt_dataset.sizes) if i not in filter_indice])
            tgt_dataset.size = len(tgt_dataset.tokens_list)
        print("tgt data number : "+str(len(tgt_dataset)))
        print("tgt data example : "+str(tgt_dataset[0]))
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)


        logger.info('{} {} {}-{} {} examples'.format(
            data_path, split_k, src, tgt, len(src_datasets[-1])
        ))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    print("src data example : "+str(src_dataset[0]))


    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(src_dataset, src_dict.index('[{}]'.format(src)))
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(tgt_dataset, tgt_dict.index('[{}]'.format(tgt)))
        eos = tgt_dict.index('[{}]'.format(tgt))

    print("src data example : "+str(src_dataset[0]))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, '{}.{}-{}.align'.format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(align_path, None, dataset_impl = "json", append_eos = False)         
            if filter_indice is not None:
                align_dataset.tokens_list = [item for i, item in enumerate(align_dataset.tokens_list) if i not in filter_indice]
                align_dataset.sizes = np.array([item for i, item in enumerate(align_dataset.sizes) if i not in filter_indice])
                align_dataset.size = len(align_dataset.tokens_list)


    src_align_dataset = None
    if load_alignments:
        src_align_path = os.path.join(data_path, '{}.{}-{}.src_align'.format(split, src, tgt))
        if indexed_dataset.dataset_exists(src_align_path, impl=dataset_impl):
            src_align_dataset = data_utils.load_indexed_dataset(src_align_path, None, dataset_impl = "json", append_eos = False)
            if filter_indice is not None:
                src_align_dataset.tokens_list = [item for i, item in enumerate(src_align_dataset.tokens_list) if i not in filter_indice]
                src_align_dataset.sizes = np.array([item for i, item in enumerate(src_align_dataset.sizes) if i not in filter_indice])
                src_align_dataset.size = len(src_align_dataset.tokens_list)



    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

    
    return LanguagePairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset, 
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
    )


def load_langtriple_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    tag,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions, prepend_bos=False, load_alignments=False,
    truncate_source=False, append_source_id=False,
    num_buckets=0,
    shuffle=True,
    data_name=None,
    tag_input=False,
):

# load source, tag and target datasets


    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        print('filename : '+str(filename))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []
    tag_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
        print("src data example : "+str(src_dataset[0]))
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)
        print("prefix :  "+str(prefix))

        tag_dataset = data_utils.load_indexed_dataset(prefix + tag, tgt_dict, dataset_impl)
        tag_datasets.append(tag_dataset)


        tgt_dataset = data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info('{} {} {}-{} {} examples'.format(
            data_path, split_k, src, tgt, len(src_datasets[-1])
        ))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
        tag_dataset = tag_datasets[0]
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None
        tag_dataset = ConcatDataset(tag_datasets, sample_ratios)


    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
        tag_dataset = PrependTokenDataset(tag_dataset, tgt_dict.bos())

    print("src data example : "+str(src_dataset[0]))


    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(src_dataset, src_dict.index('[{}]'.format(src)))
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(tgt_dataset, tgt_dict.index('[{}]'.format(tgt)))
        tag_dataset = AppendTokenDataset(tag_dataset, tgt_dict.index('[{}]'.format(tgt)))
        eos = tgt_dict.index('[{}]'.format(tgt))

    print("src data example : "+str(src_dataset[0]))


    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, '{}.align.{}-{}'.format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(align_path, None, dataset_impl)

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguageTripleDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,
        tag_dataset,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset, eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        data_name=data_name,
        tag_input=tag_input,
    )






@register_task('semantic_parsing')
class SemanticParsingTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--load-alignments', action='store_true',
                            help='load the binarized alignments')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--truncate-source', action='store_true', default=False,
                            help='truncate source to max-source-positions')
        parser.add_argument('--num-batch-buckets', default=0, type=int, metavar='N',
                            help='if >0, then bucket source and target lengths into N '
                                 'buckets and pad accordingly; this is useful on TPUs '
                                 'to minimize the number of compilations')
        parser.add_argument('--bert-path', default=None, type=str,
                             help='bert path')
        parser.add_argument('--bert-encoder', action='store_true', default=False)
        parser.add_argument('--bert-decoder', action='store_true', default=False)
        parser.add_argument('--freeze-bert', action='store_true', default=False)

        #use disenntagled data for training and validationn
        parser.add_argument('--dis-data', action='store_true', default=False)

        parser.add_argument('--output-hiddens', default=False,
                            action='store_true',
                            help='output hidden_states at each time step')




    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict


    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol('<mask>')
        return dictionary
        
    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        # if args.bert_encoder:
        #     src_dict.add_symbol('<mask>')
        src_dict.add_symbol('<mask>')

        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        # if args.bert_decoder:
        #     tgt_dict.add_symbol('<mask>')
        tgt_dict.add_symbol('<mask>')

        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info('[{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        logger.info('[{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)


    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        print("Load original dataset")
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang
        if self.args.dis_data and (split == "train" or split == "valid"):
            self.datasets[split] = load_langpair_dataset(
                data_path, split, src, self.src_dict, tgt, self.tgt_dict,
                combine=combine, dataset_impl=self.args.dataset_impl,
                upsample_primary=self.args.upsample_primary,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
                load_alignments=self.args.load_alignments,
                truncate_source=self.args.truncate_source,
                num_buckets=self.args.num_batch_buckets,
                shuffle=(split != 'test'),
                dis_data=True
            )
        else:
            self.datasets[split] = load_langpair_dataset(
                data_path, split, src, self.src_dict, tgt, self.tgt_dict,
                combine=combine, dataset_impl=self.args.dataset_impl,
                upsample_primary=self.args.upsample_primary,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
                load_alignments=self.args.load_alignments,
                truncate_source=self.args.truncate_source,
                num_buckets=self.args.num_batch_buckets,
                shuffle=(split != 'test'),
            )

        print("split : "+str(split)+" ; dataset size : "+str(len(self.datasets[split])))

    def build_model(self, args):
        model = super().build_model(args)
        self.bpe = encoders.build_bpe(args)

        return model

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    # def build_model(self, args):
    #     model = super().build_model(args)
    #     if getattr(args, 'eval_bleu', False):
    #         assert getattr(args, 'eval_bleu_detok', None) is not None, (
    #             '--eval-bleu-detok is required if using --eval-bleu; '
    #             'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
    #             'to disable detokenization, e.g., when using sentencepiece)'
    #         )
    #         detok_args = json.loads(getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
    #         self.tokenizer = encoders.build_tokenizer(Namespace(
    #             tokenizer=getattr(args, 'eval_bleu_detok', None),
    #             **detok_args
    #         ))

    #         gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
    #         self.sequence_generator = self.build_generator([model], Namespace(**gen_args))
    #     return model

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict



@register_task('semantic_parsing_tag')
class SemanticParsingTagTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('-g', '--tag-lang', default=None, metavar='TAG',
                            help='tag language')
        parser.add_argument('--load-alignments', action='store_true',
                            help='load the binarized alignments')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--truncate-source', action='store_true', default=False,
                            help='truncate source to max-source-positions')
        parser.add_argument('--num-batch-buckets', default=0, type=int, metavar='N',
                            help='if >0, then bucket source and target lengths into N '
                                 'buckets and pad accordingly; this is useful on TPUs '
                                 'to minimize the number of compilations')
        parser.add_argument('--data-name', default=None, metavar='DATA-NAME',
                            help='data name e.g. atis_logic')
        parser.add_argument('--tag-input', action='store_true', default=False,
                            help='provide tag as input to the encoder')

        parser.add_argument('--bert-path', default=None, type=str,
                             help='bert path')
        parser.add_argument('--bert-encoder', action='store_true', default=False)
        parser.add_argument('--freeze-bert', action='store_true', default=False)





    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        if args.bert_encoder:
            src_dict.add_symbol('<mask>')
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info('[{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        logger.info('[{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt, tag = self.args.source_lang, self.args.target_lang, self.args.tag_lang

        self.datasets[split] = load_langtriple_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            tag,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            truncate_source=self.args.truncate_source,
            num_buckets=self.args.num_batch_buckets,
            shuffle=(split != 'test'),
            data_name=self.args.data_name,
            tag_input=self.args.tag_input,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    # def build_model(self, args):
    #     model = super().build_model(args)
    #     if getattr(args, 'eval_bleu', False):
    #         assert getattr(args, 'eval_bleu_detok', None) is not None, (
    #             '--eval-bleu-detok is required if using --eval-bleu; '
    #             'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
    #             'to disable detokenization, e.g., when using sentencepiece)'
    #         )
    #         detok_args = json.loads(getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
    #         self.tokenizer = encoders.build_tokenizer(Namespace(
    #             tokenizer=getattr(args, 'eval_bleu_detok', None),
    #             **detok_args
    #         ))

    #         gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
    #         self.sequence_generator = self.build_generator([model], Namespace(**gen_args))
    #     return model

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict



