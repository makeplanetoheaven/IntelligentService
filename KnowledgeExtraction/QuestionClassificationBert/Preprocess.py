# -*- coding: utf-8 -*-
# @Author  : Lone Ranger
# @Function :

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import json
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch


def reverse(dataframe):
    res = []
    for i in dataframe:
        res.append(i[::-1])
    res = pd.DataFrame(res)
    return res


def concat_dataframe(row_df, label):
    reversed_df = reverse(row_df)
    row_df = pd.concat([row_df, reversed_df], axis=0)
    row_df = row_df.reset_index(drop=True)
    label = [label] * len(row_df)
    label = pd.DataFrame(label)
    row_df = pd.concat([row_df, label], axis=1)
    return row_df


def df2json(dataframe, mode):
    temp = dataframe.to_dict(orient='records')
    #     print(temp)
    write_path = 'QAData/' + mode + '.json'
    with open(write_path, 'w', encoding='utf-8') as f:
        for content in temp:
            json_data = json.dumps(content, ensure_ascii=False)
            f.write(json_data)
            f.write('\n')


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_sentences_examples(self, sentence):
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        dicts = []
        with codecs.open(input_file, 'r', 'utf-8') as infs:
            for inf in infs:
                inf = inf.strip()
                dicts.append(json.loads(inf))
        return dicts

    @classmethod
    def _read_list(cls, input_list):
        res = []
        temp = ''
        for i in input_list:
            temp += i
        res.append(temp)
        return res




class MyPro(DataProcessor):
    '''自定义数据读取方法，针对json文件

    Returns:
        examples: 数据集，包含index、中文文本、类别三个部分
    '''

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json").replace('\\', '/')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "val.json").replace('\\', '/')), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json").replace('\\', '/')), 'test')

    def get_labels(self):
        return [0, 1]

    def get_text_a(self, data_dir):
        dicts = self._read_json(os.path.join(data_dir, "test.json").replace('\\', '/'))
        res = [0] * len(dicts)
        for i, text in enumerate(dicts):
            text = dicts[i]['content']
            res[i] = text
        return res

    def write_predict_result(self, contents, predict_list, gt_list):
        try:
            res = []
            for index, text in enumerate(contents):
                temp = {'index': index, 'content': text, 'predict_label': str(predict_list[index]),
                        'gt_label': str(gt_list[index])}
                res.append(temp)
            with open('./KnowledgeMemory/predict.json', 'w', encoding='utf-8') as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(e)
            return False

    def _create_examples(self, dicts, set_type):
        examples = []
        for (i, infor) in enumerate(dicts):
            guid = "%s-%s" % (set_type, i)
            text_a = infor['content']
            #             print('--------------------------------------\n'+ text_a)
            label = infor['label']
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))

        return examples


class sentencePro(DataProcessor):

    def get_sentences_examples(self, sentence_list):
        return self._create_sentences_examples(self._read_list(sentence_list), 'test')

    def get_labels(self):
        return [0, 1]

    def _create_sentences_examples(self, sentences_list, set_type):
        examples = []
        for (i, infor) in enumerate(sentences_list):
            guid = "%s-%s" % (set_type, i)
            text_a = infor
            label = 0
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))

        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, show_exp=True):
    '''Loads a data file into a list of `InputBatch`s.
    Args:
        examples      : [List] 输入样本，包括question, label, index
        label_list    : [List] 所有可能的类别，可以是int、str等，如['book', 'city', ...]
        max_seq_length: [int] 文本最大长度
        tokenizer     : [Method] 分词方法
    Returns:
        features:
            input_ids  : [ListOf] token的id，在chinese模式中就是每个分词的id，对应一个word vector
            input_mask : [ListOfInt] 真实字符对应1，补全字符对应0
            segment_ids: [ListOfInt] 句子标识符，第一句全为0，第二句全为1
            label_id   : [ListOfInt] 将Label_list转化为相应的id表示
    '''
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        if type(example.text_a) != str:
            continue
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5 and show_exp:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan
