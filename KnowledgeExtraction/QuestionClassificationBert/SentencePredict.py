# -*- coding: utf-8 -*-
# @Author  : Lone Ranger
# @Function :


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from KnowledgeExtraction.QuestionClassificationBert.Preprocess import convert_examples_to_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test(model, processor, args, label_list, tokenizer, device, input):
    '''模型测试

    Args:
        model: 模型
	processor: 数据读取方法
	args: 参数表
	label_list: 所有可能类别
	tokenizer: 分词方法
	device

    Returns:
        f1: F1值
    '''
    # test_examples = processor.get_test_examples(args.data_dir)
    test_examples = processor.get_sentences_examples(input)
    test_features = convert_examples_to_features(
        test_examples, label_list, args.get('max_seq_length'), tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.get('eval_batch_size'))

    model.eval()
    predict = np.zeros((0,), dtype=np.int32)
    gt = np.zeros((0,), dtype=np.int32)
    for input_ids, input_mask, segment_ids, label_ids in test_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
            pred = logits.max(1)[1]
            predict = np.hstack((predict, pred.cpu().numpy()))
            gt = np.hstack((gt, label_ids.cpu().numpy()))

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
    print('predict result is {}'.format(predict[0]))
    acc = np.mean(metrics.accuracy_score(gt, predict, normalize=True))
    print("Accuracy is {}".format(acc))
    f1 = np.mean(metrics.f1_score(predict, gt, average=None))
    print('F1 score in test set is {}'.format(f1))

    return f1

