# -*- coding: utf-8 -*-
# @Author  : Lone Ranger
# @Function :


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
from collections import OrderedDict

import numpy as np
import torch
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from sklearn import metrics
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from KnowledgeExtraction.QuestionClassificationBert import Args
from KnowledgeExtraction.QuestionClassificationBert.Preprocess import convert_examples_to_features, sentencePro

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test(model, processor, args, label_list, tokenizer, device):
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
    test_examples = processor.get_sentences_examples(input())
    test_features = convert_examples_to_features(
        test_examples, label_list, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

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

    # review = processor.get_text_a(args.data_dir)
    # isSaved = processor.write_predict_result(review, predict, gt)
    # for i in range(len(predict)):
    #     if predict[i] == 1 and predict[i] != gt[i]:
    #         print('predict is: {}'.format(str(predict[i])) + '\n' + 'gt is: {}'.format(str(gt[i])) + '\n' + review[i])
    return f1


model_path = './checkpoints/bert_classification.pth'
# processors = {'mypro': MyPro}
processors = {'sentencePro': sentencePro}
processor = processors['sentencePro']()
label_list = processor.get_labels()
parser = argparse.ArgumentParser()

# required parameters
# 调用add_argument()向ArgumentParser对象添加命令行参数信息，这些信息告诉ArgumentParser对象如何处理命令行参数
parser.add_argument("--data_dir",
                    default='./KnowledgeMemory/BertDataDir',
                    type=str,
                    # required = True,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--bert_model",
                    default='bert-base-chinese',
                    type=str,
                    # required = True,
                    help="choose [bert-base-chinese] mode.")
parser.add_argument("--task_name",
                    default='sentencePro',
                    type=str,
                    # required = True,
                    help="The name of the task to train.")
parser.add_argument("--output_dir",
                    default='./ModelMemory/QuestionClassificationBert/model/checkpoints',
                    type=str,
                    # required = True,
                    help="The output directory where the model checkpoints will be written")
parser.add_argument("--model_save_pth",
                    default='./ModelMemory/QuestionClassificationBert/model/checkpoints/bert_classification.pth',
                    type=str,
                    # required = True,
                    help="The output directory where the model checkpoints will be written")

# other parameters
parser.add_argument("--max_seq_length",
                    default=100,
                    type=int,
                    help="字符串最大长度")
parser.add_argument("--do_train",
                    default=True,
                    action='store_true',
                    help="训练模式")
parser.add_argument("--do_eval",
                    default=True,
                    action='store_true',
                    help="验证模式")
parser.add_argument("--do_lower_case",
                    default=False,
                    action='store_true',
                    help="英文字符的大小写转换，对于中文来说没啥用")
parser.add_argument("--train_batch_size",
                    default=64,
                    type=int,
                    help="训练时batch大小")
parser.add_argument("--eval_batch_size",
                    default=1,
                    type=int,
                    help="验证时batch大小")
parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    help="Adam初始学习步长")
parser.add_argument("--num_train_epochs",
                    default=5.0,
                    type=float,
                    help="训练的epochs次数")
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for."
                         "E.g., 0.1 = 10%% of training.")
parser.add_argument("--no_cuda",
                    default=True,
                    action='store_true',
                    help="用不用CUDA")
parser.add_argument("--local_rank",
                    default=-1,
                    type=int,
                    help="local_rank for distributed training on gpus.")
parser.add_argument("--seed",
                    default=777,
                    type=int,
                    help="初始化时的随机数种子")
parser.add_argument("--gradient_accumulation_steps",
                    default=1,
                    type=int,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--optimize_on_cpu",
                    default=False,
                    action='store_true',
                    help="Whether to perform optimization and keep the optimizer averages on CPU.")
parser.add_argument("--fp16",
                    default=False,
                    action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit.")
parser.add_argument("--loss_scale",
                    default=128,
                    type=float,
                    help="Loss scaling, positive power of 2 values can improve fp16 convergence.")

#     args = parser.parse_args()
# args, unknown = parser.parse_known_args()
args = Args.return_args()
device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
print(device)

tokenizer = BertTokenizer.from_pretrained(args.bert_model)
# model = BertForSequenceClassification.from_pretrained(args.bert_model,cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))
model = BertForSequenceClassification.from_pretrained(args.bert_model)
state_dict = torch.load(args.model_save_pth, map_location='cpu')['state_dict']

# 创建一个新的state_dict加载预训练的模型参数进行预测
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]  # remove module.
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)

if __name__ == '__main__':
    test(model, processor, args, label_list, tokenizer, device)
