# -*- coding: utf-8 -*-
# @Author  : Lone Ranger
# @Function :
import argparse
from collections import OrderedDict

import torch
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification

from KnowledgeExtraction.QuestionClassificationBert import sentencePro

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
args, unknown = parser.parse_known_args()
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
