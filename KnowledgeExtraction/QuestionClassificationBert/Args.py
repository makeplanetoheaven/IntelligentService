# -*- coding: utf-8 -*-
# @Author  : Lone Ranger
# @Function :
import argparse

from mock.mock import self


class return_args():
    def __init__(self, data_dir='./KnowledgeMemory/BertDataDir', bert_model='bert_base-chinese',
                 task_name='sentencePro', output_dir='./ModelMemory/QuestionClassificationBert/model/checkpoints',
                 ):
        self.data_dir = data_dir
        self.bert_model = bert_model
        self.task_name = task_name
        self.output_dir = output_dir
        self.mode_save_path = model_save_path
        self.max_seq_length = 256
        self.do_train = do_train
        self.do_eval = do_eval
        self.do_lower_case = False
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.learing_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.warmup_proportion = 0.1
        self.no_cuda = no_cuda
        self.local_rank = local_rank
        self.seed = 777
        self.gradient_accumulation_steps = 1
        self.optimize_on_cpu = False
        self.fp16 = False
        self.loss_scale = 128

    parser = argparse.ArgumentParser()

    # required parameters
    # 调用add_argument()向ArgumentParser对象添加命令行参数信息，这些信息告诉ArgumentParser对象如何处理命令行参数
    parser.add_argument("--data_dir",
                        default=self.data_dir,
                        type=str,
                        # required = True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model",
                        default=self.bert_model,
                        # default='bert-base-chinese',
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
    args, unknown = parser.parse_known_args()
    return args
