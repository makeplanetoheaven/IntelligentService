# -*- coding: utf-8 -*-
# @Author  : Lone Ranger
# @Function :

# 引入外部包
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer
from sklearn import metrics
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

#  引入内部包
from KnowledgeExtraction.QuestionClassificationBert.Preprocess import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BertForClassification:
    def __init__(self, train_args):
        self.train_args = train_args

    def val(self, model, processor, args, label_list, tokenizer, device):
        """模型验证

        Args:
            model: 模型
        processor: 数据读取方法
        args: 参数表
        label_list: 所有可能类别
        tokenizer: 分词方法
        device

        Returns:
            f1: F1值
        """
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, show_exp=False)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        predict = np.zeros((0,), dtype=np.int32)
        gt = np.zeros((0,), dtype=np.int32)
        for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
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

        print(len(gt))
        f1 = np.mean(metrics.f1_score(predict, gt, average=None))
        print(f1)

        return f1

    def train(self):
        # 1.设置参数
        # ArgumentParser对象保存了所有必要的信息，用以将命令行参数解析为相应的python数据类型
        # GlobalVariable.set_value('BERT_DATA','./KnowledgeMemory/BertDataDir')
        bert_args = self.train_args
        args = bert_args.set_args()

        # 调用add_argument()向ArgumentParser对象添加命令行参数信息，这些信息告诉ArgumentParser对象如何处理命令行参数
        # parser.add_argument("--data_dir",
        #                     default='./KnowledgeMemory/BertDataDir',
        #                     type=str,
        #                     # required = True,
        #                     help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
        # parser.add_argument("--bert_model",
        #                     default='bert-base-chinese',
        #                     type=str,
        #                     # required = True,
        #                     help="choose [bert-base-chinese] mode.")
        # parser.add_argument("--task_name",
        #                     default='MyPro',
        #                     type=str,
        #                     # required = True,
        #                     help="The name of the task to train.")
        # parser.add_argument("--output_dir",
        #                     default='./ModelMemory/QuestionClassificationBert/model/checkpoints',
        #                     type=str,
        #                     # required = True,
        #                     help="The output directory where the model checkpoints will be written")
        # parser.add_argument("--model_save_pth",
        #                     default='./ModelMemory/QuestionClassificationBert/model/checkpoints/bert_classification.pth',
        #                     type=str,
        #                     # required = True,
        #                     help="The output directory where the model checkpoints will be written")
        #
        # # other parameters
        # parser.add_argument("--max_seq_length",
        #                     default=100,
        #                     type=int,
        #                     help="字符串最大长度")
        # parser.add_argument("--do_train",
        #                     default=True,
        #                     action='store_true',
        #                     help="训练模式")
        # parser.add_argument("--do_eval",
        #                     default=True,
        #                     action='store_true',
        #                     help="验证模式")
        # parser.add_argument("--do_lower_case",
        #                     default=False,
        #                     action='store_true',
        #                     help="英文字符的大小写转换，对于中文来说没啥用")
        # parser.add_argument("--train_batch_size",
        #                     default=64,
        #                     type=int,
        #                     help="训练时batch大小")
        # parser.add_argument("--eval_batch_size",
        #                     default=1,
        #                     type=int,
        #                     help="验证时batch大小")
        # parser.add_argument("--learning_rate",
        #                     default=5e-5,
        #                     type=float,
        #                     help="Adam初始学习步长")
        # parser.add_argument("--num_train_epochs",
        #                     default=5.0,
        #                     type=float,
        #                     help="训练的epochs次数")
        # parser.add_argument("--warmup_proportion",
        #                     default=0.1,
        #                     type=float,
        #                     help="Proportion of training to perform linear learning rate warmup for."
        #                          "E.g., 0.1 = 10%% of training.")
        # parser.add_argument("--no_cuda",
        #                     default=True,
        #                     action='store_true',
        #                     help="用不用CUDA")
        # parser.add_argument("--local_rank",
        #                     default=-1,
        #                     type=int,
        #                     help="local_rank for distributed training on gpus.")
        # parser.add_argument("--seed",
        #                     default=777,
        #                     type=int,
        #                     help="初始化时的随机数种子")
        # parser.add_argument("--gradient_accumulation_steps",
        #                     default=1,
        #                     type=int,
        #                     help="Number of updates steps to accumulate before performing a backward/update pass.")
        # parser.add_argument("--optimize_on_cpu",
        #                     default=False,
        #                     action='store_true',
        #                     help="Whether to perform optimization and keep the optimizer averages on CPU.")
        # parser.add_argument("--fp16",
        #                     default=False,
        #                     action='store_true',
        #                     help="Whether to use 16-bit float precision instead of 32-bit.")
        # parser.add_argument("--loss_scale",
        #                     default=128,
        #                     type=float,
        #                     help="Loss scaling, positive power of 2 values can improve fp16 convergence.")
        #
        # #     args = parser.parse_args()
        # args, unknown = parser.parse_known_args()
        # 对模型输入进行处理的processor，git上可能都是针对英文的processor

        # 2.设置processor
        processors = {'mypro': MyPro}

        # 3.判断是否使用GPU或fp16
        if args.get('local_rank') == -1 or args.get('no_cuda'):
            device = torch.device("cuda" if torch.cuda.is_available() and not args.get('no_cuda') else "cpu")
            n_gpu = torch.cuda.device_count()
            # n_gpu = 0
        else:
            device = torch.device("cuda", args.get('local_rank'))
            n_gpu = 1
            torch.distributed.init_process_group(backend='nccl')
            if args.get('fp16'):
                logger.info("16-bits training currently not supported in distributed training")
                args['fp16'] = False  # (see https://github.com/pytorch/pytorch/pull/13496)
        logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.get('local_rank') != -1))

        if args.get('gradient_accumulation_steps') < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.get('gradient_accumulation_steps')))
        # 计算训练时的batch_size
        args['train_batch_size'] = int(args.get('train_batch_size') / args.get('gradient_accumulation_steps'))

        # 随机初始化种子
        random.seed(args.get('seed'))
        np.random.seed(args.get('seed'))
        torch.manual_seed(args.get('seed'))
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.get('seed'))

        # 至少要选训练或验证其中一种模式
        if not args.get('do_train') and not args.get('do_eval'):
            raise ValueError("At least one of `do_train` or `do_eval` must be True.")

        # 判断输出路径是否存在同时路径下是否为空
        if os.path.exists(args.get('output_dir')) and os.listdir(args.get('output_dir')):
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.get('output_dir')))
        os.makedirs(args.get('output_dir'), exist_ok=True)

        # 选择任务名，需要确保任务名在processors中预定义
        task_name = args.get('task_name').lower()
        if task_name not in processors:
            raise ValueError("Task not found: %s" % (task_name))
        # 加载自定义processor，加载label_list
        processor = processors[task_name]()
        label_list = processor.get_labels()

        # 调用分割方法，使用预训练模型中的自带方法
        tokenizer = BertTokenizer.from_pretrained(args.get('bert_model'), do_lower_case=args.get('do_lower_case'))

        # 获取训练样本，并计算训练所需步数
        train_examples = None
        num_train_steps = None
        if args['do_train']:
            train_examples = processor.get_train_examples(args.get('data_dir'))
            num_train_steps = int(
                len(train_examples) / args.get('train_batch_size') / args.get('gradient_accumulation_steps') * args.get(
                    'num_train_epochs'))

        # 准备模型，使用预训练的bert_base_chinese模型的SequenceClassification用于分类任务
        model = BertForSequenceClassification.from_pretrained(args.get('bert_model'),
                                                              cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                                  args.get('local_rank')), num_labels=3)
        # 是否使用16位精度提升速度
        if args.get('fp16'):
            model.half()
            model.to(device)
        if args.get('local_rank') != -1:
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[args.get('local_rank')],
                                                              output_device=args.get('local_rank'))
        # 如果 gpu数大于1，可以使用数据分布式计算加速，CPU上没有
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # 配置优化器，使用BertAdam，根据是GPU运算还是CPU还是16位精度，获取parameter——optimizer(待优化参数列表)
        if args.get('fp16'):
            param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                               for n, param in model.named_parameters()]
        elif args.get('optimize_on_cpu'):
            param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                               for n, param in model.named_parameters()]
        else:
            param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
        t_total = num_train_steps
        if args.get('local_rank') != -1:
            t_total = t_total // torch.distributed.get_world_size()
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.get('learning_rate'),
                             warmup=args.get('warmup_proportion'),
                             t_total=t_total)

        # 开始训练
        # 准备步骤：
        # (1)将examples转为features
        # (2)将features中的参数加载到torch.tensor中
        # (3)使用TensorDataset将所有的featuretensor打包作为训练集数据，选择随机采样或分布式采样
        # (4)使用DataLoader将训练集加载，确定采样方式和batch_size
        global_step = 0
        if args.get('do_train'):
            train_features = convert_examples_to_features(
                train_examples, label_list, args.get('max_seq_length'), tokenizer, show_exp=False)
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", args.get('train_batch_size'))
            logger.info("  Num steps = %d", num_train_steps)
            all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            if args.get('local_rank') == -1:
                train_sampler = RandomSampler(train_data)
            else:
                train_sampler = DistributedSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.get('train_batch_size'))

            # 开始训练，计算loss，同样分为GPU/CPU/16位精度分别计算
            # 每次计算完后反向传播 loss.backward()
            #
            model.train()
            best_score = 0
            flags = 0
            for _ in trange(int(args.get('num_train_epochs')), desc="Epoch"):
                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                    batch = tuple(t for t in batch)
                    input_ids, input_mask, segment_ids, label_ids = batch
                    loss = model(input_ids, segment_ids, input_mask, label_ids)
                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.get('fp16') and args.get('loss_scale') != 1.0:
                        # rescale loss for fp16 training
                        # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                        loss = loss * args.get('loss_scale')
                    if args.get('gradient_accumulation_steps') > 1:
                        loss = loss / args.get('gradient_accumulation_steps')
                    loss.backward()

                    if (step + 1) % args.get('gradient_accumulation_steps') == 0:
                        if args.get('fp16') or args.get('optimize_on_cpu'):
                            if args.get('fp16') and args.get('loss_scale') != 1.0:
                                # scale down gradients for fp16 training
                                for param in model.parameters():
                                    if param.grad is not None:
                                        param.grad.data = param.grad.data / args.get('loss_scale')
                            is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                            if is_nan:
                                logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                                args['loss_scale'] = args.get('loss_scale') / 2
                                model.zero_grad()
                                continue
                            optimizer.step()
                            copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                        else:
                            optimizer.step()
                        model.zero_grad()
                # 调用val()在验证集上验证当前模型效果，每次得到更高的f1值，就将模型对应的参数保存，
                # 并设置如果连续n次f1值都没有提升，认为模型收敛，结束训练
                f1 = self.val(model, processor, args, label_list, tokenizer, device)
                if f1 > best_score:
                    best_score = f1
                    print('*f1 score = {}'.format(f1))
                    flags = 0
                    checkpoint = {
                        'state_dict': model.state_dict()
                    }
                    torch.save(checkpoint, args.get('model_save_path'))
                else:
                    print('f1 score = {}'.format(f1))
                    flags += 1
                    if flags >= 6:
                        break

        model.load_state_dict(torch.load(args.get('model_save_path'))['state_dict'])
        # test(model, processor, args, label_list, tokenizer)

# if __name__ == '__main__':
#     main()
