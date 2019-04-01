# -*- coding: utf-8 -*-
# @Author  : Lone Ranger
# @Function :


class BertArgs:

    # TODO：增加注释
    def __init__(self,
                 data_dir='./KnowledgeMemory/BertDataDir',
                 bert_model='bert-base-chinese',
                 task_name='MyPro',
                 output_dir='./ModelMemory/QuestionClassificationBert/model/checkpoints',
                 model_save_path='./ModelMemory/QuestionClassificationBert/model/checkpoints/bert_classification.pth',
                 max_seq_length=128,
                 do_train=False,
                 do_eval=False,
                 train_batch_size=64,
                 eval_batch_size=1,
                 learning_rate=5e-5,
                 num_train_epochs=5,
                 no_cuda=True,
                 local_rank=-1,
                 ):
        """
        :param data_dir: 训练集验证集测试集地址
        :param bert_model: bert模型，默认为bert-base-chinese
        :param task_name: 任务名，用于选择处理方法
        :param output_dir: 输出地址，用于存储模型及其他文件
        :param model_save_path: 模型存储地址，位于输出地址内
        :param max_seq_length: 句子（单句、双句）的最大长度，过长要求更大显存，默认128左右
        :param do_train: 是否训练，默认TRUE
        :param do_eval: 是否验证，默认TRUE
        :param train_batch_size:训练batch，默认64
        :param eval_batch_size: 验证batch，默认1
        :param learning_rate: 学习率，默认5e-5
        :param num_train_epochs: 训练轮数，默认5次
        :param no_cuda: 是否使用cuda，默认False，即使用
        :param local_rank: 本地缓存，默认-1
        其他默认参数：
        do_lower_case:是否小写，对中文无影响
        warmup_proportion：进行线性学习率热身的训练率
        seed:初始化时随机的种子数，默认777
        gradient_accumulation_steps:反向更新前的累积步数
        optimize_on_cpu:是否在CPU上优化并保存，默认False
        fp16:是否使用16位浮点数精度加速计算，但会影响精度，一般默认False
        loss_scale:损失缩放，使用2的幂可以提升16位精度的收敛性
        """
        self.data_dir = data_dir
        self.bert_model = bert_model
        self.task_name = task_name
        self.output_dir = output_dir
        self.model_save_path = model_save_path
        self.max_seq_length = max_seq_length
        self.do_train = do_train
        self.do_eval = do_eval
        self.do_lower_case = False
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.warmup_proportion = 0.1
        self.no_cuda = no_cuda
        self.local_rank = local_rank
        self.seed = 777
        self.gradient_accumulation_steps = 1
        self.optimize_on_cpu = False
        self.fp16 = False
        self.loss_scale = 128

    def set_args(self):
        args = {'data_dir': self.data_dir,
                'bert_model': self.bert_model,
                'task_name': self.task_name,
                'output_dir': self.output_dir,
                'model_save_path': self.model_save_path,
                'max_seq_length': self.max_seq_length,
                'do_train': self.do_train,
                'do_eval': self.do_eval,
                'do_lower_case': self.do_lower_case,
                'train_batch_size': self.train_batch_size,
                'eval_batch_size': self.eval_batch_size,
                'learning_rate': self.learning_rate,
                'num_train_epochs': self.num_train_epochs,
                'warmup_proportion': self.warmup_proportion,
                'no_cuda': self.no_cuda,
                'local_rank': self.local_rank,
                'seed': self.seed,
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'optimize_on_cpu': self.optimize_on_cpu,
                'fp16': self.fp16,
                'loss_scale': self.loss_scale}

        print(type(args))
        return args
