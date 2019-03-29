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
