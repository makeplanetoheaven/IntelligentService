# -*- coding: utf-8 -*-
# @Author  : Lone Ranger
# @Function :
from collections import OrderedDict

import torch
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification

from KnowledgeExtraction.QuestionClassificationBert.Args import BertArgs
from KnowledgeExtraction.QuestionClassificationBert.Preprocess import sentencePro
from KnowledgeExtraction.QuestionClassificationBert.SentencePredict import test
from KnowledgeExtraction.QuestionClassificationBert.TrainClassificationModel import BertForClassification

if __name__ == '__main__':

    do_train = True
    if do_train:
        #  1.训练模型
        #  设置训练模式时的Bert模型参数
        train_args = BertArgs(do_train=True, do_eval=True, no_cuda=False)
        #  加载训练类
        train_classification = BertForClassification(train_args)
        #  训练

        print('---------Start Training-------------')
        train_classification.train()
        print('---------Finish Training------------')

    #  2.加载训练好的模型进行预测
    #  设置预测时的processor、label_list
    processors = {'sentencePro': sentencePro}
    processor = processors['sentencePro']()
    label_list = processor.get_labels()
    # 设置预测时的Bert模型参数
    test_args = BertArgs(task_name=processor)
    args = test_args.set_args()
    # 根据args获取对应的device、tokenizer、model
    model_path = args.get('model_save_path')
    device = torch.device("cuda" if torch.cuda.is_available() and not args.get('no_cuda') else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.get('bert_model'))
    model = BertForSequenceClassification.from_pretrained(args.get('bert_model'))
    # 加载已有的模型参数到cpu
    state_dict = torch.load(args.get('model_save_path'), map_location='cpu')['state_dict']

    # 创建一个新的state_dict加载预训练的模型参数进行预测
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove module.
        new_state_dict[name] = v
    # 获取真正的权重矩阵
    model.load_state_dict(new_state_dict)
    # 测试，输入一句话判断是否是问题
    # TODO：（1）改为意图识别（2）规范接口对接
    # 输入一句话进行问题分类或意图识别：
    input_sentence = input()
    test(model, processor, args, label_list, tokenizer, device, input_sentence)
