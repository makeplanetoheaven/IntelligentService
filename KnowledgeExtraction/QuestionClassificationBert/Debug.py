# -*- coding: utf-8 -*-
# @Author  : Lone Ranger
# @Function :
from collections import OrderedDict

import torch
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification

from KnowledgeExtraction.QuestionClassificationBert.Args import BertArgs
from KnowledgeExtraction.QuestionClassificationBert.Preprocess import sentencePro
from KnowledgeExtraction.QuestionClassificationBert.SentencePredict import test

if __name__ == '__main__':

    # 模型路径、processor、label_list
    model_path = './checkpoints/bert_classification.pth'
    processors = {'sentencePro': sentencePro}
    processor = processors['sentencePro']()
    label_list = processor.get_labels()
    # 调用BertArgs设置相关参数，根据任务不同选择设置train/test
    bert_args = BertArgs(task_name=processor)
    args = bert_args.set_args()
    # 根据args获取对应的device、tokenizer、model
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
    test(model, processor, args, label_list, tokenizer, device)
