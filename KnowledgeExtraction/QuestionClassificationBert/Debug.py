# -*- coding: utf-8 -*-
# @Author  : Lone Ranger
# @Function :


from KnowledgeExtraction.QuestionClassificationBert.Args import BertArgs
from KnowledgeExtraction.QuestionClassificationBert.TrainClassificationModel import BertForClassification
from UtilArea import GlobalVariable

if __name__ == '__main__':

    do_train = False
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
    # TODO：（1）改为意图识别（2）规范接口对接
    # 输入一句话进行问题分类或意图识别：
    GlobalVariable._init()
    PredictModel = GlobalVariable.get_value('QUESTION_CLASSIFICATION_MODEL')
    input_sentence = input('请输入问题：')
    res = PredictModel.test(input_sentence)

    # test(model, processor, args, label_list, tokenizer, device, input_sentence)
