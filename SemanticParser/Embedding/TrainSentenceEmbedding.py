# -*- coding: utf-8 -*-
# @Author  : Lone Ranger
# @Function :

import json
import os

import numpy as np
# 导入外部包
import pandas as pd
from bert_serving.client import BertClient


class GetSentenceEmbedding:

    def __init__(self, filepath, title, save_path):
        self.filepath = filepath
        self.title = title
        self.save_path = save_path

    def get_text(self, filepath, title):
        """
        :param filepath:FAQ问答对的文件地址
        :param title: 文本内容对应的title
        :return: 文本内容的dataframe格式
        """
        print('Starting Get Text From File')
        dataframe = pd.read_csv(filepath)
        content = dataframe[title]
        print('Finishing Get Text From File')
        return content

    def df2list(self, dataframe):
        """
        :param dataframe:文本内容对应的dataframe
        :return: dataframe对应的list存储
        """
        print('Starting Convert DataFrame To List' + '\n')
        temp = np.array(dataframe)
        res = temp.tolist()
        print('Finishing Convert DataFrame To List' + '\n')
        return res

    def get_sentence_embedding(self, sentences):
        """
        :param sentences:用于编码为句向量的句子
        :return: 句向量
        """
        print('Starting Convert Sentences To Embedding' + '\n')
        try:
            bc = BertClient(ip='221.226.81.54', port=5555, port_out=5556)
        except (Exception) as e:
            print(e)
            print("Cannot Connect To Bert-Server" + '\n')
        print('%d questions loaded, avg. len of %d' % (
            len(sentences), np.mean([len(sentence) for sentence in sentences])) + '\n')
        sentences_vec = bc.encode(sentences)
        print('Finishing Convert Sentences To Embedding' + '\n')
        return sentences_vec

    def save_sentence_embedding(self, save_path, save_name, sentences_embedding):
        """
        :param save_path: 句向量的存储地址
        :param save_name: 句向量的存储名
        :return: 是否存储成功
        """
        print('Starting Saving Embedding' + '\n')
        if os.path.exists(save_path):
            pass
        else:
            os.mkdir(save_path)
        flag = False
        dic = {}
        try:
            for index, embedding in enumerate(sentences_embedding):
                dic[index] = embedding.tolist()
            json_embedding = json.dumps(dic)
            with open(save_path + '\\' + save_name, 'w', encoding='utf-8') as f:
                # json.dump(sentences_embedding, f)
                f.write(json_embedding)
                f.write('\n')
            return flag
        except(Exception) as e:
            print(e)
        print('Finishing Saving Embedding' + '\n')
        return flag
