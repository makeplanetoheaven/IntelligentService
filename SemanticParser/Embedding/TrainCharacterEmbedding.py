# -*- coding: utf-8 -*-
# @Author  : Lone Ranger
# @Function : 训练每个字的字向量


import json
import os
import re

import numpy as np
import pandas as pd
from bert_serving.client import BertClient


class TrainCharacterEmbedding:
    def __init__(self, read_path, write_path):
        """
        :param read_path:读取json地址
        :param write_path: 写入字向量地址
        """
        self.read_path = read_path
        self.write_path = write_path

    def get_text(self):
        """
        获取文本内容，这里是获取问题及答案，转为dataframe格式返回
        :return:
        """
        with open(self.read_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # print(temp)
        question = []
        answer = []
        for i in data:
            question.append(i['问题'])
            answer.append(i['答案'])
        res = {"question": question,
               "answer": answer}
        dataframe = pd.DataFrame(res)
        return dataframe

    def split_character(self, dataframe):
        """
        :param dataframe:存储需要分词的文本的dataframe
        :return: 经过处理得到的包含每个字的list，用于输入bert模型获取字向量
        """
        # 用正则表达式过滤一下非空字符
        re_pattern = '\S'

        questions = dataframe['question']
        answers = dataframe['answer']

        all_contents = pd.concat([questions, answers])
        # 将dataframe格式转为ndarray格式
        all_contents = np.array(all_contents)
        all_characters = []
        # 对文本进行分字，使用list()实现
        for content in all_contents:
            content_character = list(content)
            for chac in content_character:
                if re.match(re_pattern, chac):
                    all_characters.append(chac)
        # 使用set去重，最后转为list
        characters_in_total = list(set(all_characters))
        return characters_in_total

    def train_character_embedding(self, character_list):
        """
        :param character_list: 包含所有字的list
        :return: 每个字对应的index，character,embedding格式的list
        """
        character_embedding_list = []
        re_pattern = '\S'
        bc = BertClient('221.226.81.54')
        index = [i for i in range(len(character_list))]
        # 获取每个非空字的字向量
        for character in character_list:
            if re.match(re_pattern, character):
                character_embedding = bc.encode(list(str(character)))
                character_embedding = np.array(character_embedding).tolist()
                character_embedding_list.append(character_embedding)
        # 将每个字对应的index，character，embedding作为一个对象加入到字典中，再将字典转为list，用于使用json格式存储
        em_list = []
        for i in range(len(index)):
            temp_dict = {"index": index[i], 'word': character_list[i], 'embedding': character_embedding_list[i]}
            em_list.append(temp_dict)

        return em_list

    def save_character_embedding(self, characters_embedding, save_path, filename):
        """
        :param characters_embedding:需要存储的字向量list
        :param save_path: 保存地址
        :param filename: 保存文件名
        :return: 是否保存成功
        """
        try:
            if os.path.exists(save_path):
                pass
            else:
                os.mkdir(save_path)
            with open(save_path + filename, 'w', encoding='utf-8') as f:
                json.dump(characters_embedding, f, ensure_ascii=False, indent=2)
                # f.write(characters_embedding)
                # f.write('\n')
            return True
        except Exception as e:
            print(e)
            return False
