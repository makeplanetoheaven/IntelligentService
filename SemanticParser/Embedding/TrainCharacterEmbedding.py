# -*- coding: utf-8 -*-
# @Author  : Lone Ranger
# @Function : 训练每个字的字向量


import json
import os
import re

import numpy as np
import pandas as pd
from bert_serving.client import BertClient


class TrainCharacterEmbedding():
    def __init__(self, read_path, write_path):
        self.read_path = read_path
        self.write_path = write_path

    def get_text(self):
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
        # thu1 = thulac.thulac(seg_only=True)
        re_pattern = '\S'
        questions = dataframe['question']
        answers = dataframe['answer']
        all_contents = pd.concat([questions, answers])

        all_contents = np.array(all_contents)
        # questions = np.array(questions).tolist()
        # answers = np.array(answers).tolist()
        # question_charcters = []
        # answer_charcters = []
        all_characters = []

        # for question in questions:
        #     question_charcter = list(question)
        #     for chac in question_charcter:
        #         if re.match(re_pattern, chac):
        #             question_charcters.append(chac)
        #
        # for answer in answers:
        #     answer_charcter = list(answer)
        #     for chac in answer_charcter:
        #         if re.match(re_pattern, chac):
        #             answer_charcters.append(chac)
        for content in all_contents:
            content_charcter = list(content)
            for chac in content_charcter:
                if re.match(re_pattern, chac):
                    all_characters.append(chac)

        characters_in_totla = list(set(all_characters))
        # characters_in_answers = list(set(answer_charcters))
        # print(characters_in_answers)
        # print(characters_in_questions)
        return characters_in_totla

    def train_character_embedding(self, character_list):
        character_embedding_list = []
        re_pattern = '\S'
        bc = BertClient('221.226.81.54')
        index = [i for i in range(len(character_list))]
        # print(len(index))
        for character in character_list:
            if re.match(re_pattern, character):
                # print(character)
                character_embedding = bc.encode(list(str(character)))
                character_embedding = np.array(character_embedding).tolist()
                character_embedding_list.append(character_embedding)

        # print(len(character_embedding_list))
        print(character_list)
        # res = zip(index, zip(character_list, character_embedding_list))
        em_list = []
        for i in range(len(index)):
            temp_dict = {}
            temp_dict["index"] = index[i]
            temp_dict['word'] = character_list[i]
            temp_dict['embedding'] = character_embedding_list[i]
            em_list.append(temp_dict)
        print(em_list)
        return em_list

    def save_character_embedding(self, characters_embedding, save_path, filename):
        try:
            if os.path.exists(save_path):
                pass
            else:
                os.mkdir(save_path)
            with open(save_path + filename, 'w', encoding='utf-8') as f:
                json.dump(characters_embedding, f, ensure_ascii=False, indent=2)
                # f.write(characters_embedding)
                # f.write('\n')
        except (Exception) as e:
            print(e)
