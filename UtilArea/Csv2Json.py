# -*- coding: utf-8 -*-
# @Author  : Lone Ranger
# @Function :将原始的csv文件转为json文件存储，用作输入

import json
import os

import pandas as pd


class Csv2Json():
    def __init__(self, read_path, write_path):
        self.read_path = read_path
        self.write_path = write_path

    def read_csv(self):
        dataframe = pd.read_csv(self.read_path, encoding='utf-8')
        print('\t'.join(dataframe.columns))
        dataframe = dataframe.reset_index()
        new_df = dataframe.drop(axis=1, columns=[dataframe.columns[1]], inplace=False)
        # print(dataframe)
        dictionary = new_df.to_dict(orient='records')
        return dictionary

    def write_json(self, dic):
        try:
            with open(self.write_path, 'w', encoding='utf-8') as f:
                for content in dic:
                    json_data = json.dumps(content, ensure_ascii=False)
                    f.write(json_data)
                    f.write('\n')
        except (Exception) as e:
            print(e)

    def csv_to_json(self):
        df = pd.read_csv(self.read_path, engine='python', encoding='utf-8')
        with open(self.write_path, 'w', encoding='utf-8') as file_object:
            temp_dict = []
            key_list = []
            for key in df:
                key_list.append(key)
            key_list.pop(0)

            for i in range(len(df[key_list[0]])):
                row_dict = {}
                row_dict["index"] = i
                for key in key_list:
                    row_dict[key] = df[key][i].replace('\n', "").replace(' ', "")
                temp_dict.append(row_dict)
            json.dump(temp_dict, file_object, ensure_ascii=False, indent=2)