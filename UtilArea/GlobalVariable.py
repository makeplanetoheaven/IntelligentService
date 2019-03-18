# coding=utf-8

"""
author: 王黎成
function: 全局变量使用模块
"""


# 引入外部库
import json


# 引入内部库

def _init ():  # 初始化
	global _global_dict
	_global_dict = {}

	print('loading dir FAQ_PATH')
	_global_dict['FAQ_PATH'] = './KnowledgeMemory/FAQ/FAQ.json'

	print('loading dir BERT_CHARACTER_EMBEDDING_PATH')
	_global_dict['BERT_CHARACTER_EMBEDDING_PATH'] = './KnowledgeMemory/Embedding/BERT/CharactersEmbedding.json'

	print('loading data FAQ_DATA')
	with open('./KnowledgeMemory/FAQ/FAQ.json', 'r', encoding='utf-8') as file_object:
		_global_dict['FAQ_DATA'] = json.load(file_object)

	print('loading data Word2Vec_CHARACTER_EMBEDDING')
	with open('./KnowledgeMemory/Embedding/Word2Vec/CharactersEmbedding.json', 'r', encoding='utf-8') as file_object:
		_global_dict['Word2Vec_CHARACTER_EMBEDDING'] = json.load(file_object)


def set_value (key, value):
	""" 定义一个全局变量 """
	_global_dict[key] = value


def get_value (key, def_value=None):
	try:
		return _global_dict[key]
	except KeyError:
		return def_value
