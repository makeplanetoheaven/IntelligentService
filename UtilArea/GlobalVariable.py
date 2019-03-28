# coding=utf-8

"""
author: 王黎成
function: 全局变量使用模块
"""

# 引入外部库
import json

# 引入内部库
from KnowledgeMatching.SimNet.DSSM.AttentionDSSM import *


# 全局变量
_global_dict = None


def _init ():
	global _global_dict
	_global_dict = {}

	print('loading Dir FAQ_PATH')
	_global_dict['FAQ_PATH'] = './KnowledgeMemory/FAQ/FAQ.json'

	print('loading Dir DSSM_FAQ_EMBEDDING_PATH')
	_global_dict['DSSM_FAQ_EMBEDDING_PATH'] = {
		'MultiGruModel': './KnowledgeMemory/Embedding/DSSM/MultiGruDSSM/SentenceEmbedding.json',
		'AttentionDSSM': './KnowledgeMemory/Embedding/DSSM/AttentionDSSM/SentenceEmbedding.json'}

	print('loading Dir BERT_CHARACTER_EMBEDDING_PATH')
	_global_dict['BERT_CHARACTER_EMBEDDING_PATH'] = './KnowledgeMemory/Embedding/BERT/CharactersEmbedding.json'

	print('loading Data FAQ_DATA')
	with open('./KnowledgeMemory/FAQ/FAQ.json', 'r', encoding='utf-8') as file_object:
		_global_dict['FAQ_DATA'] = json.load(file_object)

	print('loading Data Word2Vec_CHARACTER_EMBEDDING')
	with open('./KnowledgeMemory/Embedding/Word2Vec/CharactersEmbedding.json', 'r', encoding='utf-8') as file_object:
		_global_dict['Word2Vec_CHARACTER_EMBEDDING'] = json.load(file_object)

	print('loading Data DSSM_FAQ_SENTENCE_EMBEDDING')
	_global_dict['DSSM_FAQ_SENTENCE_EMBEDDING'] = {}
	with open('./KnowledgeMemory/Embedding/DSSM/AttentionDSSM/SentenceEmbedding.json', 'r', encoding='utf-8') as file_object:
		_global_dict['DSSM_FAQ_SENTENCE_EMBEDDING']['AttentionDSSM'] = json.load(file_object)

	print('loading Model Database')
	_global_dict['MODEL'] = {}
	_global_dict['MODEL']['DSSM'] = {}

	# 字向量字典获取
	embedding_dict = _global_dict['Word2Vec_CHARACTER_EMBEDDING']
	word_dict = {}
	vec_set = []
	i = 0
	for key in embedding_dict:
		word_dict[key] = i
		vec_set.append(embedding_dict[key][0])
		i += 1
	_global_dict['MODEL']['DSSM']['AttentionDSSM_INFER'] = AttentionDSSM(dict_set=word_dict, vec_set=vec_set, is_train=False)
	_global_dict['MODEL']['DSSM']['AttentionDSSM_INFER'].build_graph()


def set_value (key, value):
	""" 定义一个全局变量 """
	_global_dict[key] = value


def get_value (key, def_value=None):
	try:
		return _global_dict[key]
	except KeyError:
		return def_value
