# coding=utf-8

"""
author: 王黎成
function: 全局变量使用模块
"""

# 引入外部库
import json

# 引入内部库
from KnowledgeExtraction.QuestionClassificationBert import SentencePredict
from KnowledgeExtraction.QuestionClassificationBert.Args import BertArgs
from KnowledgeMatching.SimNet.DSSM.AttentionDSSM import *
# 全局变量
from UtilArea.ClassificationModelParameters import load_parameters_to_cpu

__global_dict = None


def _init ():
	"""
	全局变量的初始化
	:return:
	"""
	global __global_dict
	__global_dict = {}

	# <editor-fold desc="Dir">
	print('loading Dir FAQ_PATH')
	__global_dict['FAQ_PATH'] = './KnowledgeMemory/FAQ/FAQ.json'

	print('loading Dir DSSM_FAQ_EMBEDDING_PATH')
	__global_dict['DSSM_FAQ_EMBEDDING_PATH'] = {
		'MultiGruModel': './KnowledgeMemory/Embedding/DSSM/MultiGruDSSM/SentenceEmbedding.json',
		'AttentionDSSM': './KnowledgeMemory/Embedding/DSSM/AttentionDSSM/SentenceEmbedding.json'}

	print('loading Dir BERT_CHARACTER_EMBEDDING_PATH')
	__global_dict['BERT_CHARACTER_EMBEDDING_PATH'] = './KnowledgeMemory/Embedding/BERT/CharactersEmbedding.json'
	# </editor-fold>
	# <editor-fold desc="Data">
	print('loading BERT_ARGS')
	bert_args = BertArgs(task_name='SentencePro')
	__global_dict['BERT_ARGS'] = bert_args.set_args()
	
	print('loading Data FAQ_DATA')
	with open('./KnowledgeMemory/FAQ/FAQ.json', 'r', encoding='utf-8') as file_object:
		__global_dict['FAQ_DATA'] = json.load(file_object)

	print('loading Data Word2Vec_CHARACTER_EMBEDDING')
	with open('./KnowledgeMemory/Embedding/Word2Vec/CharactersEmbedding.json', 'r', encoding='utf-8') as file_object:
		__global_dict['Word2Vec_CHARACTER_EMBEDDING'] = json.load(file_object)

	print('loading Data DSSM_FAQ_SENTENCE_EMBEDDING')
	__global_dict['DSSM_FAQ_SENTENCE_EMBEDDING'] = {}
	with open('./KnowledgeMemory/Embedding/DSSM/AttentionDSSM/SentenceEmbedding.json', 'r',
	          encoding='utf-8') as file_object:
		__global_dict['DSSM_FAQ_SENTENCE_EMBEDDING']['AttentionDSSM'] = json.load(file_object)

	print('loading PRETRAINED_MODEL PARAMETERS')
	parameters = load_parameters_to_cpu(__global_dict.get('BERT_ARGS'))
	__global_dict['NEW_STATE_DICT'] = parameters.load()
	# </editor-fold>
	# <editor-fold desc="Model">
	print('loading Model DSSM')
	__global_dict['MODEL'] = {}
	# DSSM
	__global_dict['MODEL']['DSSM'] = {}

	print('loading QUESTION_CLASSIFICATION_MODEL')
	predict_model = SentencePredict.PredictModel()
	__global_dict['QUESTION_CLASSIFICATION_MODEL'] = predict_model

	# 字向量字典获取
	embedding_dict = __global_dict['Word2Vec_CHARACTER_EMBEDDING']
	word_dict = {}
	vec_set = []
	i = 0
	for key in embedding_dict:
		word_dict[key] = i
		vec_set.append(embedding_dict[key][0])
		i += 1
	__global_dict['MODEL']['DSSM']['AttentionDSSM_INFER'] = AttentionDSSM(dict_set=word_dict, vec_set=vec_set,
	                                                                     is_train=False)
	__global_dict['MODEL']['DSSM']['AttentionDSSM_INFER'].build_graph()
	# </editor-fold>
	# <editor-fold desc="Interface">
	print('loading Interface IO')
	__global_dict['OUTPUT'] = print
	__global_dict['INPUT'] = input
	# </editor-fold>

	pass


def set_value (key, value):
	"""
	全局变量的添加及修改
	:param key: 关键字
	:param value: 需要修改或添加的值
	:return:
	"""
	__global_dict[key] = value


def get_value (key, def_value=None):
	"""
	全局变量的获取
	:param key: 关键字
	:param def_value:默认值
	:return:
	"""
	try:
		return __global_dict[key]
	except KeyError:
		return def_value
