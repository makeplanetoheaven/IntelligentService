# coding=utf-8

"""
author: 王黎成
function: 调用SimNet中的语义相似度模型进行计算
"""

# 引入外部库
import json

# 引入内部库
from KnowledgeMatching.SimNet.DSSM.GruDSSM import *
from UtilArea import GlobalVariable


# 全局变量


def dssm_model_train ():
	"""
	dssm模型训练函数，从指定路径加载数据
	:return: None
	"""
	# 训练数据获取
	query_set = []
	answer_set = []
	faq_dict = GlobalVariable.get_value('FAQ_DATA')
	for element in faq_dict:
		query_set.append(list(element['问题']))
		answer_set.append(list(element['答案']))

	# 词向量字典获取
	word_set = []
	embedding_set = []
	embedding_dict = GlobalVariable.get_value('CHARACTER_EMBEDDING')
	for element in embedding_dict:
		word_set.append(element["word"])
		embedding_set.append(element["embedding"][0])

	# 模型训练
	dssm = GruDSSM(q_set=query_set, t_set=answer_set, dict_set=word_set, vec_set=embedding_set)
	dssm.init_model_parameters()
	dssm.generate_data_set()
	dssm.build_graph()
	dssm.train()


def dssm_model_infer (queries, top_k=1):
	"""
	dssm模型计算函数，通过参数获取问题，从指定路径加载需要匹配数据, 获取top-k个候选答案
	:param queries: list型问题集
	:return: 二维数组，每一个元素对应一个问题top-k候选ID
	"""
	# 问题格式转换
	query_set = []
	for query in queries:
		query_set.append(list(query))

	# 匹配数据获取
	answer_set = []
	faq_dict = GlobalVariable.get_value('FAQ_DATA')
	for element in faq_dict:
		answer_set.append(list(element['问题']))

	# 词向量字典获取
	word_set = []
	embedding_set = []
	embedding_dict = GlobalVariable.get_value('CHARACTER_EMBEDDING')
	for element in embedding_dict:
		word_set.append(element["word"])
		embedding_set.append(element["embedding"][0])

	# 模型计算
	dssm = GruDSSM(q_set=query_set, t_set=answer_set, dict_set=word_set, vec_set=embedding_set, is_train=False,
	               top_k=top_k)
	dssm.init_model_parameters()
	dssm.generate_data_set()
	dssm.build_graph()

	return dssm.inference()
