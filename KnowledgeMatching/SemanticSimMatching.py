# coding=utf-8

"""
author: 王黎成
function: 调用SimNet中的语义相似度模型进行计算
"""

# 引入外部库
import json

# 引入内部库
from KnowledgeMatching.SimNet.DSSM.LstmDSSM import *

# 全局变量
embedding_path = './KnowledgeMemory/Embedding/.json'
faq_path = './KnowledgeMemory/FAQ/FAQ.json'


def dssm_model_train():
	"""
	dssm模型训练函数，从指定路径加载数据
	:return: None
	"""
	# 训练数据获取
	query_set = []
	answer_set = []
	with open(faq_path, 'r', encoding='utf-8') as file_object:
		faq_dict = json.load(file_object)
		for element in faq_dict:
			query_set.append(list(element['问题']))
			answer_set.append(list(element['答案']))

	# 词向量字典获取
	word_set = []
	embedding_set = []
	with open(embedding_path, 'r', encoding='utf-8') as file_object:
		embedding_dict = json.load(file_object)
		for element in embedding_dict:
			word_set.append(element)
			embedding_set.append(element)

	# 模型训练
	dssm = LstmDSSM(q_set=query_set,t_set=answer_set,dict_set=word_set,vec_set=embedding_set)
	# dssm.init_model_parameters()
	# dssm.generate_data_set()
	# dssm.build_graph()
	# dssm.train()