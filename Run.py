# coding=utf-8

"""
author: 王黎成
function: 整个系统流程运行控制模块
"""

# 引入外部库
import json

# 引入内部库
from KnowledgeMatching.SemanticSimMatching import dssm_model_infer
from UtilArea import GlobalVariable


def init_system ():
	"""
	系统的初始化函数
	:return:
	"""
	print('init global variable---------')
	GlobalVariable._init()


def get_answer (queries, top_k=1):
	"""
	根据输入的多个问题，获取每个问题对应的前k个答案
	:param queries: 问题列表
	:param top_k: 候选答案数
	:return: 二维数组
	"""
	print('get answer---------')
	# 调用模型计算，获取每一个问题对应top-k个答案ID
	answer_id_list = dssm_model_infer(queries, top_k)

	# 数据加载并获取指定ID答案
	answer_set = []
	faq_dict = GlobalVariable.get_value('FAQ_DATA')
	for answer in answer_id_list:
		answer_list = []
		for id in answer:
			answer_list.append(faq_dict[id]["答案"])
		answer_set.append(answer_list)

	return answer_set
