# coding=utf-8

"""
author: 王黎成
function: 整个系统流程运行控制模块
"""

# 引入外部库
import json

# 引入内部库
from KnowledgeMatching.SemanticSimMatching import dssm_model_infer

# 全局变量
faq_path = './KnowledgeMemory/FAQ/FAQ.json'


def get_answer(queries, top_k=2):
	"""
	根据输入的多个问题，获取每个问题对应的前k个答案
	:param queries: 问题列表
	:param top_k: 候选答案数
	:return: 二维数组
	"""
	# 调用模型计算，获取每一个问题对应top-k个答案ID
	answer_id_list = dssm_model_infer(queries, top_k)

	# 数据加载并获取指定ID答案
	answer_set = []
	with open(faq_path, 'r', encoding='utf-8') as file_object:
		faq_dict = json.load(file_object)
		for answer in answer_id_list:
			answer_list = []
			for id in answer:
				answer_list.append(faq_dict[id]["答案"])
			answer_set.append(answer_list)

	return answer_set
