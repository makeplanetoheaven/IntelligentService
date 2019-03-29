# coding=utf-8

"""
author: 王黎成
function: 整个系统流程运行控制模块,系统唯一的对外接口
"""

# 引入外部库

# 引入内部库
from DialogueManagement.DM import *
from KnowledgeMatching.SemanticSimMatching import *
from UtilArea import GlobalVariable

# 全局变量


def init_system ():
	"""
	系统的初始化函数, 第一个需要运行的函数
	:return: NULL
	"""
	print('init global variable---------')
	GlobalVariable._init()


def get_answer (queries, model_name='MultiGruModel', top_k=1):
	"""
	不包含多轮对话，根据输入的多个问题，到指定模型中获取每个问题对应的前k个答案
	:param queries: 问题列表
	:param model_name: 调用模型名字
	:param top_k: 返回的问题数
	:return: 实际答案和问题二维数组
	"""
	print('get answer---------')
	# 调用模型计算，获取每一个问题对应top-k个答案ID
	answer_id_list = dssm_model_infer(queries, model_name=model_name, top_k=top_k)

	# 数据加载
	faq_dict = GlobalVariable.get_value('FAQ_DATA')

	# 获取指定ID问题
	query_set = []
	for answer in answer_id_list:
		query_list = []
		for id in answer:
			query_list.append(faq_dict[id]["问题"])
		query_set.append(query_list)

	# 获取指定ID答案
	answer_set = []
	for answer in answer_id_list:
		answer_list = []
		for id in answer:
			answer_list.append(faq_dict[id]["答案"])
		answer_set.append(answer_list)

	return query_set, answer_set


def run ():
	"""
	整个系统运行控制函数
	:return:
	"""
	# 一个用户维护一个对话管理模块
	user_id = 0
	dm = DM(user_id)
	dm.create_dialog()


def set_input_interface(func):
	"""
	修改输入接口的函数
	:param func: 函数无形参，有返回值
	:return:
	"""
	GlobalVariable.set_value('INPUT', func)


def set_output_interface(func):
	"""
	修改输出接口的函数
	:param func: 函数有形参，无返回值
	:return:
	"""
	GlobalVariable.set_value('OUPUT', func)
