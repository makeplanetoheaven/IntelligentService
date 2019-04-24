# coding=utf-8

"""
author: 王黎成
function: 基于有限状态机（FSM）的faq对话决策
"""

# 引入外部库
from transitions import Machine

# 引入内部库
from KnowledgeMatching.SemanticSimMatching import *
from UtilArea import GlobalVariable


class FAQGuiding:
	states = ['START', 'END', 'Query Detecting', 'Answer Matching', 'Type Provide', 'Types Selecting',
	          'Candidate Selecting']

	def __init__ (self):
		# FSM初始化
		self.machine = Machine(model=self, states=FAQGuiding.states, initial='START')

		# 状态转移添加
		self.machine.add_transition(trigger='input_query', source='START', dest='Query Detecting',
		                            after='query_detecting')
		self.machine.add_transition(trigger='correct', source='Query Detecting', dest='Answer Matching',
		                            after='answer_matching')
		self.machine.add_transition(trigger='error', source='Query Detecting', dest='Query Detecting',
		                            after='query_detecting')
		self.machine.add_transition(trigger='correct', source='Answer Matching', dest='END', after='end_process')
		self.machine.add_transition(trigger='error', source='Answer Matching', dest='Type Provide',
		                            after='type_provide')
		self.machine.add_transition(trigger='correct', source='Type Provide', dest='Candidate Selecting',
		                            after='candidate_selecting')
		self.machine.add_transition(trigger='error', source='Type Provide', dest='Types Selecting',
		                            after='types_selecting')
		self.machine.add_transition(trigger='selecting', source='Candidate Selecting', dest='END', after='end_process')
		self.machine.add_transition(trigger='error', source='Candidate Selecting', dest='Types Selecting',
		                            after='types_selecting')
		self.machine.add_transition(trigger='selecting', source='Types Selecting', dest='Candidate Selecting',
		                            after='candidate_selecting')

		self.machine.add_transition(trigger='quit', source='*', dest='END', after='end_process')

	def query_detecting (self, query):
		self.correct(query)

	def answer_matching (self, query):
		user_query_list = [query]

		# 调用模型计算，获取问题对应答案ID
		answer_id_list = dssm_model_infer(user_query_list, model_name='TransformerDSSM', top_k=1)

		# 数据加载
		faq_dict = GlobalVariable.get_value('FAQ_DATA')

		# 输出指定ID答案
		GlobalVariable.get_value('OUTPUT')(faq_dict[answer_id_list[0][0]]["答案"])

		# 是否正确的判断
		GlobalVariable.get_value('OUTPUT')('是否是正确答案？（是/否）')
		if GlobalVariable.get_value('INPUT')() == '是':
			self.correct()
		else:
			self.error(query, answer_id_list[0][0])

	def type_provide (self, query, answer_id):
		# 数据加载
		faq_dict = GlobalVariable.get_value('FAQ_DATA')

		# 问题类型判断
		query_type = faq_dict[answer_id]["专业"]
		GlobalVariable.get_value('OUTPUT')('您想询问的是否是[%s]相关问题？（是/否）' % query_type)
		if GlobalVariable.get_value('INPUT')() == '是':
			self.correct(query, query_type)
		else:
			self.error(query)

	def candidate_selecting (self, query, query_type):
		user_query_list = [query]

		# 调用模型计算，获取指定类型下，问题对应候选5个答案ID
		answer_id_list = dssm_model_infer(user_query_list, model_name='TransformerDSSM', top_k=5, query_type=query_type)

		# 数据加载
		faq_dict = GlobalVariable.get_value('FAQ_DATA')

		# 输出指定ID问题
		for index in range(len(answer_id_list[0])):
			GlobalVariable.get_value('OUTPUT')(str(index+1)+'.'+faq_dict[answer_id_list[0][index]]["问题"])

		# 候选问题选择
		GlobalVariable.get_value('OUTPUT')('上述问题是否包含您想问的问题，如果是，请返回相应问题序号，如果不是，请回[否]')
		respond = GlobalVariable.get_value('INPUT')()
		if respond.isdigit():
			GlobalVariable.get_value('OUTPUT')(faq_dict[answer_id_list[0][int(respond)-1]]["答案"])
			self.selecting()
		else:
			self.error(query)

	def types_selecting (self, query):
		# 获取问题类型
		query_types = []
		faq_list = GlobalVariable.get_value('FAQ_DATA')
		for faq_dict in faq_list:
			if faq_dict['专业'] not in query_types:
				query_types.append(faq_dict['专业'])

		# 输出问题类型
		GlobalVariable.get_value('OUTPUT')('目前已有的问题类型有：')
		for index in range(len(query_types)):
			GlobalVariable.get_value('OUTPUT')(str(index+1)+'.'+query_types[index])

		# 问题类型选择
		GlobalVariable.get_value('OUTPUT')('上述类型是否包含您想问的问题类型，如果是，请返回相应类型序号，如果不是，请回[否]')
		respond = GlobalVariable.get_value('INPUT')()
		if respond.isdigit():
			query_type = query_types[int(respond) - 1]
			self.selecting(query, query_type)
		else:
			self.error(query)

	def end_process (self):
		GlobalVariable.get_value('OUTPUT')('感谢为您解答！')
