# coding=utf-8

"""
author: 王黎成
function: 对话管理模块的主类，作为对话管理模块的接口
"""

# 引入外部库


# 引入内部库
from DialogueManagement.DST import *
from DialogueManagement.DialogPolicy.FSM import *


class DM:
	def __init__ (self, user_id):
		# 用户唯一标识ID
		self.user_id = user_id

		# 用户会话过程记录
		self.dialog_tree = DialogTree.DialogTree()

		# 用户对话策略
		self.dialog_policy = FAQGuiding.FAQGuiding()

	def create_dialog (self):
		# 问题输入
		print("您好，有什么能帮助您的？")
		init_sentence = input()

		# 新增一个对话场景，并将初始句加入相应对话场景中
		self.dialog_tree.add_dialog_branch(init_sentence, 'FAQ')

		# 开始对话决策过程
		self.dialog_policy.input_query(init_sentence)