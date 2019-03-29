# coding=utf-8

"""
author: 王黎成
function: 用于运行控制模块的调试
"""

# 引入外部库
import pandas as pd
import json
import numpy as np

# 引入内部库
from Run import *


def test_get_answer ():
	init_system()
	user_query_list = ["处理工程项目退库业务时候，找不到项目怎么办呀,请尽快反馈我", "发票真伪结果显示无法查到该发票，在影像发票校验的时候", '“提交”按钮置灰，企业信息维护后', '为啥看不到采购公告']
	real_query_list, answer_list = get_answer(user_query_list, model_name='AttentionDSSM', top_k=5)
	for i in range(len(user_query_list)):
		print('用户问题:' + user_query_list[i])
		for j in range(len(answer_list[i])):
			print('实际问题' + str(j) + ':' + real_query_list[i][j])
			print('答案' + str(j) + ':' + answer_list[i][j])
		print('--------------------------------')


def test_run ():
	init_system()
	run()


test_run()
