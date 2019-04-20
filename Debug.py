# coding=utf-8

"""
author: 王黎成
function: 用于运行控制模块的调试
"""

# 引入外部库
import pandas as pd
import json
import numpy as np
import time

# 引入内部库
from Run import *
from UtilArea.Csv2Json import *


def test_get_answer ():
	init_system()
	user_query_list = [input()]
	time_start = time.time()
	real_query_list, answer_list = get_answer(user_query_list, model_name='AttentionDSSM', top_k=5)
	for i in range(len(user_query_list)):
		print('用户问题:' + user_query_list[i])
		for j in range(len(answer_list[i])):
			print('实际问题' + str(j) + ':' + real_query_list[i][j])
			print('答案' + str(j) + ':' + answer_list[i][j])
		print('--------------------------------')
	time_end = time.time()
	print('totally cost', time_end - time_start)


def test_run ():
	init_system()
	run()


test_get_answer()
