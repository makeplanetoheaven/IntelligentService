# coding=utf-8

"""
author: 王黎成
function: 用于运行控制模块的调试
"""

# 引入外部库
import pandas as pd
import json

# 引入内部库
from Run import *


def test_get_answer():
	init_system()
	print(get_answer(["营收数据重复了"], top_k=1))

test_get_answer()
