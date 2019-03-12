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
	print(get_answer(["帐号密码不正确", "供应商账户信息找不到开户银行"]))

test_get_answer()
