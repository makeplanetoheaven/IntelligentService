# coding=utf-8

"""
author: 王黎成
function: 用于知识匹配模块的调试
"""

# 引入外部库
import pandas as pd
import json

# 引入内部库
from KnowledgeMatching.SemanticSimMatching import *
from Run import *
from UtilArea.Csv2Json import *


def test_semantic():
	init_system()
	dssm_model_train('TransformerDSSM')


test_semantic()
