# coding=utf-8

"""
author: 王黎成
function: 调用SimNet中的语义相似度模型进行计算
"""

# 引入外部库

# 引入内部库
from KnowledgeMatching.SimNet.DSSM.MultiGruDSSM import *
from KnowledgeMatching.SimNet.DSSM.AttentionDSSM import *
from UtilArea import GlobalVariable


# 全局变量
dssm_model = {'MultiGruModel': MultiGruDSSM, 'AttentionDSSM': AttentionDSSM}


def dssm_model_train (model_name='MultiGruModel'):
	"""
	dssm模型训练函数，从指定路径加载数据
	:return: None
	"""
	# 训练数据获取
	query_set = []
	answer_set = []
	faq_list = GlobalVariable.get_value('FAQ_DATA')
	for faq_dict in faq_list:
		query_set.append(list(faq_dict['问题']))
		answer_set.append(list(faq_dict['答案']))

	# 翻转问题，增加数据多样性，变成两个问题指向同一答案
	for faq_dict in faq_list:
		query_set.append(list(faq_dict['问题'])[::-1])
		answer_set.append(list(faq_dict['答案']))

	# 字向量字典获取
	embedding_dict = GlobalVariable.get_value('Word2Vec_CHARACTER_EMBEDDING')
	word_dict = {}
	vec_set = []
	i = 0
	for key in embedding_dict:
		word_dict[key] = i
		vec_set.append(embedding_dict[key][0])
		i += 1

	# 模型训练
	dssm = dssm_model[model_name](q_set=query_set, t_set=answer_set, dict_set=word_dict, vec_set=vec_set,
	                              batch_size=len(query_set) // 2)
	dssm.init_model_parameters()
	dssm.generate_data_set()
	dssm.build_graph()
	dssm.train()


def dssm_model_infer (queries, model_name='MultiGruModel', top_k=1):
	"""
	dssm模型计算函数，通过参数获取问题，从指定路径加载需要匹配数据, 获取top-k个候选答案
	:param queries: list型问题集
	:return: 二维数组，每一个元素对应一个问题top-k候选ID
	"""
	# 问题格式转换
	query_set = []
	for query in queries:
		query_set.append(list(query))

	# 匹配数据获取
	answer_set = []
	faq_dict = GlobalVariable.get_value('FAQ_DATA')
	for element in faq_dict:
		answer_set.append(list(element['问题']))

	# 字向量字典获取
	embedding_dict = GlobalVariable.get_value('Word2Vec_CHARACTER_EMBEDDING')
	word_dict = {}
	vec_set = []
	i = 0
	for key in embedding_dict:
		word_dict[key] = i
		vec_set.append(embedding_dict[key][0])
		i += 1

	# 模型计算
	dssm = dssm_model[model_name](q_set=query_set, t_set=answer_set, dict_set=word_dict, vec_set=vec_set, is_train=False, top_k=top_k)
	dssm.init_model_parameters()
	dssm.generate_data_set()
	dssm.build_graph()

	return dssm.inference()
