# coding=utf-8

"""
author: 王黎成
function: 调用SimNet中的语义相似度模型进行计算
"""

# 引入外部库
import json

# 引入内部库
from KnowledgeMatching.SimNet.DSSM.MultiGruDSSM import *
from KnowledgeMatching.SimNet.DSSM.AttentionDSSM import *
from KnowledgeMatching.SimNet.DSSM.TransformerDSSM import *
from UtilArea import GlobalVariable

# 全局变量
dssm_model = {'MultiGruModel': MultiGruDSSM, 'AttentionDSSM': AttentionDSSM, 'TransformerDSSM':TransformerDSSM}


def dssm_model_train (model_name='MultiGruModel'):
	"""
	dssm模型训练函数，从指定路径加载数据
	:return: None
	"""
	# 训练数据获取
	query_set = []
	answer_set = []
	faq_dict = GlobalVariable.get_value('FAQ_DATA')
	for key in faq_dict:
		query_set.append(list(faq_dict[key]['问题']))
		answer_set.append(list(faq_dict[key]['答案']))

	# 翻转问题，增加数据多样性，变成两个问题指向同一答案
	for key in faq_dict:
		query_set.append(list(faq_dict[key]['问题'])[::-1])
		answer_set.append(list(faq_dict[key]['答案']))

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


def dssm_model_infer (queries, model_name='MultiGruModel', top_k=1, threshold=0., query_type='所有'):
	"""
	dssm模型计算函数，通过参数获取问题，从指定路径加载需要匹配数据, 获取top-k个候选答案
	并根据给定阈值过滤答案
	:param queries: 问题列表
	:param model_name: 调用的模型名称
	:param top_k: 候选答案数目
	:param threshold: 相似度阈值
	:param query_type: 要匹配的问题类型
	:return: 包含答案ID的二维数组
	"""
	# 问题格式转换
	query_set = []
	for query in queries:
		query_set.append(list(query))

	# 匹配数据索引获取
	index_list = GlobalVariable.get_value('FAQ_INDEX')[query_type]

	# 匹配数据对应特征向量的获取
	t_set = []
	faq_data = GlobalVariable.get_value('FAQ_DATA')
	for index in index_list:
		t_set.append(faq_data[index]['embedding'])

	# 模型计算
	dssm = GlobalVariable.get_value('MODEL')['DSSM'][model_name + '_INFER']
	dssm.q_set = query_set
	dssm.t_set = t_set
	dssm.init_model_parameters()
	dssm.generate_data_set()
	result_prob_list, result_id_list = dssm.inference(top_k)
	answer_id_list = []
	for i in range(len(result_id_list)):
		answer_id = []
		for j in range(len(result_id_list[i])):
			if result_prob_list[i][j] <= threshold:
				break
			answer_id.append(index_list[result_id_list[i][j]])
		answer_id_list.append(answer_id)

	return answer_id_list


def dssm_model_extract_t_pre (model_name='MultiGruModel'):
	# 匹配数据获取
	query_dict = {'Domain': [], 'Encyclopedia': [], 'Gossip': []}
	faq_dict = GlobalVariable.get_value('FAQ_DATA')
	for key in faq_dict:
		if faq_dict[key]['专业'] == '百科':
			query_dict['Encyclopedia'].append(list(faq_dict[key]['问题']))
		elif faq_dict[key]['专业'] == '闲聊':
			query_dict['Gossip'].append(list(faq_dict[key]['问题']))
		else:
			query_dict['Domain'].append(list(faq_dict[key]['问题']))

	# 字向量字典获取
	embedding_dict = GlobalVariable.get_value('Word2Vec_CHARACTER_EMBEDDING')
	word_dict = {}
	vec_set = []
	i = 0
	for key in embedding_dict:
		word_dict[key] = i
		vec_set.append(embedding_dict[key][0])
		i += 1

	for key in query_dict:
		# 模型计算
		dssm = dssm_model[model_name](t_set=query_dict[key], dict_set=word_dict, vec_set=vec_set, is_extract=True)
		dssm.init_model_parameters()
		dssm.generate_data_set()
		dssm.build_graph()
		t_state = dssm.extract_t_pre()

		# 匹配数据对应特征向量的存储
		t_pre_dict = {}
		for i in range(len(t_state)):
			t_pre_dict[i] = list(map(float, list(t_state[i])))

		with open('./KnowledgeMemory/Embedding/DSSM/AttentionDSSM/' + key + 'Embedding.json', 'w',
		          encoding='utf-8') as file_object:
			json.dump(t_pre_dict, file_object, ensure_ascii=False, indent=2)
