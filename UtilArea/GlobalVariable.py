# coding=utf-8

"""
author: 王黎成
function: 全局变量使用模块
"""

# 引入外部库
import json

# 引入内部库
from KnowledgeMatching.SimNet.DSSM.TransformerDSSM import *

# 全局变量
__global_dict = None


def _init ():
	"""
	全局变量的初始化
	:return:
	"""
	global __global_dict
	__global_dict = {}

	# <editor-fold desc="Dir">
	print('loading Dir FAQ_PATH')
	__global_dict['FAQ_PATH'] = './KnowledgeMemory/FAQ/FAQ.json'

	print('loading Dir BERT_CHARACTER_EMBEDDING_PATH')
	__global_dict['BERT_CHARACTER_EMBEDDING_PATH'] = './KnowledgeMemory/Embedding/BERT/CharactersEmbedding.json'
	# </editor-fold>
	# <editor-fold desc="Data">
	print('loading Data FAQ_DATA')
	__global_dict['FAQ_DATA'] = {}

	index = 0
	index_dict = dict()
	# Domain
	index_dict['领域'] = []
	embedding_object = open('./KnowledgeMemory/Embedding/DSSM/TransformerDSSM/DomainEmbedding.json', 'r',
	                        encoding='utf-8')
	faq_object = open('./KnowledgeMemory/FAQ/DomainFAQ.json', 'r', encoding='utf-8')
	domain_embedding = json.load(embedding_object)
	domain_faq = json.load(faq_object)
	for i in range(len(domain_faq)):
		__global_dict['FAQ_DATA'][index] = {}
		__global_dict['FAQ_DATA'][index]['专业'] = domain_faq[i]['专业']
		if domain_faq[i]['专业'] in index_dict:
			index_dict[domain_faq[i]['专业']].append(index)
		else:
			index_dict[domain_faq[i]['专业']] = [index]
		if str(i) in domain_embedding:
			__global_dict['FAQ_DATA'][index]['embedding'] = domain_embedding[str(i)]
		__global_dict['FAQ_DATA'][index]['问题'] = domain_faq[i]['问题']
		__global_dict['FAQ_DATA'][index]['答案'] = domain_faq[i]['答案']
		index_dict['领域'].append(index)
		index += 1
	embedding_object.close()
	faq_object.close()

	# Encyclopedia
	# index_dict['百科'] = []
	# embedding_object = open('./KnowledgeMemory/Embedding/DSSM/TransformerDSSM/EncyclopediaEmbedding.json', 'r',
	#                         encoding='utf-8')
	# faq_object = open('./KnowledgeMemory/FAQ/EncyclopediaFAQ.json', 'r', encoding='utf-8')
	# encyclopedia_embedding = json.load(embedding_object)
	# encyclopedia_faq = json.load(faq_object)
	# for i in range(len(encyclopedia_faq)):
	# 	__global_dict['FAQ_DATA'][index] = {}
	# 	__global_dict['FAQ_DATA'][index]['专业'] = encyclopedia_faq[i]['专业']
	# 	if str(i) in encyclopedia_embedding:
	# 		__global_dict['FAQ_DATA'][index]['embedding'] = encyclopedia_embedding[str(i)]
	# 	__global_dict['FAQ_DATA'][index]['问题'] = encyclopedia_faq[i]['问题']
	# 	__global_dict['FAQ_DATA'][index]['答案'] = encyclopedia_faq[i]['答案']
	# 	index_dict['百科'].append(index)
	# 	index += 1
	# embedding_object.close()
	# faq_object.close()

	# Gossip
	# index_dict['闲聊'] = []
	# embedding_object = open('./KnowledgeMemory/Embedding/DSSM/TransformerDSSM/GossipEmbedding.json', 'r',
	#                         encoding='utf-8')
	# faq_object = open('./KnowledgeMemory/FAQ/GossipFAQ.json', 'r', encoding='utf-8')
	# gossip_embedding = json.load(embedding_object)
	# gossip_faq = json.load(faq_object)
	# for i in range(len(gossip_faq)):
	# 	__global_dict['FAQ_DATA'][index] = {}
	# 	__global_dict['FAQ_DATA'][index]['专业'] = gossip_faq[i]['专业']
	# 	if str(i) in gossip_embedding:
	# 		__global_dict['FAQ_DATA'][index]['embedding'] = gossip_embedding[str(i)]
	# 	__global_dict['FAQ_DATA'][index]['问题'] = gossip_faq[i]['问题']
	# 	__global_dict['FAQ_DATA'][index]['答案'] = gossip_faq[i]['答案']
	# 	index_dict['闲聊'].append(index)
	# 	index += 1
	# embedding_object.close()
	# faq_object.close()

	print('loading Data FAQ_INDEX')
	index_dict['所有'] = [i for i in range(len(__global_dict['FAQ_DATA']))]
	__global_dict['FAQ_INDEX'] = index_dict

	print('loading Data Word2Vec_CHARACTER_EMBEDDING')
	with open('./KnowledgeMemory/Embedding/Word2Vec/CharactersEmbedding.json', 'r', encoding='utf-8') as file_object:
		__global_dict['Word2Vec_CHARACTER_EMBEDDING'] = json.load(file_object)
	# </editor-fold>
	# <editor-fold desc="Model">
	# DSSM MODEL
	print('loading Model DSSM')
	__global_dict['MODEL'] = {}
	__global_dict['MODEL']['DSSM'] = {}
	embedding_dict = __global_dict['Word2Vec_CHARACTER_EMBEDDING']
	word_dict = {}
	vec_set = []
	i = 0
	for key in embedding_dict:
		word_dict[key] = i
		vec_set.append(embedding_dict[key][0])
		i += 1
	__global_dict['MODEL']['DSSM']['TransformerDSSM_INFER'] = TransformerDSSM(dict_set=word_dict, vec_set=vec_set,
	                                                                          is_train=False)
	__global_dict['MODEL']['DSSM']['TransformerDSSM_INFER'].build_graph_by_cpu()
	__global_dict['MODEL']['DSSM']['TransformerDSSM_INFER'].start_session()
	# </editor-fold>
	# <editor-fold desc="Interface">
	print('loading Interface IO')
	__global_dict['OUTPUT'] = print
	__global_dict['INPUT'] = input
	# </editor-fold>

	pass


def set_value (key, value):
	"""
	全局变量的添加及修改
	:param key: 关键字
	:param value: 需要修改或添加的值
	:return:
	"""
	__global_dict[key] = value


def get_value (key, def_value=None):
	"""
	全局变量的获取
	:param key: 关键字
	:param def_value:默认值
	:return:
	"""
	try:
		return __global_dict[key]
	except KeyError:
		return def_value
