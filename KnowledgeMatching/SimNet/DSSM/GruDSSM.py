# coding=utf-8

"""
author: 王黎成
function: 通过使用BI-LSTM作为表示层进行语义相似度计算,该模型是将整个数据一起进行训练，故无BS
"""

# 引入外部库
import os
import random
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, DropoutWrapper


# 引入内部库


class GruDSSM:
	def __init__ (self, q_set=None,  # 问题集,二维数组
	              t_set=None,  # 答案集,二维数组
	              dict_set=None,  # 字典集，[词]
	              vec_set=None,  # 向量集，[向量]，与dict_set顺序一致,
				  negative_sample_num=50, # 负采样数目
	              hidden_num=1024,  # 隐藏层个数
	              learning_rate=0.01,  # 学习率
	              epoch_steps=300,  # 训练迭代次数
	              gamma=20, # 余弦相似度平滑因子
	              is_train=True,  # 是否进行训练
				  top_k=1 # 返回候选问题数目
	              ):
		# 外部参数
		self.q_set = q_set
		self.t_set = t_set
		self.dict_set = dict_set
		self.vec_set = vec_set
		self.negative_sample_num = negative_sample_num
		self.hidden_num = hidden_num
		self.learning_rate = learning_rate
		self.epoch_steps = epoch_steps
		self.gamma = gamma
		self.is_train = is_train

		# 内部参数
		self.top_k_answer = top_k
		self.batch_size = 0
		self.q_length = 0
		self.q_actual_length = []
		self.t_actual_length = []
		self.q_max_length = 0
		self.t_max_length = 0
		self.model_save_name = './ModelMemory/SimNet/DSSM/GruDSSM/model/gruDSSM'
		self.model_save_checkpoint = './ModelMemory/SimNet/DSSM/GruDSSM/model/checkpoint'

		# 模型参数
		self.graph = None
		self.session = None
		self.saver = None
		self.q_inputs = None
		self.q_inputs_actual_length = None
		self.t_inputs = None
		self.t_inputs_actual_length = None
		self.outputs = None
		self.accuracy = None
		self.loss = None
		self.train_op = None

	def init_model_parameters (self):
		# 获取问题数据大小
		self.batch_size = len(self.q_set)

		# 获取答案数据大小
		self.q_length = len(self.t_set)

		# 获取q_set实际长度及最大长度
		for data in self.q_set:
			self.q_actual_length.append(len(data))
		self.q_max_length = max(self.q_actual_length)
		print('the max length of q set is %d' % self.q_max_length)

		# q_set数据补全
		for i in range(len(self.q_set)):
			if len(self.q_set[i]) < self.q_max_length:
				self.q_set[i] = self.q_set[i] + ['UNK' for _ in range(self.q_max_length - len(self.q_set[i]))]

		# 获取t_set实际长度及最大长度
		for data in self.t_set:
			self.t_actual_length.append(len(data))
		self.t_max_length = max(self.t_actual_length)
		print('the max length of t set is %d' % self.t_max_length)

		# t_set数据补全
		for i in range(len(self.t_set)):
			if len(self.t_set[i]) < self.t_max_length:
				self.t_set[i] = self.t_set[i] + ['UNK' for _ in range(self.t_max_length - len(self.t_set[i]))]

		pass

	def generate_data_set (self):
		# 最后一行表示字典中没有的词
		self.dict_set.append('UNK')

		# 最后一行表示字典中没有词的词向量
		self.vec_set.append(list(np.zeros(len(self.vec_set[0]))))

		# 将t_set每一个字转换为其在字典中的序号
		for i in range(len(self.t_set)):
			for j in range(len(self.t_set[i])):
				if self.t_set[i][j] in self.dict_set:
					self.t_set[i][j] = self.dict_set.index(self.t_set[i][j])
				else:
					self.t_set[i][j] = len(self.dict_set) - 1
		self.t_set = np.array(self.t_set)

		# 将q_set每一个字转换为其在字典中的序号
		for i in range(len(self.q_set)):
			for j in range(len(self.q_set[i])):
				if self.q_set[i][j] in self.dict_set:
					self.q_set[i][j] = self.dict_set.index(self.q_set[i][j])
				else:
					self.q_set[i][j] = len(self.dict_set) - 1
		self.q_set = np.array(self.q_set)

		pass

	def presentation_bi_gru (self, inputs, inputs_actual_length, reuse=None):
		with tf.variable_scope('presentation_layer', reuse=reuse):
			# 正向
			fw_cell = GRUCell(num_units=self.hidden_num)
			drop_fw_cell = DropoutWrapper(fw_cell, output_keep_prob=0.5)

			# 反向
			bw_cell = GRUCell(num_units=self.hidden_num)
			drop_bw_cell = DropoutWrapper(bw_cell, output_keep_prob=0.5)

			# 动态rnn函数传入的是一个三维张量，[batch_size,n_steps,n_input]  输出是一个元组 每一个元素也是这种形状
			if self.is_train:
				_, (presentation_fw_final_state, presentation_bw_final_state) = tf.nn.bidirectional_dynamic_rnn(
					cell_fw=drop_fw_cell, cell_bw=drop_bw_cell, inputs=inputs, sequence_length=inputs_actual_length,
					dtype=tf.float32)
			else:
				_, (presentation_fw_final_state, presentation_bw_final_state) = tf.nn.bidirectional_dynamic_rnn(
					cell_fw=fw_cell, cell_bw=bw_cell, inputs=inputs, sequence_length=inputs_actual_length,
					dtype=tf.float32)

			# hiddens的长度为2，其中每一个元素代表一个方向的隐藏状态序列，将每一时刻的输出合并成一个输出
			final_state = tf.concat((presentation_fw_final_state, presentation_bw_final_state), 1)

		return final_state

	def matching_layer_training (self, q_final_state, t_final_state):
		with tf.name_scope('TrainProgress'):
			# 负采样
			t_temp_state = tf.tile(t_final_state, [1, 1])
			for i in range(self.negative_sample_num):
				rand = int((random.random() + i) * self.batch_size / self.negative_sample_num)
				t_final_state = tf.concat((t_final_state,
				                           tf.slice(t_temp_state, [rand, 0], [self.batch_size - rand, -1]),
				                           tf.slice(t_temp_state, [0, 0], [rand, -1])), 0)

			# ||q|| * ||t||
			q_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(q_final_state), 1, True)),
			                 [self.negative_sample_num + 1, 1])
			t_norm = tf.sqrt(tf.reduce_sum(tf.square(t_final_state), 1, True))
			norm_prod = tf.multiply(q_norm, t_norm)

			# qT * d
			prod = tf.reduce_sum(tf.multiply(tf.tile(q_final_state, [self.negative_sample_num + 1, 1]), t_final_state),
			                     1, True)

			# cosine
			cos_sim_raw = tf.truediv(prod, norm_prod)
			cos_sim = tf.transpose(
				tf.reshape(tf.transpose(cos_sim_raw), [self.negative_sample_num + 1, self.batch_size])) * self.gamma

		return cos_sim

	def matching_layer_infer (self, q_final_state, t_final_state):
		with tf.name_scope('InferProgress'):
			# ||q|| * ||t||
			q_norm = tf.tile(tf.transpose(tf.sqrt(tf.reduce_sum(tf.square(q_final_state), 1, True))),
			                 [self.q_length, 1])
			t_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(t_final_state), 1, True)), [1, self.batch_size])
			norm_prod = tf.transpose(tf.multiply(q_norm, t_norm))

			# qT * d
			prod = tf.reshape(tf.transpose(tf.reduce_sum(tf.multiply(
				tf.tile(tf.transpose(tf.reshape(q_final_state, [-1, self.hidden_num * 2, 1]), [2, 1, 0]),
				        [self.q_length, 1, 1]),
				tf.tile(tf.reshape(t_final_state, [-1, self.hidden_num * 2, 1]), [1, 1, self.batch_size])), 1, True),
				[2, 0, 1]), [self.batch_size, self.q_length])

			# cosine
			cos_sim = tf.truediv(prod, norm_prod) * self.gamma

		return cos_sim

	def build_graph (self):
		# 构建模型训练所需的数据流图
		self.graph = tf.Graph()
		with self.graph.as_default():
			with tf.name_scope('placeholder'):
				# 定义Q输入
				self.q_inputs = tf.placeholder(dtype=tf.int64, shape=[None, self.q_max_length])
				self.q_inputs_actual_length = tf.placeholder(dtype=tf.int32, shape=[None])

				# 定义T输入
				self.t_inputs = tf.placeholder(dtype=tf.int64, shape=[None, self.t_max_length])
				self.t_inputs_actual_length = tf.placeholder(dtype=tf.int32, shape=[None])

			with tf.name_scope('InputLayer'):
				# 定义词向量
				embeddings = tf.constant(self.vec_set)

				# 将句子中的每个字转换为字向量
				q_embeddings = tf.nn.embedding_lookup(embeddings, self.q_inputs)
				t_embeddings = tf.nn.embedding_lookup(embeddings, self.t_inputs)

			with tf.name_scope('PresentationLayer'):
				q_final_state = self.presentation_bi_gru(q_embeddings, self.q_inputs_actual_length)
				t_final_state = self.presentation_bi_gru(t_embeddings, self.t_inputs_actual_length, reuse=True)

			with tf.name_scope('MatchingLayer'):
				cos_sim_train = self.matching_layer_training(q_final_state, t_final_state)
				cos_sim_infer = self.matching_layer_infer(q_final_state, t_final_state)

			with tf.name_scope('Loss'):
				# softmax归一化并输出
				prob_train = tf.nn.softmax(cos_sim_train)
				prob_infer = tf.nn.softmax(cos_sim_infer)
				output_train = tf.argmax(prob_train, axis=1)
				_, self.outputs = tf.nn.top_k(prob_infer, self.top_k_answer)

				# 取正样本
				hit_prob = tf.slice(prob_train, [0, 0], [-1, 1])
				self.loss = -tf.reduce_sum(tf.log(hit_prob)) / self.batch_size

			with tf.name_scope('Accuracy'):
				self.accuracy = tf.reduce_sum(
					tf.cast(tf.equal(output_train, tf.zeros_like(output_train)), dtype=tf.float32)) / self.batch_size

			# 优化并进行梯度修剪
			with tf.name_scope('Train'):
				optimizer = tf.train.AdamOptimizer(self.learning_rate)
				# 分解成梯度列表和变量列表
				grads, vars = zip(*optimizer.compute_gradients(self.loss))

				# 梯度修剪
				gradients, _ = tf.clip_by_global_norm(grads, 5)  # clip gradients

				# 将每个梯度以及对应变量打包
				self.train_op = optimizer.apply_gradients(zip(gradients, vars))

			# 设置模型存储所需参数
			self.saver = tf.train.Saver()

	def train (self):
		with tf.Session(graph=self.graph) as self.session:
			# 判断模型是否存在
			if os.path.exists(self.model_save_checkpoint):
				# 恢复变量
				print('restoring------')
				self.saver.restore(self.session, self.model_save_name)
			else:
				# 初始化变量
				print('Initializing------')
				tf.initialize_all_variables().run()

			# 开始迭代，使用Adam优化的随机梯度下降法，并将结果输出到日志文件
			print('training------')
			file_object = open('./ModelMemory/SimNet/DSSM/GruDSSM/log/' + time.strftime('%Y-%m-%d', time.localtime(
				time.time())) + '.log', 'w', encoding='utf-8')
			for i in range(self.epoch_steps):
				# 开始训练
				feed_dict = {self.q_inputs: self.q_set, self.q_inputs_actual_length: self.q_actual_length,
				             self.t_inputs: self.t_set, self.t_inputs_actual_length: self.t_actual_length}
				_, loss, accuracy = self.session.run([self.train_op, self.loss, self.accuracy], feed_dict=feed_dict)

				file_object.write('[epoch:%d] loss %f accuracy %f\n' % (i, loss, accuracy))
			file_object.close()

			# 保存模型
			print('save model------')
			self.saver.save(self.session, self.model_save_name)

		pass

	def inference (self):
		with tf.Session(graph=self.graph) as self.session:
			self.saver.restore(self.session, self.model_save_name)

			feed_dict = {self.q_inputs: self.q_set, self.q_inputs_actual_length: self.q_actual_length,
			             self.t_inputs: self.t_set, self.t_inputs_actual_length: self.t_actual_length}
			results = self.session.run(self.outputs, feed_dict=feed_dict)

			return results
