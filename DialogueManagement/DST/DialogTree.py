# coding=utf-8

"""
author: 王黎成
function: 用于记录用户历史会话信息的树
"""

# 引入外部库
import numpy as np
import random


# 引入内部库


class RootNode:
	"""
	用于存储每一个会话过程的根节点
	"""

	def __init__ (self):
		self.dialog_dict = {}
		self.dialog_nums = 0

	def add_branch (self, init_sentence, scenes):
		init_node = StateNode(talk=init_sentence, state='START')
		self.dialog_dict[scenes] = init_node
		self.dialog_nums += 1

		return init_node


class StateNode:
	"""
	用于存储每一个会话状态的节点
	"""

	def __init__ (self, pattern=None, intent=None, talk=None, state=None):
		# 外部数据成员
		self.pattern = pattern
		self.intent = intent
		self.talk = talk
		self.state = state

		# 内部数据成员
		self.last_state = None
		self.next_state = None

		pass

	def __repr__ (self):
		return "nodeInfo: {},  ".format(self.nodeInfo)

	def __eq__ (self, other):
		selfVal = "{}".format(self.nodeInfo)
		otherVal = "{}".format(other.nodeInfo)

		if hash(selfVal) == hash(otherVal):
			return True
		return False


class DialogTree:
	"""
	记录用户历史会话信息
	一个完整的会话过程由根节点的一棵单分支的子树来记录
	不同分支对应不同场景
	一棵树维护一个用户的会话信息
	"""

	def __init__ (self):
		# 会话树根节点
		self.root_node = RootNode()

		# 当前所处会话节点
		self.cur_node = self.root_node

		# 会话场景处理
		# stack用于场景切换，[(场景,cur_node)}]
		# 场景切换时，才入栈
		self.cur_scenes = None
		self.dialog_scenes_stack = []

		pass

	def add_dialog_branch (self, init_sentence, scenes):
		self.cur_scenes = scenes
		self.cur_node = self.root_node.add_branch(init_sentence, scenes)

	def reset (self):
		self.cur_node = self.root_node
		pass

	def load (self, file_name=None):
		if file_name is None:
			file_name = './data.npy'
		npData2 = np.load(file_name)
		lst2 = npData2.tolist()
		for node in lst2:
			info = node[0:3]
			data = node[3:]
			if (info[0] == 0):
				continue
			print('want to add:')
			print(node)

			if (info[2] == self.cur_Node.nodeInfo[0]):
				self.addSubNodeToCur_Data(data, NodeId=info[0])
				continue

			# self.cur_Node
			count = 10
			while (info[2] != self.cur_Node.nodeInfo[0]):

				ret = self.moveUp()

				if (ret == False):
					print('error ......')
					return

				continue

			self.addSubNodeToCur_Data(data, NodeId=info[0])

		self.printTree()
		pass

	def save (self, file_name=None):
		if (file_name == None):
			file_name = './data.npy'

		nodeLst = self.fetchAllNode()
		dataList = []
		for node in nodeLst:
			print(node)
			dataList += [node.getDataInfo()]
		# Node.
		pass

		npData = np.array(dataList)
		np.save(file_name, npData)

		'''
		npData2 = np.load( './data.npy' )
		lst2 = npData2.tolist()
		print 'lst2:', lst2
	  '''

	def printTree (self):
		nodeLst = self.fetchAllNode()
		for node in nodeLst:
			print(node.getDataInfo())
		pass

	def get_root_node (self):
		return self.root_node

	def get_cur_node (self):
		return self.cur_node

	# nodeinfo： id, level, parentId, childrenIdList[0, 0, 0, []]
	def addSubNodeToCur_Node (self, subNode, isMoveToSub=True):
		newNodeId = self.NodeCount
		self.NodeCount += 1
		self.cur_Node.children += [subNode]
		self.cur_Node.nodeInfo[3] += [newNodeId]

		subNode.parent = self.cur_Node
		subNode.nodeinfo[0] = newNodeId
		subNode.nodeinfo[1] = self.cur_Node.nodeInfo[1] + 1
		subNode.nodeinfo[2] = self.cur_Node.nodeInfo[0]
		subNode.nodeinfo[3] = []
		if (isMoveToSub):
			self.cur_Node = subNode

		pass

	def addSubNodeToCur_Data (self, data, NodeId=0, isMoveToSub=True):
		subNode = Node(data=data)

		if (NodeId == 0):
			newNodeId = self.NodeCount
		else:
			newNodeId = NodeId
		self.NodeCount += 1
		self.cur_Node.children += [subNode]
		self.cur_Node.nodeInfo[3] += [newNodeId]

		subNode.parent = self.cur_Node

		subNode.nodeInfo[0] = newNodeId
		subNode.nodeInfo[1] = self.cur_Node.nodeInfo[1] + 1
		subNode.nodeInfo[2] = self.cur_Node.nodeInfo[0]
		subNode.nodeInfo[3] = []
		# print 'addSubNodeToCur_Data, now in :', self.cur_Node.nodeInfo[0]
		if (isMoveToSub):
			self.cur_Node = subNode

		# print 'addSubNodeToCur_Data, now in :', self.cur_Node.nodeInfo[0]

	def moveToNode (self, nodeId):
		node = self.serachNodeId(self.root_Node, nodeId)
		if (node != None):
			self.cur_Node = node
			return node
		else:
			return None
		pass

	def moveToNode_byNode (self, node):
		self.cur_Node = node
		pass

	def fetchAllNode (self):
		return self.touchAllNode(self.getRootNode())
		pass

	def touchAllNode (self, thisNode):
		allNode = [thisNode]
		for node in thisNode.children:
			allNode += self.touchAllNode(node)

		return allNode

	def serachNodeId (self, thisNode, nodeId):
		if (thisNode.nodeInfo[0] == nodeId):
			return thisNode
		else:
			for node in thisNode.children:
				ret = self.serachNodeId(node, nodeId)
				if (ret != None):
					return ret

		return None

	def move_up (self):
		if (self.cur_node.nodeInfo[0] == 0):
			print('moveUp error')
			return False

		self.cur_node = self.cur_node.parent
		return True
