# -*- coding: utf-8 -*-
# @Author  : Lone Ranger
# @Function :

from collections import OrderedDict

import torch


class load_parameters_to_cpu():
    def __init__(self, args):
        self.args = args

    def load(self):
        # 加载已有的模型参数到cpu
        state_dict = torch.load(self.args.get('model_save_path'), map_location='cpu')['state_dict']

        # 创建一个新的state_dict加载预训练的模型参数进行预测
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove module.
            new_state_dict[name] = v
        return new_state_dict
