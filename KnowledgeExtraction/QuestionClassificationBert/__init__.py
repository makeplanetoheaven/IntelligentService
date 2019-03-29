# -*- coding: utf-8 -*-
# @Author  : Lone Ranger
# @Function :

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from KnowledgeExtraction.QuestionClassificationBert.Preprocess import convert_examples_to_features, MyPro, sentencePro

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
