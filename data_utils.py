#-*- coding:utf-8 -*-
import pandas as pd
from collections import OrderedDict

def encode_by_count(DataFrameCol):
    '''
    :param DataFrameCol: pandas Series
    :return: descending value and count
    '''
    info = DataFrameCol.value_counts()
    axes = info.index.values
    counts = info.values
    class_range = len(axes)
    str2id = OrderedDict(zip(axes,range(class_range)))
    # print str2id
    return str2id