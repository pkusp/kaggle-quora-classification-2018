# encoding: utf-8
"""
@author: pkusp
@contact: pkusp@outlook.com

@version: 1.0
@file: config.py
@time: 2018/11/9 下午9:50

这一行开始写关于本文件的说明与解释
"""
import os

config_path = os.path.abspath(__file__)

project_root = os.path.dirname(config_path)

glove_embd_path= "../"

print("root:\n",config_path)
print("root:\n",project_root)
print("hello")

