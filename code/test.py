import os,sys
sys.path.append("./feature_lib")
from calc_feature import get_feature
import random
from Bio import SeqIO
import numpy as np
import pandas as pd
import keras
from keras.layers import Dense, Input, Conv1D, CuDNNLSTM, Conv2D, Concatenate, Multiply, RepeatVector, Permute
from keras import Model, losses, optimizers
import tensorflow as tf
import keras.backend as K
from sklearn import metrics
from sklearn.preprocessing import minmax_scale, scale,StandardScaler
from optparse import OptionParser

def get_features(seq_path):
    features =get_feature(seq_path=seq_path)
    return features

def get_seq_id(seq_path):
    records = list(SeqIO.parse(seq_path,format="fasta"))
    ids = [str(record.id) for record in records]
    return ids

'''
python test.py -i ./demo.fasta -m cppnet.h5 -o result.tsv
'''

parser = OptionParser()
parser.add_option("-i", "--seq_path", dest="seq_path", help="sequences path with Fasta format")
parser.add_option("-m", "--model_file", dest="model_file", help="NCResNet model file")
parser.add_option("-o", "--output_file", dest="output_file", help="Output file path with tsv format")
option, args = parser.parse_args()

model = keras.models.load_model(option.model_file,compile=False)
ids = get_seq_id(option.seq_path)
features = get_features(option.seq_path)
features = StandardScaler().fit_transform(X=features)
pred_res = model.predict(features).flatten()
pred_label = []
for x in pred_res:
    if x>0.5:
        pred_label.append("pcRNA")
    else:
        pred_label.append("ncRNA")
f= open(option.output_file,"w")
f.write("id\tlabel\tscore\n")
for i in range(len(features)):
    print("%s\t%s\t%f"%(ids[i],pred_label[i],pred_res[i]),file=f)
f.close()













