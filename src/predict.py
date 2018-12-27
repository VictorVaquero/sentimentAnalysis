import csv
from os import listdir
from os.path import isfile
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

BUSINESS = ["apple","google","ibm","microsoft","nvidia"]
DIR = "../database/"
PREDICTIONS_DIR = DIR + "pred/"
SAVE_DIR = "../store/"
PRED_DIR = SAVE_DIR + "pred/"
FILE_WORDS = "words.tsv"
FILE_SUFIX = "_limpios.txt"

SHAPE_TWEET = 0 # Segunda columna tweets

def cleanFiles():
    files = [b+"/"+f for b in BUSINESS for f in listdir(DIR+b+ "/") if isfile(DIR+b+ "/" + f) and f[-len(FILE_SUFIX):]== FILE_SUFIX]
    return files

def readCSV(name,shape = [None], delimiter = ","):
    """ Lectura de archivo csv name
        Devuelve matriz con los datos y cabecera
    """
    data = []
    with open(name, 'r') as f:
        reader = csv.reader(f,delimiter = delimiter)
        for row in reader:
            data.append(row[slice(*shape)])
    return data 
def preprocessingNoLabel(data,di):
    """ Prepara los datos para su posterior
    uso en el algoritmo, devuelve el texto con 
    cada palabra su respectiva id y un diccionario de palabra:id
    """
    valid_size = 0
    out = [] 
    for line in data: 
        twe = line[SHAPE_TWEET]
        sp = twe.split(" ") # Tokeniza el texto
        valid = [ di[x] for x in sp if x in di]
        if(len(valid)>0):
            valid_size += len(valid)
            out.append(valid)
    return out, valid_size 

def padded(data,pad = None):
    max_r = 0
    for r in data:
        max_r = max(max_r,len(r))
    if pad is not None:
        max_r = pad
    m = np.zeros((len(data),max_r))
    for i in range(len(data)):
        m[i,:len(data[i])] = data[i]
    return m

# ------------------------- Main ---------------
coding = readCSV(SAVE_DIR+FILE_WORDS, delimiter = "\t")
coding = { a[0]:i  for i,a in enumerate(coding) } # Diccionario con la codificacion

files = cleanFiles()
print("\nArchivos limpios: {}".format(files))
data = [readCSV(DIR+x,[0,2]) for x in files]
date = [[d[0] for d in f]for f in data]
print(date[0][0])
data = [[d[1] for d in f]for f in data]
print(data[0][0])
x_vec = [preprocessingNoLabel(f, coding) for f in data]
print("Len: {} tweets".format([l for _,l in x_vec]))
x_vec_lengths = [[len(x) for x in f ]for f,_ in x_vec  ]
x_vec = [ padded(f) for f,_ in x_vec]


graph = tf.Graph()
with graph.as_default():
    with tf.Session(graph = graph) as sess:
        tf.saved_model.loader.load(
                sess,
                [tag_constants.SERVING],
                PRED_DIR)
        inp = graph.get_tensor_by_name('Entry/Input:0')
        inp_lengths = graph.get_tensor_by_name('Entry/Input_lengths:0')
        pred = graph.get_tensor_by_name('Accuracy/Output:0')

        predictions = [sess.run(pred, feed_dict = { inp: x, inp_lengths: x_l }) for x,x_l in zip(x_vec,x_vec_lengths)]

for i,fil in enumerate(predictions):
    name = PREDICTIONS_DIR + files[i]
    os.makedirs(os.path.dirname(name), exist_ok = True)
    with open(name, "w") as f:
        wr = csv.writer(f)
        for d,w in zip(date[i],fil):
            wr.writerow([d] +w.tolist())





