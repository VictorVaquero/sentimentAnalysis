from collections import Counter
import csv

import numpy as np
import tensorflow as tf

# Archivo de texto con los datos a procesar
TEXT = "pruebas.txt"
DIR = "./database/"
FILE = "tweets_apple.csv"
SHAPE = 10 # Diez columnas, n filas
# Tamaño de vocabulario
V = 100
# Tamaño de la proyeccion
D = 10
# Numero de pasadas a la base de datos
EPOCH = 10
# Tamaño de cada grupo entrenamiento
BATCH = 4



def readData(name):
    """ Lectura de archivo csv en directorio DIR 
        Devuelve matriz con los datos y cabecera
    """
    data = []
    with open(DIR + name, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    m = np.matrix(data[1:])
    m.shape = ((len(data)-1)//SHAPE,SHAPE)
    return m, data[0] 


def preprocessing(data):
    """ Limpia y prepara los datos para su posterior
    uso en el algoritmo, devuelve ids y un diccionario
    """
    sp = data.split(" ") # Tokeniza el texto
    count = [ a for (a,_) in Counter(sp).most_common(V-1) ] # Solo queremos palabras mas comunes
    di = { a : count.index(a) for a in count } # Diccionario con la codificacion
    return [ di[x] for x in sp if x in di], di 

def extract_tuples(data):
    """ Saca las tuplas (palabra,prediccion),
    y las devuelve como dos arrays entradas y
    salidas """
    inp = []
    out = []
    for i in range(1,len(data)-1):
        inp.append(data[i])
        inp.append(data[i])
        out.append(data[i-1])
        out.append(data[i+1])
    return inp,out


def writeGraph(sess):
    """ Escribe un archivo con el
    grafo computacional actual """



data = readData(FILE)
data_small = data[0:20,]
print(data_small)
tokens, coding = preprocessing(data)
x_vec, l_vec = extract_tuples(tokens)


# Vectores de entradas y etiquetas
# y pasar a representacion dispersa
with tf.name_scope("Entry"):
    inp = tf.placeholder(tf.int32, shape = (None), name = "Input")
    inp_one_hot = tf.one_hot(inp, depth = V, name = "One_hot_input")
with tf.name_scope("Labels"):
    label = tf.placeholder(tf.int32, shape = (None), name = "Words")
    label_one_hot = tf.one_hot(label, depth = V, name = "One_hot_words")

# Crear las capas de nuestra arquitectura
with tf.name_scope("Proyection_layer"):
    w1 = tf.get_variable("word_embeddings", shape = [V,D], dtype = tf.float32, initializer = tf.random_normal_initializer())
    b1 = tf.get_variable("biases_1", shape = [D], dtype = tf.float32, initializer = tf.zeros_initializer())
    proyection = tf.matmul(inp_one_hot,w1) + b1 # Proyecion lineal 
with tf.name_scope("Output_layer"):
    w2 = tf.get_variable("weitghts_2", shape = [D,V], dtype = tf.float32, initializer = tf.random_normal_initializer())
    b2 = tf.get_variable("biases_2", shape = [V], dtype = tf.float32,  initializer = tf.zeros_initializer())
    output = tf.nn.softmax( tf.matmul(proyection,w2) + b2) # Pasar a probabilidades

with tf.name_scope("Loss"):
    loss = tf.reduce_sum(tf.losses.log_loss(label_one_hot, output))
tf.summary.scalar("loss", loss)
with tf.name_scope("Train"):
    optimizer = tf.train.GradientDescentOptimizer(0.02)
    train = optimizer.minimize(loss)

# Inicializa los valores de las capas
init = tf.global_variables_initializer()
# La session guarda el estado actual de los tensores
sess = tf.Session()
sess.run(init) # Inicializa sesion

# Informacion extra para analisis a posteriori del grafo
wr = tf.summary.FileWriter('./eventos')
wr.add_graph(sess.graph)
merged = tf.summary.merge_all()

# Entrenamiento del grafo
ratio = len(x_vec)//BATCH
for i in range(EPOCH):
    # Pasar a representacion dispersa
    x = x_vec[i*ratio:(i+1)*ratio]
    l = l_vec[i*ratio:(i+1)*ratio]

    feed_dict = {inp : x , label: l}
    summary,_, loss_value = sess.run([merged,train,loss], feed_dict = feed_dict)
    wr.add_summary(summary,i)
    print(loss_value)

wr.close()

