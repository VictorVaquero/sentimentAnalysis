from collections import Counter
import csv

import numpy as np
import tensorflow as tf

# Archivo de texto con los datos a procesar
TEXT = "market.txt"
DIR = "./database/"
FILE = "tweets_apple.csv"
SHAPE = 10 # Diez columnas, n filas
# Tamaño de vocabulario
V = 200
# Tamaño de la proyeccion
D = 32
# Numero de pasadas a la base de datos
EPOCH = 10
# Tamaño de cada grupo entrenamiento
BATCH = 4



def readCSV(name):
    """ Lectura de archivo csv name 
        Devuelve matriz con los datos y cabecera
    """
    data = []
    with open(name, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    m = np.matrix(data[1:])
    m.shape = ((len(data)-1)//SHAPE,SHAPE)
    return m, data[0] 

def readData(name):
    """ Lectura de archivo txt
    """
    with open(name, 'r') as f:
        data = list(f)
    return " ".join(data)

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



#data = readCSV(DIR+FILE)
data = readData(TEXT)
data_small = data[0:20]
print(TEXT+ "  len: {}".format(len(data)))
print(data_small)
tokens, coding = preprocessing(data)
print("N tokens: {}".format(len(tokens)))
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
    tf.summary.histogram("weights",w1)
    tf.summary.histogram("biases",b1)
with tf.name_scope("Output_layer"):
    w2 = tf.get_variable("weitghts_2", shape = [D,V], dtype = tf.float32, initializer = tf.random_normal_initializer())
    b2 = tf.get_variable("biases_2", shape = [V], dtype = tf.float32,  initializer = tf.zeros_initializer())
    output = tf.nn.softmax( tf.matmul(proyection,w2) + b2) # Pasar a probabilidades
    tf.summary.histogram("weights",w2)
    tf.summary.histogram("biases",b2)

with tf.name_scope("Loss"):
    loss = tf.reduce_sum(tf.losses.softmax_cross_entropy(label_one_hot, output))
    tf.summary.scalar("loss", loss)
with tf.name_scope("Accuracy"):
    correct = tf.equal(tf.argmax(output,0),tf.argmax(label_one_hot,0))
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
    tf.summary.scalar("accuracy",accuracy)
with tf.name_scope("Train"):
    optimizer = tf.train.GradientDescentOptimizer(1.01)
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
print("Batch: {}, N epoch: {}".format(BATCH, ratio))
for i in range(ratio):
    # Pasar a representacion dispersa
    x = x_vec[i*BATCH:(i+1)*BATCH]
    l = l_vec[i*BATCH:(i+1)*BATCH]

    feed_dict = {inp : x , label: l}
    if i % 500 == 0:
        accur,lo = sess.run([accuracy,loss],feed_dict = feed_dict)
        print("Accuracy: {}, loss: {}".format(accur,lo))
    summary,_ = sess.run([merged,train], feed_dict = feed_dict)
    wr.add_summary(summary,i)

wr.close()

