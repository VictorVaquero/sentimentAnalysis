from collections import Counter

import numpy as np
import tensorflow as tf

# Archivo de texto con los datos a procesar
TEXT = "pruebas.txt"
# Tamaño de vocabulario
V = 100
# Tamaño de la proyeccion
D = 10
# Numero de pasadas a la base de datos
EPOCH = 10
# Tamaño de cada grupo entrenamiento
BATCH = 4

def readData(name):
    """ Lectura de archivo con condificacion utf_8
    """
    with open(name, encoding = 'utf_8') as f:
        data = f.read()
    return data


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



data = readData(TEXT)
tokens, coding = preprocessing(data)
x_vec, l_vec = extract_tuples(tokens)


# Vectores de entradas y etiquetas
inp = tf.placeholder(tf.int32, shape = (None))
label = tf.placeholder(tf.int32, shape = (None))
# Pasar a representacion dispersa
inp_one_hot = tf.one_hot(inp, depth = V)
label_one_hot = tf.one_hot(label, depth = V)

# Crear las capas de nuestra arquitectura
w1 = tf.get_variable("word_embeddings", shape = [V,D], dtype = tf.float32, initializer = tf.random_normal_initializer())
b1 = tf.get_variable("biases_1", shape = [D], dtype = tf.float32, initializer = tf.zeros_initializer())
proyection = tf.matmul(inp_one_hot,w1) + b1 # Proyecion lineal 

w2 = tf.get_variable("weitghts_2", shape = [D,V], dtype = tf.float32, initializer = tf.random_normal_initializer())
b2 = tf.get_variable("biases_2", shape = [V], dtype = tf.float32,  initializer = tf.zeros_initializer())
output = tf.nn.softmax( tf.matmul(proyection,w2) + b2) # Pasar a probabilidades

loss = tf.reduce_sum(tf.losses.log_loss(label_one_hot, output))
tf.summary.scalar("loss", loss)
optimizer = tf.train.GradientDescentOptimizer(0.02)
train = optimizer.minimize(loss)

# Inicializa los valores de las capas
init = tf.global_variables_initializer()
# La session guarda el estado actual de los tensores
sess = tf.Session()
sess.run(init) # Inicializa sesion

# Informacion extra para analisis a posteriori del grafo
wr = tf.summary.FileWriter('./eventos',sess.graph)
merged = tf.summary.merge_all()

# Entrenamiento del grafo
ratio = len(x_vec)//BATCH
for i in range(EPOCH):
    # Pasar a representacion dispersa
    x = x_vec[i*ratio:(i+1)*ratio]
    l = l_vec[i*ratio:(i+1)*ratio]

    feed_dict = {inp : x , label: l}
    summary, loss_value = sess.run([merged,(train, loss)], feed_dict = feed_dict)
    wr.add_summary(summary,i)
    print(loss_value)

wr.close()

