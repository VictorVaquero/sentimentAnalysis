from collections import Counter
from os import listdir
from os.path import isfile
import random 
import csv
import datetime

import numpy as np
import tensorflow as tf

EVENTS_SENT_DIR = "./eventos_sent/"
PREPROCES_DIR = "./JavaPreprocesamiento/"
SAVE_DIR = "./store/"
DIR = "./database/"

SENT_FILE = "training.1000000.processed.noemoticon_limpios.txt"
KEY_WORDS = "listaPalabrasClave2.txt"
SAVE_FILE = "embedings"

SHAPE_SENT = [0,1] # primera columna etiquetas, segunda columna tweet


# Tamaño de vocabulario
V = 1000
# Tamaño de la proyeccion
D = 64
# Numero de pasadas a la base de datos
EPOCH = 100
# Tamaño de cada grupo entrenamiento
BATCH_SIZE = 1
# Parametro de aprendizaje por backtracking
LEARNING_RATE = 0.05
# Parametro de numero de unidades en la celula lstm
LSTM_NUM_UNITS = 32
# Numero de sentimientos distingidos en nuestra base de datos
SENT_DATASET = 2
# Tamaño del vector de siguientes tensores a usar de donde 
# se escoje el siguiente de manera aleatoria
SHUFFLE_SIZE = 100
# Numero de sentimientos diferentes
SENT_NUMBER = 3

def readData(name):
    """ Lectura de archivo txt
    """
    with open(name, 'r') as f:
        data = list(f)
    return " ".join(data)



def preprocessingSent(data,key_words):
    """ Prepara los datos para su posterior
    uso en el algoritmo, devuelve el texto con 
    cada palabra su respectiva id y un diccionario de palabra:id
    """
    key = key_words.split(" ")
    count = key[0:V] # Solo queremos V diferentes palabras
    di = { a : count.index(a) for a in count } # Diccionario con la codificacion
    valid_size = 0

    out = [] 
    label = []
    for line in data: 
        cols = line.split(",")
        twe = cols[SHAPE_SENT[1]]
        lab = cols[SHAPE_SENT[0]]
        sp = twe.split(" ") # Tokeniza el texto
        valid = [ di[x] for x in sp if x in di]
        if(len(valid)>0):
            valid_size += len(valid)
            out.append(valid)
            label.append(lab)
    return out,label, di, valid_size 

def padded(data):
    max_r = 0
    for r in data:
        max_r = max(max_r,len(r))
    m = np.zeros((len(data),max_r))
    for i in range(len(data)):
        m[i,:len(data[i])] = data[i]
    return m



def variableSummaries(var):
    """ Recoje datos en general de la 
        variable var para tensorboard
    """
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var, name = "mean")
        tf.summary.scalar("mean",mean)
        with tf.name_scope("variance"):
            variance = tf.reduce_mean(tf.square(var-mean))
        tf.summary.scalar("variance",variance)
        tf.summary.scalar("max", tf.reduce_max(var))
        tf.summary.scalar("min", tf.reduce_min(var))
        tf.summary.histogram("histogram", var)



# ------------------------------------ Main ------------------------------


data = []
with open(DIR + SENT_FILE, 'r') as f:
    for line in f:
        data.append(line)

key_words = readData(PREPROCES_DIR + KEY_WORDS)
x_vec,l_vec, coding, valid_size = preprocessingSent(data, key_words)
x_vec_lengths = [len(x) for x in x_vec]
x_vec= padded(x_vec) 
print("{}  Len: {} tweets, {} tamaño maximo".format(SENT_FILE,len(data),x_vec.shape[1]))
print("Vocabulario: {}, N tokens validos: {}".format(V,valid_size))





# Vectores de entradas y etiquetas
# y pasar a representacion dispersa
with tf.name_scope("Entry"):
    inp = tf.placeholder(tf.int32, shape = [None,None], name = "Input")
    inp_lengths = tf.placeholder(tf.int32, shape = [None], name = "Input_lengths") # Longitud de cada vector de entrada
    #padded_length = tf.placeholder(tf.int32, shape = (), name = "Padded_length") 
    inp_one_hot = tf.one_hot(inp, depth = V, name = "One_hot_input")
    batch_size = tf.shape(inp)[0]
    
with tf.name_scope("Labels"):
    label = tf.placeholder(tf.int32, shape = [None], name = "Sentiments")

# Primera capa, proyeccion de las palabras 
with tf.name_scope("Proyection_layer"):
    with tf.name_scope("weights"):
        w1 = tf.get_variable("embedding", shape = [V,D], dtype = tf.float32)
        variableSummaries(w1)
    with tf.name_scope("bias"):
        b1 = tf.get_variable("emb_bias", shape = [D], dtype = tf.float32)
        variableSummaries(b1)
    aux = tf.reshape(inp_one_hot, [-1,V])
    proyection = tf.matmul(aux,w1) + b1 # Proyecion lineal, esto es la matriz de reducion de dimensionalidad
    proyection = tf.reshape(proyection, [batch_size, -1, D])
    proyection = tf.stop_gradient(proyection) # Ya esta entrenada, no necesitamos actualizarla
    tf.summary.histogram("proyection", proyection)
# (LSTM) capa recurrente
with tf.name_scope("Recursive_layer"):
    initializer = tf.random_uniform_initializer(-0.01,0.01)
    cell = tf.nn.rnn_cell.LSTMCell(LSTM_NUM_UNITS, initializer = initializer)
    initial_state = cell.zero_state(batch_size, dtype = tf.float32)
    r_output, state = tf.nn.dynamic_rnn(cell = cell, 
                                        inputs = proyection,
                                        sequence_length = inp_lengths,
                                        initial_state = initial_state,
                                        dtype = tf.float32)

# Capa de salida, totalmente conectada
with tf.name_scope("Output_layer"):
    with tf.name_scope("weights"):
        w2 = tf.get_variable("weitghts_2", shape = [LSTM_NUM_UNITS, SENT_NUMBER], dtype = tf.float32, initializer = tf.random_normal_initializer())
        variableSummaries(w2)
    with tf.name_scope("bias"):
        b2 = tf.get_variable("biases_2", shape = [SENT_NUMBER], dtype = tf.float32,  initializer = tf.zeros_initializer())
        variableSummaries(b2)
    aux = tf.reshape(r_output, [-1,LSTM_NUM_UNITS])
    indices = tf.cumsum(inp_lengths)-1
    aux = tf.gather(aux,indices)
    output =  tf.matmul(aux,w2) + b2 # Salida no escalada ( logits )
    tf.summary.histogram("output", output)

# Usamos una softmax para escalar la salida y crear probabilidades,
# ademas usa cross entropy como funcion de perdida
with tf.name_scope("Loss"):
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = label, logits = output, name = "softmax"))
    tf.summary.scalar("loss", loss)
with tf.name_scope("Accuracy"): # Calculo de cuantas palabras a acertado
    correct = tf.cast(tf.equal(tf.argmax(tf.nn.softmax(output),1, output_type = tf.int32),label), tf.float32)
    accuracy = tf.reduce_mean(correct)
    tf.summary.scalar("accuracy",accuracy)
with tf.name_scope("Train"): # Entrenar la red con descenso de gradiente
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    train = optimizer.minimize(loss)

# La session guarda el estado actual de los tensores
sess = tf.Session()

# Informacion extra para analisis a posteriori del grafo
merged = tf.summary.merge_all()
now = datetime.datetime.now()
desc = "V{}_D{}_B{}_L{}".format(V,D,BATCH_SIZE,LEARNING_RATE)
# Diferentes directorios para entrenamiento y pruebas
wr_train = tf.summary.FileWriter(EVENTS_SENT_DIR + "train/" + desc + now.isoformat())
wr_test = tf.summary.FileWriter(EVENTS_SENT_DIR + "test/" + desc + now.isoformat())
wr_train.add_graph(sess.graph) # Añado el grafo

# Inicializa los valores de las capas
init = tf.global_variables_initializer()
# Inicializa sesion
sess.run(init) 
# Crear variable para guardar la matriz de proyeccion
saver = tf.train.Saver({"embedding" : w1, "emb_bias": b1})
# Inicializa los pesos pre entrenados
saver.restore(sess, SAVE_DIR + SAVE_FILE)

# Entrenamiento del grafo
try:
    ratio = x_vec.shape[0]//BATCH_SIZE
    for e in range(EPOCH):
        print("Epoch numero: {} tamaño: {}, Batch tamaño: {}, N batchs: {}".format(e,EPOCH,BATCH_SIZE, ratio))
        for i in range(ratio):
            # Aleatoriza el vector
            rand = random.random()
            random.Random(rand).shuffle(x_vec)
            random.Random(rand).shuffle(x_vec_lengths)
            random.Random(rand).shuffle(l_vec)
            # Selectionar parte para entrenar
            x = x_vec[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:]
            le = x_vec_lengths[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            l = l_vec[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

            feed_dict = {inp : x , inp_lengths: le, label: l}
            if i % 1000 == 0: # Solo cada x veces paso la red a ver como va
                summary, accur,lo = sess.run([merged, accuracy,loss], feed_dict = 
                    {inp: x_vec,
                    inp_lengths: x_vec_lengths,
                    label: l_vec})
                wr_test.add_summary(summary,i)
                print("Pasada {} Paso {} --> Accuracy: {}, loss: {}".format(e,i,accur,lo))
            if i %100 == 99: # Cada x veces, captura informacion de tiempo de ejecucion
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary,_ = sess.run([merged,train], feed_dict = feed_dict, options = run_options, run_metadata = run_metadata)
                wr_train.add_run_metadata(run_metadata, "Pasada {} Paso {}".format(e,i))
                wr_train.add_summary(summary,e*ratio + i)
            else:
                summary,_ = sess.run([merged,train], feed_dict = feed_dict)
                wr_train.add_summary(summary,e*ratio + i)
except KeyboardInterrupt:
    print("Finalizar entrenamiento")

wr_train.close()
wr_test.close()
