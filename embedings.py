from collections import Counter
from os import listdir
from os.path import isfile
import random 
import csv
import datetime

import numpy as np
import tensorflow as tf

EVENTS_DIR = "./eventos/"
PREPROCES_DIR = "./JavaPreprocesamiento/"
SAVE_DIR = "./store/"
DIR = "./database/"
BUSINESS = ["apple","google","ibm","microsoft","nvidia"]
END_FILE = ".txt" # Supongo que si acaba en .txt es el archivo con tweets limpios
# Archivo de texto con los datos a procesar
FILE = "apple/tweets_2018-10-01_limpios.txt"
KEY_WORDS = "listaPalabrasClave2.txt"
SAVE_FILE = "embedings"

SHAPE = 10 # Diez columnas, n filas


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

def preprocessing(data,key_words):
    """ Prepara los datos para su posterior
    uso en el algoritmo, devuelve el texto con 
    cada palabra su respectiva id y un diccionario de palabra:id
    """

    key = key_words.split(" ")
    sp = data.split(" ") # Tokeniza el texto
    count = key[0:V] # Solo queremos V diferentes palabras
    di = { a : count.index(a) for a in count } # Diccionario con la codificacion
    return [ di[x] for x in sp if x in di], di 

def extractTuples(data):
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


def cleanFiles():
    path = DIR+ b + "/"
    files = [f for b in BUSINESS for f in listdir(path) if isfile(join(path,f))]
    return files


# -------------------------------- Programa principal ------------------------------


data = readData(DIR+FILE)
#data = readCSV(DIR+FILE)
#data = readData(TEXT)
print(FILE+ "  Len: {} palabras".format(len(data)))
key_words = readData(PREPROCES_DIR + KEY_WORDS)
tokens, coding = preprocessing(data, key_words)
print("Vocabulario: {}, N tokens validos: {}".format(V,len(tokens)))
x_vec, l_vec = extractTuples(tokens)


# Vectores de entradas y etiquetas
# y pasar a representacion dispersa
with tf.name_scope("Entry"):
    inp = tf.placeholder(tf.int32, shape = (None), name = "Input")
    inp_one_hot = tf.one_hot(inp, depth = V, name = "One_hot_input")
with tf.name_scope("Labels"):
    label = tf.placeholder(tf.int32, shape = (None), name = "Words")
    # label_one_hot = tf.one_hot(label, depth = V, name = "One_hot_words", dtype = tf.int32) # Etiquetas, stop_gradient evita la propagacion del gradiente

# Crear las capas de nuestra arquitectura
with tf.name_scope("Proyection_layer"):
    with tf.name_scope("weights"):
        w1 = tf.get_variable("word_embeddings", shape = [V,D], dtype = tf.float32, initializer = tf.random_normal_initializer())
        variableSummaries(w1)
    with tf.name_scope("bias"):
        b1 = tf.get_variable("biases_1", shape = [D], dtype = tf.float32, initializer = tf.zeros_initializer())
        variableSummaries(b1)
    proyection = tf.matmul(inp_one_hot,w1) + b1 # Proyecion lineal, esto es la matriz de reducion de dimensionalidad
    tf.summary.histogram("proyection", proyection)

# Capa de salida, denuevo lineal
with tf.name_scope("Output_layer"):
    with tf.name_scope("weights"):
        w2 = tf.get_variable("weitghts_2", shape = [D,V], dtype = tf.float32, initializer = tf.random_normal_initializer())
        variableSummaries(w2)
    with tf.name_scope("bias"):
        b2 = tf.get_variable("biases_2", shape = [V], dtype = tf.float32,  initializer = tf.zeros_initializer())
        variableSummaries(b2)
    output =  tf.matmul(proyection,w2) + b2 # Salida no escalada ( logits )
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
wr_train = tf.summary.FileWriter(EVENTS_DIR + "train/" + desc + now.isoformat())
wr_test = tf.summary.FileWriter(EVENTS_DIR + "test/" + desc + now.isoformat())
wr_train.add_graph(sess.graph) # Añado el grafo

# Crear variable para guardar la matriz de proyeccion
saver = tf.train.Saver({"embedding" : w1, "emb_bias": b1})

# Inicializa los valores de las capas
init = tf.global_variables_initializer()
sess.run(init) # Inicializa sesion

# Entrenamiento del grafo
try:
    for e in range(EPOCH):
        ratio = len(x_vec)//BATCH_SIZE
        print("Epoch tamaño: {}, Batch tamaño: {}, N batchs: {}".format(EPOCH,BATCH_SIZE, ratio))
        # Aleatoriza el vector
        rand = random.random()
        random.Random(rand).shuffle(x_vec)
        random.Random(rand).shuffle(l_vec)
        for i in range(ratio):
            # Selectionar parte para entrenar
            x = x_vec[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            l = l_vec[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

            feed_dict = {inp : x , label: l}
            if i % 1000 == 0: # Solo cada x veces paso la red a ver como va
                summary, accur,lo = sess.run([merged, accuracy,loss],feed_dict = {inp : x_vec, label : l_vec})
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
# Guarda los valores en SAVE_FILE
save_path = saver.save(sess, SAVE_DIR + SAVE_FILE)
print("Modelo guardado en {}".format(save_path))












