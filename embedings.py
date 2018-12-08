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
OTHER_FILES = ["training.1600000.processed.noemoticon_random_limpios.txt"]
OTHER_SHAPE = 2 # Diez columnas, n filas
END_FILE = ".txt" # Supongo que si acaba en .txt es el archivo con tweets limpios
# Archivo de texto con los datos a procesar
FILE = "apple/tweets_2018-10-01_limpios.txt"
FILE_SUFIX = "_limpios.txt"
KEY_WORDS = "listaPalabrasClave2.txt"
SAVE_FILE = "embedings"



# Tamaño de vocabulario
V = 1000
# Tamaño de la proyeccion
D = 64
# Parametro para la perdida NCE
NUM_SAMPLED = 10
# Numero de objetivos por pase NCE
NUM_TARGET = 1
# Contex windown
CONTEXT_WINDOW = 2
# Numero de pasadas a la base de datos
EPOCH = 100
# Tamaño de cada grupo entrenamiento
BATCH_SIZE = CONTEXT_WINDOW*2
# Parametro de aprendizaje por backtracking
LEARNING_RATE = 0.001
# Parametro proporcion datos para entrenamiento / datos para test
VALIDATE_RATIO = 0.8


def readCSV(name,shape):
    """ Lectura de archivo csv name
        Devuelve matriz con los datos y cabecera
    """
    data = []
    with open(name, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row[shape])
    return data 

def readData(name):
    """ Lectura de archivo txt
    """
    with open(name, 'r') as f:
        data = list(f)
    return data 

def preprocessing(data,key_words):
    """ Prepara los datos para su posterior
    uso en el algoritmo, devuelve el texto con 
    cada palabra su respectiva id y un diccionario de palabra:id
    """
    key = key_words.split(" ")
    sp = [r.split(" ") for d in data for r in d ] # Tokeniza el texto
    count = key[0:V] # Solo queremos V diferentes palabras
    di = { a : count.index(a) for a in count } # Diccionario con la codificacion
    return [[ di[x] for x in r if x in di] for r in sp], di 

def extractTuples(data):
    """ Saca las tuplas (palabra,prediccion),
    y las devuelve como dos arrays entradas y
    salidas """
    inp = []
    out = []
    for r in data:
        for i in range(len(r)):
            for j in range(-CONTEXT_WINDOW,CONTEXT_WINDOW+1):
                if j == CONTEXT_WINDOW or i+j <0 or i+j >= len(r):
                    continue
                inp.append(r[i])
                out.append(r[i+j])
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
    files = [b+"/"+f for b in BUSINESS for f in listdir(DIR+b+ "/") if isfile(DIR+b+ "/" + f) and f[-len(FILE_SUFIX):]== FILE_SUFIX]
    return files

def generateBatch(data):
   return None

# ----------------------------grafo --------------------------

# Vectores de entradas y etiquetas
# y pasar a representacion dispersa
with tf.name_scope("Entry"):
    inp = tf.placeholder(tf.int32, shape = (None), name = "Input")
    inp_one_hot = tf.one_hot(inp, depth = V, name = "One_hot_input")
with tf.name_scope("Labels"):
    label = tf.placeholder(tf.int32, shape = [NUM_TARGET,None], name = "Words")
    label_one_hot = tf.one_hot(label, depth = V, name = "One_hot_words", dtype = tf.int32) # Etiquetas, stop_gradient evita la propagacion del gradiente

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
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(label,[-1]), logits = output, name = "softmax"))
    #loss = tf.reduce_mean(tf.nn.nce_loss(weights = tf.transpose(w2), biases = b2, labels = tf.transpose(label), inputs = proyection, num_sampled = NUM_SAMPLED, num_classes = V, num_true = NUM_TARGET, partition_strategy = "div", name = "softmax"))
    tf.summary.scalar("loss", loss)
with tf.name_scope("Accuracy"): # Calculo de cuantas palabras a acertado
    correct = tf.cast(tf.equal(tf.argmax(tf.nn.softmax(output),1, output_type = tf.int32),label), tf.float32)
    accuracy = tf.reduce_mean(correct)
    tf.summary.scalar("accuracy",accuracy)
with tf.name_scope("Train"): # Entrenar la red con descenso de gradiente
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
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

# -------------------------------- Programa principal ------------------------------


files = cleanFiles()
print("Archivos limpios: {}".format(files))
print("Otros archivos: {}".format(OTHER_FILES))
data = [readCSV(DIR+x,0) for x in files]
data.extend([readCSV(DIR+x,OTHER_SHAPE) for x in OTHER_FILES])
#data = " ".join(data)

key_words = " ".join(readData(PREPROCES_DIR + KEY_WORDS))
tokens, coding = preprocessing(data, key_words)
print("\nVocabulario: {}, N tokens validos: {}".format(V,len(tokens)))
x_vec, l_vec = extractTuples(tokens)

# Crear variable para guardar la matriz de proyeccion
saver = tf.train.Saver({"embedding" : w1, "emb_bias": b1})

# Inicializa los valores de las capas
init = tf.global_variables_initializer()
sess.run(init) # Inicializa sesion

print("\nEjemplo 1: token {}".format(tokens[0]))
for i in range(6):
    print("{} --> label: {}".format(x_vec[i],l_vec[i]))
print("\n")

# Desordenar vectores
rand = random.random()
random.Random(rand).shuffle(x_vec)
random.Random(rand).shuffle(l_vec)

len_l = len(l_vec)
l_vec = np.array(l_vec)
l_vec.shape = (NUM_TARGET,-1)

# Reserva datos para comprovar calidad de la red
# Separar tokens para entrenamiento y tokens para testear nuestra red
x_vec_test = x_vec[int(len(x_vec)*VALIDATE_RATIO):]
l_vec_test = l_vec[:,int(len_l*VALIDATE_RATIO):]
x_vec = x_vec[:int(len(x_vec)*VALIDATE_RATIO)]
l_vec = l_vec[:,:int(len_l*VALIDATE_RATIO)]
print("Label.shape: {}, Input.shape: {}".format(l_vec.shape, len(x_vec)))
print("Test label.shape: {},test Input.shape: {}".format(l_vec_test.shape, len(x_vec_test)))


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
            l = l_vec[:,i*BATCH_SIZE:(i+1)*BATCH_SIZE]

            feed_dict = {inp : x , label: l}
            if i % 1000 == 0: # Solo cada x veces paso la red a ver como va
                summary, accur,lo = sess.run([merged, accuracy,loss],feed_dict = {inp : x_vec_test, label : l_vec_test})
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












