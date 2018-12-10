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
PRED_DIR = SAVE_DIR + "pred/"
DIR = "./database/"

BUSINESS = ["apple","google","ibm","microsoft","nvidia"]
SENT_FILE = "training.1600000.processed.noemoticon_random_limpios.txt"
KEY_WORDS = "listaPalabrasClave2.txt"
SAVE_FILE = "embedings"
FILE_WORDS = "words.tsv"
FILE_SUFIX = "_limpios.txt"

SHAPE_TWEET = 0 # Segunda columna tweets
SHAPE_SENT = [1,2] # Primera columna etiquetas, segunda columna tweet


V = 1000 # Tamaño de vocabulario
D = 64 # Tamaño de la proyeccion
EPOCH = 20 # Numero de pasadas a la base de datos
BATCH_SIZE = 2 # Tamaño de cada grupo entrenamiento
LEARNING_RATE = 0.05 # Parametro de aprendizaje por backtracking
LSTM_NUM_UNITS = 32 # Parametro de numero de unidades en la celula lstm
SENT_DATASET = 2 # Numero de sentimientos distingidos en nuestra base de datos
# Tamaño del vector de siguientes tensores a usar de donde 
# se escoje el siguiente de manera aleatoria
SHUFFLE_SIZE = 100
SENT_NUMBER = 2 # Numero de sentimientos diferentes
VALIDATE_RATIO = 0.8 # Ratio de datos 
PRELABEL_PASS = 0.8 # Si la confianza en la prediccion es mayor que este valor, se añade a la base de datos de entrenamiento


def readData(name):
    """ Lectura de archivo txt
    """
    with open(name, 'r') as f:
        data = list(f)
    return " ".join(data)

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


def preprocessingSent(data, di):
    """ Prepara los datos para su posterior
    uso en el algoritmo, devuelve el texto con 
    cada palabra su respectiva id y un diccionario de palabra:id
    """
    valid_size = 0

    out = [] 
    label = []
    for line in data: 
        twe = line[SHAPE_SENT[1]]
        lab = int(line[SHAPE_SENT[0]])
        sp = twe.split(" ") # Tokeniza el texto
        valid = [ di[x] for x in sp if x in di]
        if(len(valid)>0):
            valid_size += len(valid)
            out.append(valid)
            label.append(lab if lab == 0 else 1) # Remapeando sentimiento etiqueta 4 --> 1
    return out,label, valid_size 

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
    return m,max_r



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

def randomizeVecs(vecs):
    rand = random.random()
    for v in vecs:
        random.Random(rand).shuffle(v)



# ----------------------------- Red ----------------------------------
def training(sess, vecs, vecs_test, inputs, outputs, writters, hibrid):
    """ Entrenamiento del grafo
    """
    x_vec, x_vec_lengths, l_vec = vecs
    x_vec_test, x_vec_lengths_test, l_vec_test = vecs_test
    inp, inp_lengts, label = inputs
    merged, train, accuracy, loss, ou = outputs
    wr_test, wr_train = writters
    try:
        ratio = x_vec.shape[0]//BATCH_SIZE
        for e in range(EPOCH):
            print("Epoch numero: {} tamaño: {}, Batch tamaño: {}, N batchs: {}".format(e,EPOCH,BATCH_SIZE, ratio))
            for i in range(ratio):
                # Aleatoriza el vector
                randomizeVecs([x_vec,x_vec_lengths, l_vec])
                # Selectionar parte para entrenar
                batch = slice(i*BATCH_SIZE,(i+1)*BATCH_SIZE)
                x = x_vec[batch,:]
                le = x_vec_lengths[batch]
                l = l_vec[batch]

                feed_dict = {inp : x , inp_lengths: le, label: l}
                if i % 1000 == 0: # Solo cada x veces paso la red a ver como va
                    summary, accur,lo, prelabel = sess.run([merged, accuracy,loss,ou], feed_dict = 
                        {inp: x_vec_test,
                        inp_lengths: x_vec_lengths_test,
                        label: l_vec_test})
                    for i in range(10):
                        print("label: {} --> pre: {}".format(l_vec_test[i],prelabel[i]))
                    wr_test.add_summary(summary,i)
                    print("Pasada {} Paso {} --> Accuracy: {}, loss: {}".format(e,i,accur,lo))
                if i %100 == 99: # Cada x veces, captura informacion de tiempo de ejecucion
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary,_ = sess.run([merged,train], feed_dict = feed_dict, options = run_options, run_metadata = run_metadata)
                    wr_train.add_run_metadata(run_metadata, "Meta {} Pasada {} Paso {}".format(hibrid, e,i))
                    wr_train.add_summary(summary,e*ratio + i)
                else:
                    summary,_ = sess.run([merged,train], feed_dict = feed_dict)
                    wr_train.add_summary(summary,e*ratio + i)
    except KeyboardInterrupt:
        print("Finalizar entrenamiento")

def evaluate(sess, vecs, inputs, outputs):
    """ Pasada de evaluacion, devuelve 
    las etiquetas predichas del vector """
    x_vec, x_vec_lengths = vecs
    inp, inp_lengts = inputs
    ou = outputs
    prelabel = []

    for i in range(len(x_vec)):
        feed_dict = {inp : np.reshape(x_vec[i],[1,-1]) , inp_lengths: np.reshape(x_vec_lengths[i],[1])}
        pl = sess.run(ou, feed_dict = feed_dict)
        prelabel.append(pl)

    return prelabel

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
    ou = tf.nn.softmax(output, name = "Output")
    correct = tf.cast(tf.equal(tf.argmax(ou,1, output_type = tf.int32),label), tf.float32)
    accuracy = tf.reduce_mean(correct)
    tf.summary.scalar("accuracy",accuracy)
with tf.name_scope("Train"): # Entrenar la red con descenso de gradiente
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    train = optimizer.minimize(loss)

# ------------------------------------ Main ------------------------------


data_label = readCSV(DIR+SENT_FILE)
print("\nEjemplo 1: {}".format(data_label[0]))

coding = readCSV(SAVE_DIR+FILE_WORDS, delimiter = "\t")
coding = { a[0]:i  for i,a in enumerate(coding) } # Diccionario con la codificacion

x_vec,l_vec, valid_size = preprocessingSent(data_label, coding)
x_vec_lengths = [len(x) for x in x_vec]
x_vec,pad = padded(x_vec, 50) 
print("{}  Len: {} tweets, {} tamaño maximo".format(SENT_FILE,len(data_label),x_vec.shape[1]))
print("Vocabulario: {}, N tokens validos: {}".format(V,valid_size))



files = cleanFiles()
print("\nArchivos limpios: {}".format(files))
data_nolabel = [readCSV(DIR+x,[1,2]) for x in files]
print(data_nolabel[0][0])
print(data_nolabel[6][0])
x_vec_nolabel = [preprocessingNoLabel(f, coding) for f in data_nolabel]
print("Len: {} tweets".format([l for _,l in x_vec_nolabel]))
x_vec_nolabel = [ x for f,_ in x_vec_nolabel for x in f]
x_vec_lengths_nolabel = [len(x) for x in x_vec_nolabel]
x_vec_nolabel,_ = padded(x_vec_nolabel,pad = pad)


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
# Crear variable para obtener la matriz de proyeccion
saver = tf.train.Saver({"embedding" : w1, "emb_bias": b1})
# Inicializa los pesos pre entrenados
saver.restore(sess, SAVE_DIR + SAVE_FILE)

randomizeVecs([x_vec,x_vec_lengths, l_vec])
# Reserva datos para comprovar calidad de la red

trainsl= slice(None, int(len(x_vec)*VALIDATE_RATIO))
test = slice(int(len(x_vec)*VALIDATE_RATIO), None)
x_vec_test = x_vec[test,:]
x_vec_lengths_test = x_vec_lengths[test]
l_vec_test = l_vec[test]
x_vec = x_vec[trainsl,:]
x_vec_lengths = x_vec_lengths[trainsl]
l_vec = l_vec[trainsl]

train_vecs = x_vec, x_vec_lengths, l_vec
test_vecs = x_vec_test, x_vec_lengths_test, l_vec_test
inputs = (inp, inp_lengths, label)
outputs = (merged, train, accuracy, loss, ou )
writters = (wr_test, wr_train )

hibrid = 0
try:
    while len(x_vec_nolabel) > 0:
        train_vecs = x_vec, x_vec_lengths, l_vec
        test_vecs = x_vec_test, x_vec_lengths_test, l_vec_test

        training(sess, train_vecs, test_vecs, inputs, outputs, writters, hibrid)
        print("Evaluate: len {}    {}".format(x_vec_nolabel.shape, len(x_vec_lengths_nolabel)))
        prelabels = evaluate(sess, (x_vec_nolabel, x_vec_lengths_nolabel), (inp, inp_lengths), ou)

        valid_data = [x for i,x in enumerate(x_vec_nolabel) if max(prelabels[i][0]) > PRELABEL_PASS ]
        valid_data = np.array(valid_data)
        x_vec = np.vstack([x_vec,valid_data])
        x_vec_lengths.extend([ x for i,x in enumerate(x_vec_lengths_nolabel) if max(prelabels[i][0]) > PRELABEL_PASS ])

        x_vec_nolabel = np.array([x for i,x in enumerate(x_vec_nolabel) if max(prelabels[i][0]) <= PRELABEL_PASS ])
        x_vec_lengths_nolabel = [x for i,x in enumerate(x_vec_lengths_nolabel) if max(prelabels[i][0]) <= PRELABEL_PASS ]

        valid_data = [np.argmax(x[0]) for x in prelabels if max(x[0])> PRELABEL_PASS ]
        l_vec.extend(valid_data)
        hibrid += 1

except KeyboardInterrupt:
    print("End all training")


# Finalizado entrenamiento, guardar modelo para prediccion
tf.saved_model.simple_save(sess,
        PRED_DIR,
        inputs={"inp": inp, "inp_lengths": inp_lengths},
        outputs={"ou": ou})


wr_train.close()
wr_test.close()
