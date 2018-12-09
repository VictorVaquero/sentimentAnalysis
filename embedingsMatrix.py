V = 1000
D = 64 # Tamaño de la proyeccion
SAVE_DIR = "./store/"
SAVE_FILE = "embedings"

with tf.name_scope("Entry"):
    inp = tf.placeholder(tf.int32, shape = [None,None], name = "Input")
    #padded_length = tf.placeholder(tf.int32, shape = (), name = "Padded_length") 
    inp_one_hot = tf.one_hot(inp, depth = V, name = "One_hot_input")
    batch_size = tf.shape(inp)[0]

with tf.name_scope("Proyection_layer"):
    with tf.name_scope("weights"):
        w1 = tf.get_variable("embedding", shape = [V,D], dtype = tf.float32)
    with tf.name_scope("bias"):
        b1 = tf.get_variable("emb_bias", shape = [D], dtype = tf.float32)
    aux = tf.reshape(inp_one_hot, [-1,V])
    proyection = tf.matmul(aux,w1) + b1 # Proyecion lineal, esto es la matriz de reducion de dimensionalidad
    proyection = tf.reshape(proyection, [batch_size, -1, D])

sess = tf.Session()

# Inicializa sesion
sess.run() 
# Crear variable para obtener la matriz de proyeccion
saver = tf.train.Saver({"embedding" : w1, "emb_bias": b1})
# Inicializa los pesos pre entrenados
saver.restore(sess, SAVE_DIR + SAVE_FILE)
