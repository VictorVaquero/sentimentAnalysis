import csv
from collections import Counter

DIR = "./JavaPreprocesamiento/"
TEXT = "market.csv"
FINAL = "market.txt"
IRREGULAR = "IrregularVerbs.csv"
COLS = 10 # Numero de columnas de el archivo de texto
VOCABULARY = 200 # Numero de palabras a mantener

DEBUG = True

# Simbolos o palabras que no se van a considerar
SIGNS = [" œ "," £ ","\"","  ","  ", " of ", " the "," an "," a ",",",".",";",":","...","?","!","¡","(",")","[","]","—","-","_","/","'\'","*","$","@","0","1","2","3","4","5","6","7","8","9","·","#","|","º","ª","~","%","","&","¬","=","^","+","š","ç","<",">","","","","", "“", "”", "•"]
RULES = [   ("won't","will not"), ("shan't", "shall not"), ("can't", "cannot") # Negaciones problematicas
        ]
# Tipicas contracciones, no serian ambiguas porque van con un pronombre
# (la sustitucion es mas compleja en varios casos cuando no van con pronombre)
PRONOUNS = ["i","you","he","she","it","we","they"]
CONTRACTIONS = [("m","am"),("re","are"),("s","is"),("ve","have"),("ll","will"),("d","had")] # Ultimo caso dudoso 'had' o 'would'
RULES.extend([(p+"'"+c,y) for p in PRONOUNS for (c,y) in CONTRACTIONS])
if(DEBUG):
    print(RULES)
    print("\n")

# Comprobar al final de cada palabra
POSTFIX = [ ("n't"," not"), ("nŽt", " not"), ("n`t"," not"), # Negaciones generales
            ("ies","y"), ("s",""), ("es",""), ("zzes","z"), ("ves","f"), # Plural a singular
          ]


def readData(name):
    """ Lectura de archivo csv 
        Devuelve matriz con los datos y cabecera
    """
    data = []
    with open(name, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data

def writeData(name, data):
    """ Escribe en archivo name los
        datos 'data'"""
    with open(name, 'w') as f:
        f.write(data)

def preProcess(data,irregular):
    """ Procesa una cadena de caracteres
    """
    data = [x for row in data for x in row]
    data = " ".join(data[10:])
    data = data.lower() # Trabajar con minusculas
    for (x,y) in RULES: # 
        data = data.replace(x,y)
    for sign in SIGNS: # Elimina simbolos
        data = data.replace(sign, " ")
    for verb in irregular: # Pasar formas verbales irregulares a infinitivo
        for form in verb[1:]:
            if form != "":
                data = data.replace(form,verb[0])
    return data

def tokenize(data):
    sp = [ d for d in data.split(" ") if d != ""] # Tokeniza el texto
    count = [ a for (a,_) in Counter(sp).most_common(VOCABULARY-1) ] # Solo queremos palabras mas comunes
    di = { a : count.index(a) for a in count } # Diccionario con la codificacion
    return " ".join(sp),[ di[x] for x in sp if x in di], di 

    


data = readData(DIR+TEXT)
irregular_verbs = readData(DIR+IRREGULAR)

data = preProcess(data, irregular_verbs)
data,tokens, dic = tokenize(data)
writeData(FINAL,data)

print(dic)
