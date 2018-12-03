import random 
import csv


DIR = "./database/"
SENT_FILE = "training.1600000.processed.noemoticon.csv"
OUT_FILE =  "training.1600000.processed.noemoticon_random.csv"


data = []
with open(DIR + SENT_FILE, 'rb') as f:
    for line in f:
        data.append(line)


random.shuffle(data)

with open(DIR + OUT_FILE, 'wb') as f:
    for line in data:
        f.write(line)

