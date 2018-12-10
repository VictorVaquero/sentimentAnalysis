from os import listdir
from os.path import isfile
import csv

FILE_SUFIX = "_limpios.txt"

files_clean = [f for f in listdir("./") if isfile("./" + f) and f[-len(FILE_SUFIX):]== FILE_SUFIX]
files_raw = [f[:-len(FILE_SUFIX)]+".csv" for f in files_clean ]

print(files_clean)
print(files_raw)

def ope(name,shape):
    data = []
    with open(name, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row[shape])
    return data

data_clean = [ope(f,0) for f in files_clean]
data_date = [ope(f,1) for f in files_raw]

print(len(data_clean[1]))
print(len(data_date[1]))

def write(name,data,date):
    with open(name, 'w') as f:
        writer = csv.writer(f)
        for i,row in enumerate(data):
            if i >= len(date):
                break
            writer.writerow([date[i],row])


for i,f in enumerate(files_clean):
    write(f+"2",data_clean[i],data_date[i])



