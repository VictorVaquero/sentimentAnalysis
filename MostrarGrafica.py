#!/usr/bin/env python
# coding: utf-8

# In[28]:


import matplotlib.pyplot as plt
import numpy as np
import csv


# In[32]:


listaHigh = []
listaLow = []
listaClose = []

contador = 0
lineas = len(open('GOOGLPrediccion.csv').readlines())

c = input()
if(int(c)!=0):
    c = int(c)
else:
    c = lineas
cantidad = lineas - c

with open('GOOGLPrediccion.csv', newline='') as File:  
    reader = csv.reader(File)
    for row in reader:
        if(contador>cantidad):
            listaHigh.append(float(row[1]))
            listaLow.append(float(row[2]))
            listaClose.append(float(row[3]))
        contador = contador + 1
        
plt.plot(listaHigh)   # Dibuja el gráfico
plt.xlabel("Fila")   # Inserta el título del eje X 
plt.ylabel("Precio")   # Inserta el título del eje Y
plt.ioff()   # Desactiva modo interactivo de dibujo

plt.plot(listaLow)   # No dibuja datos de lista2
plt.ion()   # Activa modo interactivo de dibujo
plt.plot(listaLow)   # Dibuja datos de lista2 sin borrar datos de lista1

plt.plot(listaClose)   # No dibuja datos de lista2
plt.ion()   # Activa modo interactivo de dibujo
plt.plot(listaClose)   # Dibuja datos de lista2 sin borrar datos de lista1

plt.plot(listaHigh, label = "High", color="b")
plt.plot(listaLow, label = "Low", color="g")
plt.plot(listaClose, label = "Close", color="r")
plt.legend()



