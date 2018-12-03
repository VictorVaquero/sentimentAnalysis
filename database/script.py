import csv,os
import time
import datetime
from datetime import date # Modulo de fechas de python


import got3 as got
from dateutil.relativedelta import * # Facilita manipulacion de meses

NAME = 'tweets_'
BUSINESS = ['apple', 'google', 'microsoft', 'ibm', 'nvidia']
EXTENSION = '.csv'
DATE_INI = datetime.date(2015,1,1) # Fecha comienzo recojer datos
DATE_END = date.today()

def makeDirs():
    for name in BUSINESS:
        try:
            os.mkdir('./'+name)
        except:
            pass

def writeData(name, my_dir, data):
    """ Escribe datos en archivo name,
        directorio dir a un fichero csv
    """
    with open(my_dir + name, 'w', newline='') as f:
        writer = csv.writer(f)
        for d in data: # Por cada dia recogido del mes
            for t in d: # Por cada tweet en cada dia
                l = [t.id, t.date, t.retweets, t.mentions, t.hashtags, t.text]
                writer.writerow(l)

makeDirs()

date_end = DATE_END + relativedelta(months=0) # Variable fin de mes
date_start = date_end + relativedelta(day = 1) # Variable inicio del mes
while date_end >= DATE_INI:
    for busi in BUSINESS: # Para cada empresa
        date_end = date_start + relativedelta(months = +1,days=-1) # Has modificado date_end, reponlo
        acum = [] # Tweets acumulados de cada mes
        print("Comienza busqueda empresa {}, de fecha {} a fecha {}".format(busi,date_start,date_end))
        while date_end >= date_start:
            date_in = date_end + relativedelta(days=-1)
            twC = got.manager.TweetCriteria().setQuerySearch('#'+busi).setTopTweets(True).setSince(date_in.isoformat()).setUntil(date_end.isoformat()).setMaxTweets(100)
            tweets = got.manager.TweetManager.getTweets(twC)
            acum.append(tweets)
            print("Dia In: {} , Dia Fin: {} --> {} tweets".format(date_in,date_end,len(tweets)))
            date_end = date_end + relativedelta(days=-1)
        writeData(NAME+date_start.isoformat()+EXTENSION,"./"+busi+"/",acum)
        print("Fin busqueda")
    date_start = date_start + relativedelta(months=-1)











