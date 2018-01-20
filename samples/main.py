import os
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#data = pd.read_csv('/InformatiCup2018-master/Eingabedaten/Benzinpreise/1.csv')

PATH_TO_DATA = "InformatiCup2018/Eingabedaten/"
FUEL_FILE = "1.csv"
FUEL_PATH = ("Benzinpreise/" + FUEL_FILE,
             ["Time", "Price"])
ROUTE_PATH = ("Fahrzeugrouten/Bertha Benz Memorial Route.csv",
              ["Time", "Number"])
GAS_STATION_PATH = ("Tankstellen.csv",
                    ["ID", "Name", "Provider", "Streetname", "Housenumber", "Zipcode", "City", "Longitude", "Latitude"])

def loadData(typeOfData, pathToData=PATH_TO_DATA):
    csv_path = os.path.join(pathToData, typeOfData[0])
    return pd.read_csv(csv_path, delimiter=";", names = typeOfData[1])

fuelPrice = loadData(FUEL_PATH)
index = pd.DatetimeIndex(fuelPrice.Time)
fuelPrice.drop('Time', axis = 1, inplace = True)
fuelPrice.set_index(index, inplace = True)


print(fuelPrice.head())