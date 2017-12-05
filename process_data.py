from typing import List

import pandas as pd

import os
import re
from fbprophet import Prophet
import matplotlib.pyplot as plt
import numpy as np


class InformatiCup2018(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.gas_stations_data_dir = data_dir + os.path.join('Eingabedaten', 'Benzinpreise')
        self.gas_stations_dir = data_dir + os.path.join('Eingabedaten', 'Tankstellen.csv')
        self.gas_stations = {}
        self.gas_stations_dfs = {}
        self.gas_stations_models = {}

    def init_all(self):
        self.read_all_gas_stations()
        self.create_models_for_all_gas_stations()

    def init_gas_stations(self):
        self.gas_stations = pd.read_csv(self.gas_stations_dir, sep=';', header=None,
                                        names=['id', 'name', 'brand', 'street', 'house_no', 'plz', 'town', 'latutude',
                                               'longitude'])

    def read_gas_station_data(self, gas_station_id: int):
        print(f"Reading data for gas station {gas_station_id}.")
        data = pd.read_csv(os.path.join(self.gas_stations_data_dir, f'{gas_station_id}.csv'),
                           sep=';', header=None,
                           names=['ds', 'y'])

        data['ds'] = pd.to_datetime(data['ds'])
        data['y'] = pd.to_numeric(data['y'], downcast='float')
        self.gas_stations_dfs[gas_station_id] = data
        return data

    def read_all_gas_stations(self):
        for file in os.listdir(self.gas_stations_data_dir):
            m = re.search(r'(.*)[.]', file)
            id = int(m.group(1))
            self.gas_stations_dfs[id] = self.read_gas_station_data(id)

    def create_model_for_gas_station(self, gas_station_id: int):
        print(f"Creating model for gas station {gas_station_id}.")
        m = Prophet()
        # TODO: Add holidays to each gas station

        m.fit(self.gas_stations_dfs[gas_station_id])
        self.gas_stations_models[gas_station_id] = m
        return m

    def create_models_for_all_gas_stations(self):
        for file in os.listdir(self.gas_stations_data_dir):
            m = re.search(r'(.*)[.]', file)
            id = int(m.group(1))
            self.gas_stations_models[id] = self.create_model_for_gas_station(id)

    def predict_prices(self, gas_station_id: int, dates: List[str]):
        predict_date = pd.DataFrame()
        predict_date['ds'] = pd.to_datetime(dates)
        forecast = self.gas_stations_models[gas_station_id].predict(predict_date)
        return forecast[['ds', 'yhat']]

    # def create_




if __name__ == '__main__':
    ic = InformatiCup2018(data_dir="/Users/ole/InformatiCup2018/")
    data = ic.read_gas_station_data(1)

    # predict_dates = list(map(str, data['ds'].tolist()[-5:]))
    # predict_dates = pd.date_range('9/21/2016', periods=365*24, freq='H')
    # print(data.tail())
    # print(predict_dates)
    # ic.create_model_for_gas_station(1)
    # print(ic.predict_prices(1, predict_dates))
    # ic.init_all()
    # ic.init_gas_stations()
    # print(ic.gas_stations)