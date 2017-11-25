from typing import List

import pandas as pd

import os
import re
from fbprophet import Prophet


class InformatiCup2018(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.gas_stations_dir = data_dir + os.path.join('Eingabedaten', 'Benzinpreise')
        self.gas_stations_dfs = {}
        self.gas_stations_models = {}

    def read_gas_station_data(self, gas_station_id: int):
        """"""
        data = pd.read_csv(os.path.join(self.gas_stations_dir, f'{gas_station_id}.csv'),
                           sep=';', header=None,
                           names=['ds', 'y'])

        data['ds'] = pd.to_datetime(data['ds'])
        data['y'] = pd.to_numeric(data['y'], downcast='float')
        self.gas_stations_dfs[gas_station_id] = data
        return data

    def read_all_gas_stations(self):
        print(os.listdir(self.data_dir + os.path.join('Eingabedaten', 'Benzinpreise')))
        for file in os.listdir(self.gas_stations_dir):
            m = re.search(r'(.*)[.]', file)
            id = int(m.group(1))
            print(id)

            self.gas_stations_dfs[id] = self.read_gas_station_data(id)

    def create_model_for_gas_station(self, gas_station_id: int):
        m = Prophet()
        # TODO: Add holidays to each gas station

        m.fit(self.gas_stations_dfs[gas_station_id])
        self.gas_stations_models[gas_station_id] = m
        return m

    def predict_prices(self, gas_station_id: int, dates: List[str]):
        predict_date = pd.DataFrame()
        predict_date['ds'] = pd.to_datetime(dates)
        forecast = self.gas_stations_models[gas_station_id].predict(predict_date)
        return forecast[['ds', 'yhat']]


if __name__ == '__main__':
    ic = InformatiCup2018(data_dir="/Users/ole/InformatiCup2018/")

    data = ic.read_gas_station_data(1)
    print(data.tail())

    ic.create_model_for_gas_station(1)
    print(ic.predict_prices(1, [
        '2017-09-21 11:54:16',
        '2017-09-21 12:00:00',
        '2017-09-21 13:39:06',
        '2017-09-21 15:13:05',
        '2017-09-21 15:47:05',
        '2017-09-21 21:03:06'
    ]))
