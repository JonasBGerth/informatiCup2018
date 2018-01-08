from typing import List

import pandas as pd

import os
import re
from fbprophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
import math


def dist(a, b):
    return 6378.388 * math.acos((math.sin(math.radians(a[1])) * math.sin(math.radians(b[1]))) + math.cos(math.radians(a[1])) * math.cos(
        math.radians(b[1])) * math.cos(math.radians(b[0]) - math.radians(a[0])))


class InformatiCup2018(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.gas_stations_data_dir = data_dir + os.path.join('Eingabedaten', 'Benzinpreise')
        self.gas_stations_dir = data_dir + os.path.join('Eingabedaten', 'Tankstellen.csv')
        self.routes = data_dir + os.path.join('Eingabedaten', 'Fahrzeugrouten')
        self.gas_stations = {}
        self.gas_stations_dfs = {}
        self.gas_stations_models = {}

    def init_all(self):
        self.read_all_gas_stations()
        self.create_models_for_all_gas_stations()

    def init_gas_stations(self):
        self.gas_stations = pd.read_csv(self.gas_stations_dir, sep=';', header=None, index_col='id',
                                        names=['id', 'name', 'brand', 'street', 'house_no', 'plz', 'town', 'Long',
                                               'Lat'])

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
        """Predict a price for a gas station at specific dates/times."""
        predict_date = pd.DataFrame()
        predict_date['ds'] = pd.to_datetime(dates)
        forecast = self.gas_stations_models[gas_station_id].predict(predict_date)
        return forecast[['ds', 'yhat']]

    def get_all_route_files(self):
        """Return a list of all route files inside the `Eingabedaten/Fahrzeugrouten` folder. """
        files = []
        for file in os.listdir(self.routes):
            files.append(os.path.join(self.routes,file))

        return files

    def calculate_tank_stops(self, file_name, output_file_name):
        starting_fuel = 0.0
        dates = []
        gas_station_ids = []

        # Read initial gas load
        with open(file_name, 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                try:
                    starting_fuel = float(line.replace(';', ''))
                except ValueError:
                    dates.append(line.split(';')[0])
                    gas_station_ids.append(line.split(';')[1])

        # Read route information
        # Time is the date of arrival at the gas station
        # Hash tag is the id of the gas station
        route = pd.read_csv(file_name, delimiter=';', skiprows=1, names=["Time", "#"])
        self.init_gas_stations()

        # predict prices for given datetimes and gas stations. This will take a while!
        r = []
        for index, row in route.iterrows():
            self.read_gas_station_data(row['#'])
            self.create_model_for_gas_station(row['#'])
            a = self.predict_prices(row['#'],[row['Time']])
            r.append(a[['yhat']].loc[0][0])
            print(row['Time'], row['#'])

        capacity = starting_fuel
        tanks = self.gas_stations[['Long', 'Lat']]

        G = pd.DataFrame(columns=['V', 'E'])

        def station_dist(x=route['#'], y=tanks):
            z = []
            for i in range(len(x) - 1):
                z.append(dist(y.loc[x.loc[i]], y.loc[x.loc[i + 1]]))
            return [0] + z

        # Edge
        G['E'] = station_dist()
        # Vertice(node)
        G['V'] = r
        G['V'] = G['V'].astype(int)

        def cost2(G=G, cap=capacity):
            id_and_price = pd.DataFrame()
            # 0.056 l/km as mentioned in documentation
            l = 0.056
            fill = pd.Series(index=G.index, data=0)
            fill.loc[0] = cap
            actual_gas = cap
            id_and_price['price'] = G['V'].loc[1:]
            id_and_price['dist'] = G['E'].loc[1:]
            # id_and_price['dist_in_liter'] = id_and_price['dist'].values.cumsum()
            # id_and_price.loc[:, 'dist_in_liter'] *= l
            id_and_price.sort_values(inplace=True, by='price')
            j = len(id_and_price)
            i = 0
            # print(id_and_price)
            while i < j:
                g = id_and_price.index[i]
                # print(id_and_price.index[i])
                # 8ка это растояние между 7 и 8
                # print(id_and_price['dist'].loc[id_and_price['dist'].index <= id_and_price.index[i]])
                # we needed to fill 0.49 not cap
                x = (sum(id_and_price['dist'].loc[id_and_price['dist'].index <= id_and_price.index[i]]) * l)
                if x <= actual_gas:
                    # if id_and_price.index[i] == len(G) - 1:
                    # break
                    actual_gas = actual_gas - x
                    lst_to_drop = id_and_price.index[id_and_price.index <= id_and_price.index[i]]
                    id_and_price.drop(labels=lst_to_drop, axis=0, inplace=True)
                    # print(id_and_price)
                    k = 0
                    j = len(id_and_price)
                    while k < j:
                        x = (sum(id_and_price['dist'].loc[id_and_price['dist'].index <= id_and_price.index[k]]) * l)
                        # print(id_and_price.index[k])
                        if x <= cap:
                            actual_gas = actual_gas + x
                            fill.loc[g] = x
                            print(fill)
                            break
                        else:
                            k = k + 1
                    i = k
                    j = len(id_and_price)
                else:
                    i = i + 1
            return fill

        G['F'] = cost2()
        route_result = route.copy()
        route_result['Price'] = G['V']
        route_result['Fuel'] = G['F']
        route_result.to_csv(output_file_name, sep=';', header=None, index=None, decimal=',')

    def write_predicted_prices_to_csv(self, file, predicted_prices, gas_stations, starting_fuel):
        with open(file, 'w') as f:
            for i, gs in enumerate(gas_stations):
                line = ''
                line += str(predicted_prices[i]['ds'].get(0)) + ';'
                line += str(gs) + ';'
                line += "{:.0f}".format(predicted_prices[i]['yhat'].get(0)) + ';\n'

                f.write(line)


if __name__ == '__main__':
    ic = InformatiCup2018(data_dir="/Users/ole/InformatiCup2018/")
    # data = ic.read_gas_station_data(1)
    # for f in ic.get_all_route_files():
    ic.calculate_tank_stops('/Users/ole/InformatiCup2018/Eingabedaten/Fahrzeugrouten/Bertha Benz Memorial Route.csv',
                            '/Users/ole/InformatiCup2018/Ausgabedaten/Tankstrategien/Berta1.csv')
    # ic.predict_route_prices('/Users/ole/InformatiCup2018/Eingabedaten/Fahrzeugrouten/Bertha Benz Memorial Route.csv')
    # ic.predict_route_prices('/Users/ole/Berta2.csv')
    # predict_dates = list(map(str, data['ds'].tolist()[-5:]))
    # predict_dates = pd.date_range('9/21/2016', periods=365*24, freq='H')
    # print(data.tail())
    # print(predict_dates)
    # ic.create_model_for_gas_station(1)
    # print(ic.predict_prices(1, predict_dates))
    # ic.init_all()
    # ic.init_gas_stations()
    # print(ic.gas_stations)
