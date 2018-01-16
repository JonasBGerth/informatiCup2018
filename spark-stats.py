'''
    Written by Jonas
    Using Code from Ole and Anna
'''
from __future__ import print_function

import pandas as pd
import numpy as np
from io import StringIO
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error

from pyspark.sql import SparkSession, Row
from pyspark import SparkContext

'''
    Converting data from String to Pandas dataframe
'''
def to_pandas(files):
    readableFiles = StringIO(files)
    gas_stations = pd.read_csv(readableFiles, header = None, sep = ';', names = ('ds', 'y'))
    return gas_stations

'''
    Creating models with prophet and return MSE for the last 30 days 
'''
def createprophet_models(data):
    periods = 30*24
    data['ds'] = pd.to_datetime(data['ds'])
    data['y'] = pd.to_numeric(data['y'], downcast='float')

    # Filter error from Prophet
    history = data[data['y'].notnull()].copy()
    if history.shape[0] < 2:
        return -1.0

    model = Prophet()
    model.fit(data)

    predict_dates = pd.DataFrame(pd.date_range('8/21/2017', periods=periods, freq='H'), columns=['ds'])
    forecast = model.predict(predict_dates)
    
    forecast.set_index(forecast['ds'], inplace=True)
    forecast.drop('ds', inplace=True, axis=1)
    
    data.set_index(data['ds'], inplace=True)
    data = data.resample('H').mean()

    concat_data = pd.concat([data, forecast], axis=1)
    concat_data.dropna(inplace=True)

    if concat_data.empty: 
        return -1.0

    mse = mean_squared_error(y_true=np.log(concat_data['y']), y_pred=np.log(concat_data["yhat"]))

    print(concat_data)
    print(mse)

    return mse.item()

'''
    Run Spark Job 
'''
if __name__ == "__main__":
    
    spark = SparkSession.builder\
        .appName("Read Tankstation data")\
        .getOrCreate()

    sc = SparkContext.getOrCreate()

    tankstation = sc.wholeTextFiles("hdfs:///projects/informatiCup/Benzinpreise/")

    tankstationPandas = tankstation.mapValues(to_pandas)
    tankstationWeeklyMean = tankstationPandas.mapValues(createprophet_models)

    spark.createDataFrame(tankstationWeeklyMean.map(lambda x: Row(gasstation=x[0].split('/')[-1], mse=x[1]))).write.json("hdfs:///projects/informatiCup/mseOut/")

    tankstationWeeklyMean.collect()

    sc.stop()
    spark.stop()
