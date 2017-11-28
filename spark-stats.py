from __future__ import print_function

import sys
from random import random
from operator import add
import pandas as pd
import sklearn as sk
import statsmodels

from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf


if __name__ == "__main__":
    
    spark = SparkSession\
        .builder\
        .appName("Read Tankstation data")\
        .getOrCreate()

    sc = SparkContext.getOrCreate()

    tankstation = sc.wholeTextFiles("hdfs:///projects/informatiCup/Tankstellen")

    tankstationPandas = tankstation.mapValues(lambda x: pd.read_csv(x, sep=";", header=0, index_col=0))
    tankstationWeeklyMean = tankstationPandas.mapValues(lambda x:   x.drop('Time', axis = 1, inplace = True)\
                                                                    .set_index(pd.DatetimeIndex(x.Time), inplace = True)\
                                                                    .resample('W').mean()\
                                                                    .dropna(inplace = True))

    tankstationWeeklyMean.count()

spark.stop()
sc.stop()
