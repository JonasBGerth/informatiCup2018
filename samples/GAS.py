from samples import main
import pyflux as pf

fuelPrice = main.fuelPrice['2016-12']
H = 50


model = pf.GAS(ar = 2, sc = 2, data = fuelPrice, family = pf.Normal())
x = model.fit("MLE")
x.summary()

model.plot_z(figsize =(15,5))
model.plot_fit(figsize = (15,5))

model.plot_predict_is(h=H, figsize=(15,5))
model.plot_predict(h=H, figsize=(15,5))