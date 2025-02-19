import pandas as pd
import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from util.simplePlot  import simplePlot 
from util.simplePlot  import createPieChart 
from util.simplePlot  import createStackPlot 
 

import matplotlib.pyplot as plt


# Indlæs data fra CSV-filen
data = pd.read_csv('../../Data/MatPlotLib-Data/sales.csv')
print(data.head())

if data is not None:
    print("Data er loaded korrekt og er klar til at lave en multi-linjeplot")
    print(data.columns)
   # saleProfit = simplePlot(data) #graf der viser salget og specielt profit over tid.
   # print(saleProfit)

    multiSvin = simplePlot(data) #graf der viser salget af de forskellige produkter over tid.
    print(multiSvin)

    pieChat = createPieChart(data) #pie chart der viser fordelingen af produktsalg
    print(pieChat)

    stackPlot = createStackPlot(data)
    print("Stack plot er genereret")
  
else:
    print("Data er ikke loaded korrekt")



