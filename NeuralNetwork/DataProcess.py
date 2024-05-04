import pandas as pd
import numpy as np
import json

with open("./NeuralNetwork/modelConfig.json") as arquivo:
    dados_json = json.load(arquivo)

parcelDataTrain= dados_json['parcelDataTrain']

def getSpeiValues(df):
    SpeiValues = df["Series1"].to_numpy()
    SpeiNormalizedValues = (SpeiValues-np.min(SpeiValues))/np.max(SpeiValues-np.min(SpeiValues))

    return SpeiValues, SpeiNormalizedValues

def getMonthValues(df):
    monthValues = df["Data"].to_numpy()

    return monthValues

def splitSpeiData(xlsx):
    df = pd.read_excel(xlsx)
    df.columns = df.columns.str.replace(' ', '')
    
    SpeiValues, SpeiNormalizedValues = getSpeiValues(df)
    monthValues = getMonthValues(df)

    split= int(len(SpeiNormalizedValues)*parcelDataTrain)

    speiTrainData = SpeiNormalizedValues[0:split]
    speiTestData = SpeiNormalizedValues[split:]

    monthTrainData = monthValues[0:split]
    monthTestData = monthValues[split:]

    return speiTrainData, speiTestData, monthTrainData, monthTestData, split
