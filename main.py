from fastapi import FastAPI, File, UploadFile, Query, Body
from pydantic import BaseModel
from typing import List, Optional
import pickle
import re
import pandas as pd
import numpy as np

# вспомогательная функция 
def torque_split(text: str) -> tuple:
    if text != None:
        finds = re.findall(r'[0-9,.\-\+\/]+', text)
        if len(finds) <= 1:
            torque = finds[0]
            max_torque_rpm = None
        else:
            if 'kgm' in text:
                try:
                    torque = float(finds[0])*9.80665
                except:
                    print(float(finds[0]))
            else:
                torque = float(finds[0])
            if '+/-' in finds[1]:
                max_torque_rpm = finds[1].split('+/-')[0].replace(",", "")
            elif '-' in finds[1]:
                max_torque_rpm = finds[1].split('-')[1].replace(",", "")
            elif '/' in finds[1]:
                max_torque_rpm = finds[1].replace('/','')

            else:
                max_torque_rpm = finds[1].replace(",", "")
        return (torque, max_torque_rpm)

def brand(s: str):
    w = s.split(' ')
    return w[0]

def dict_from_class(cls):
    return dict(
        (key, value)
        for (key, value) in cls.__dict__.items()
        )


app = FastAPI()


scalerfile = 'scaler.sav'
scaler = pickle.load(open(scalerfile, 'rb'))

encoder = pickle.load(open('encoder.pkl', 'rb'))

model = pickle.load(open('model', 'rb'))

medians = pickle.load(open('medians', 'rb'))



class Item(BaseModel):
    name: str
    year: int
    # selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: Optional[str] 
    engine: Optional[str]
    max_power: Optional[str]
    torque: Optional[str]
    seats: Optional[float]
    


class Items(BaseModel):
    objects: List[Item]

@app.get('/')
def root():
    return 'Home'

@app.post("/predict_item")
def predict_item(item: Item) -> float:

    df = pd.DataFrame(item.dict(), index=[0,])
    
    df.loc[0,'brand'] = brand(df.loc[0,'name'])
    df.loc[0, ['torque', 'max_torque_rpm']]  = (torque_split(df.loc[0, 'torque']))
    df.replace('', np.nan, inplace=True)
    float_cols = ['mileage', 'engine', 'max_power', 'torque','max_torque_rpm']
    for col in ['mileage', 'engine', 'max_power']:
        df[col] = df[col].apply(lambda x: x.split(' ')[0] if type(x)!=float else x) # оставляем только числа
    df[float_cols] = df[float_cols].astype('float')
    
    

    df.loc[0, ['km_driven', 'mileage', 'engine', 'max_power', 'torque','seats','max_torque_rpm']] = scaler.transform(df.loc[0, ['km_driven', 'mileage', 'engine', 'max_power', 'torque','seats','max_torque_rpm']].values.reshape(1, -1))
    df.loc[0,'new_torque'] = df.loc[0,'torque']*df.loc[0, 'max_torque_rpm']
    df = pd.concat([df.drop(columns=['name', 'year', 'fuel', 'seller_type', 'transmission', 'owner', 'seats', 'brand']),pd.DataFrame(encoder.transform(df.loc[0, ['year', 'fuel', 'seller_type', 'transmission', 'owner', 'seats', 'brand']].values.reshape(1, -1)))], axis=1)
    df.columns=df.columns.astype(str)
    pred = model.predict(df.iloc[0].values.reshape(1, -1))
    return pred[0]

import json
@app.post("/predict_items")
def predict_items(items: Items):
    keys = ['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission',
            'owner', 'mileage', 'engine', 'max_power', 'torque', 'seats']
    init_df = pd.DataFrame([list(dict(v).values()) for v in items.__dict__['objects']], columns=keys)
    df = init_df.copy()
    df['brand'] = df['name'].apply(brand)



    df[['torque', 'max_torque_rpm']]  = pd.DataFrame(df['torque'].apply(torque_split).tolist(), index=df.index)
    df.replace('', np.nan, inplace=True)

    na_cols = ['mileage', 'engine', 'max_power', 'torque', 'seats', 'max_torque_rpm']

    df[na_cols] = df[na_cols].fillna(medians)
    float_cols = ['mileage', 'engine', 'max_power', 'torque','max_torque_rpm']
    for col in ['mileage', 'engine', 'max_power']:
        df[col] = df[col].apply(lambda x: x.split(' ')[0] if type(x)!=float else x) 
    df[float_cols] = df[float_cols].astype('float')
    


    df[['km_driven', 'mileage', 'engine', 'max_power', 'torque','seats','max_torque_rpm']] = scaler.transform(df[['km_driven', 'mileage', 'engine', 'max_power', 'torque','seats','max_torque_rpm']])
    df['new_torque'] = df['torque']*df['max_torque_rpm']
    df = pd.concat([df.drop(columns=['name', 'year', 'fuel', 'seller_type', 'transmission', 'owner', 'seats', 'brand']),pd.DataFrame(encoder.transform(df[['year', 'fuel', 'seller_type', 'transmission', 'owner', 'seats', 'brand']]))], axis=1)
    df.columns=df.columns.astype(str)

    init_df['predictions'] = model.predict(df)
    init_df.replace({float('nan') : None}, inplace=True)

    
    return init_df.to_dict(orient='records')