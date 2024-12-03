
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd


app = FastAPI()

models = pickle.load(open('models.pickle', 'rb'))
model = models['ridge']
ss = models['scaler']
ohe = models['ohe']



class Item(BaseModel):
    #name: str          # dropped
    year: int           # scaler
    #selling_price: int # target
    km_driven: int      # scaler
    fuel: str           # encoder
    seller_type: str    # encoder
    transmission: str   # encoder
    owner: str          # encoder
    mileage: float      # scaler
    engine: int         # scaler
    max_power: float    # scaler
    #torque: str        # dropped
    seats: int          # scaler/encoder


# class Items(BaseModel):
#     objects: List[Item]


def transform(df_instance):
    df_instance.insert(loc=1, column='selling_price', value=[0]*df_instance.shape[0])

    df_test_light = df_instance.drop(['fuel', 'seller_type', 'transmission', 'owner'], axis = 1)
    X_test_scaled = pd.DataFrame(ss.transform(df_test_light), columns = df_test_light.columns)
    X_test_scaled.drop(['selling_price'], axis = 1, inplace = True)

    df_test_cat = df_instance[['fuel', 'seller_type', 'transmission', 'owner', 'seats']]
    df_test_cat_ohe = ohe.transform(df_test_cat)
    cols_ohe = ohe.get_feature_names_out()
    df_test_cat_ohe = pd.DataFrame(df_test_cat_ohe, columns = cols_ohe)

    X_test_full = pd.concat([X_test_scaled, df_test_cat_ohe], axis = 1)

    return X_test_full


def pydantic_model_to_df(model_instance):
    df_instance = pd.DataFrame([jsonable_encoder(model_instance)])

    return transform(df_instance)


def pydantic_model_to_df_2(model_instances):
    df_instance = pd.DataFrame([jsonable_encoder(model_instances[0])])
    for json_ in model_instances[1:]:
      df_instance = pd.concat([df_instance, pd.DataFrame([jsonable_encoder(json_)])], axis = 0)

    return transform(df_instance)


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df_instance = pydantic_model_to_df(item)
    prediction = model.predict(df_instance)
    return prediction


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    df_instance = pydantic_model_to_df_2(items)
    prediction = model.predict(df_instance)
    return prediction
