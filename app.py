import streamlit as st
import pandas as pd
import os
import pickle

st.title("App con CSV")

@st.cache_data
def cargar_datos():
    ruta = os.path.join("data", "horse_limpio.csv")
    df = pd.read_csv(ruta)
    return df

df = cargar_datos()

st.write("Datos cargados:")
st.write(df.head())

@st.cache_resource
def cargar_modelo():
    with open('modelo_equino.pkl', 'rb') as f:
        modelo = pickle.load(f)
    return modelo
modelo =  cargar_modelo()

st.write("Modelo Cargado")
