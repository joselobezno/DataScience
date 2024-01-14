import pandas as pd 
import string
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split



os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
PUNCTUATION = string.punctuation

#lectura de los documentros y creación del DataFrame
def read_data(path:str)->pd.DataFrame:
    """
    Función para leer los archivos csv
    
    Argumentos: path -> dirección de ubicación del archivo

    retorna: pd.DataFrame -> Pandas DataFrame realizado
    """
    try:
        leer = pd.read_csv(path,sep=",",encoding="latin-1")
        return leer
    except FileNotFoundError: 
        return print("no se encuentra el error - verificar la ruta (path)")
    except:
        return print("ha ocurrido un error inesperado")

#Unión de los 2 DataFrames
def merge(df1:pd.DataFrame,df2:pd.DataFrame, metodo:str,sobre:str) ->pd.DataFrame:
    """
    Función para unir los DataFrames creados

    Args: 
    df1: - Es el DataFrame principal y sobre el cuál se aplicará el Merge
    df2: - Es el DataFrame secundario que es el que se anexará en el df1
    metodo: - Es el metodo de join que se aplicará "left,inner,right,outer..."
    sobre: - Es la columna pivote y sobre la cual se aplcará el join.

    retorna:
    pd.DataFrame: Retorna el DataFrame resultante de la unión de los otros dos.
    """
    df_merge = df1.merge(df2,how=metodo,on=sobre)  
    
    return df_merge.dropna().reset_index()


def clean(df:pd.DataFrame)->pd.DataFrame:
    """
    El objetivo de esta función es hacer la limpieza y preprocesado de la data
    
    input: df: pd.DataFrame - Es el DataFrame previo al procesamiento
    
    output: df: pd.DataFrame - Es el DataFrame ya procesado
    """
#Primero se eliminan la columna App y Translated Review que no van a ser usadas en el modelo
    df.drop(columns=['Translated_Review','App'], inplace=True)
    df.dropna(inplace=True)

#Se inicia con el preprocesado de la variable "Genres"
    df["Genres"].apply(lambda x: x.lower())
    df["Genres"].apply(lambda x: ''.join(letter for letter in x if not letter in PUNCTUATION))
    tokenize = Tokenizer(num_words=82)
    tokenize.fit_on_texts(df["Genres"])
    secuencias = tokenize.texts_to_sequences(df["Genres"])
    secuencias_padding = pad_sequences(secuencias,3,padding="post",truncating="post")
    df_Genres_sequences = pd.DataFrame(secuencias_padding)
    df_Genres_sequences.rename(columns={0:"Genres1",1:"Genres2",2:"Genres3",3:"Genres4"},inplace=True)
    df_Genres_sequences
    df = df.join(df_Genres_sequences)
    df.drop(columns='Genres',inplace=True)
    df.dropna(inplace=True)
    
    # Limpieza de variables "Type" y "Sentiment"
    replaces = {'Free':1,'Paid':0,'Positive':1,'Neutral':0,'Negative':-1}
    df.replace(replaces, regex=True, inplace=True)

#Procesamiento de variables normales "Reviews","Size","Installs","Price"
    
    #Procesando Reviews
    df["Reviews"] = df["Reviews"].astype('float64')
    
    #Procesando Size
    df["Size"].replace("M","000",regex=True, inplace=True)
    df["Size"].replace("k","",regex=True, inplace=True)
    df["Size"].replace("1,000+",1,regex=True, inplace=True)
    df["Size"].replace("Varies with device",df["Size"][df["Size"]!="Varies with device"].apply(lambda x: float(x)).mean(),regex=True, inplace=True)
    df["Size"] = df["Size"].astype('float64')
    
    #Procesamiento de Installs
    df['Installs'] = df['Installs'].apply(lambda x: x.replace("+",""))
    df['Installs'].replace(",","",regex=True, inplace=True)
    df['Installs'] = df['Installs'].astype("float64")
    
    #Procesamiento de Price
    df["Price"] = df["Price"].apply(lambda x: x.replace('$',''))
    df["Price"] = df["Price"].astype('float64')

#Limpieza de Variables Categóricas con Label Encoding
    
    #Limpieza de "Category","Content Rating","Current Ver", "Android Ver"
    model_encoding = LabelEncoder()
    
    def label_encoding(df:pd.DataFrame,column:str)-> pd.DataFrame:
        df[column] = df[column].astype('str')
        df[column] = model_encoding.fit_transform(df[column])
        return df
    label_encoding(df,"Category")
    label_encoding(df,"Content Rating")
    label_encoding(df,"Current Ver")
    label_encoding(df,"Android Ver")

    #Limpieza de Fecha "Last Update"
    """
    Se transformará al número de meses desde su ultima actualización
    
    """
    months = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}

    df['Last Updated'] = df['Last Updated'].replace(',','',regex=True).apply(lambda x: x.split())
    for i in df['Last Updated']:
        if i[0] in months:
            i[0]= str(months[i[0]])
    df['Last Updated'] = df['Last Updated'].apply(lambda x: ' '.join(x).replace(' ','/') )

    from datetime import datetime
    df['Last Updated'] = df['Last Updated'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y'))

    df['Last Updated'] = (datetime.now() - df['Last Updated']).dt.days/30

    #Un ultimo dropna para eliminar cualquier tipo de null generados en las transformaciones
    df.dropna(inplace=True)
    return df





