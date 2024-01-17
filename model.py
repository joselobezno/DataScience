from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np 
import pandas as pd 

def transform(df:pd.DataFrame,y:str)->pd.DataFrame:
    """
    Split y definición de grupos de entrenamiento y prueba.
    """
    X = df.drop(columns=y)
    y = df[y]
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=4)
    
    #std = StandardScaler()
    #X_train = std.fit_transform(X_train)
    #X_test = std.transform(X_test)


    return X_train,X_test,y_train,y_test

def tunning(X_train:pd.DataFrame,y_train:pd.DataFrame)->str:
    
    """
    Los hiperparametros que se testearan son los sigueientes:

    fit_intecept: indica si se ajusta a la intecepción del modelo 
    -Valores: [True,False]
    n_jobs: indica los procesadores que se usaran para realizar el ajuste del
    hiperparametro
    - Valores: [-1,1,2]
    """

    params = {"fit_intercept":[True,False],
            "n_jobs":[-1,1,2]}

    pipe = Pipeline(["std",StandardScaler(),
                    "lr",LinearRegression()])

    gridmodel = GridSearchCV(LinearRegression(),
                params,
                scoring='neg_mean_absolute_error',
                cv=5)


    gridmodel.fit(X_train, y_train)
    
    return gridmodel

def boosting(X_train:pd.DataFrame,y_train:str)->{str,np}:
    
    boosting = GradientBoostingRegressor(loss="squared_error",n_estimators=500, max_depth=100, random_state=4).fit(X_train,y_train)
    
    return boosting

def test_models(X_test:pd.DataFrame,y_test:pd.DataFrame,model1:["modelo Tunning"],model2:["modelo Boosting"])->str:
    


    y_pred_tunning = model1.predict(X_test)
    y_pred_boosting = model2.predict(X_test)
    test_df = pd.DataFrame({"y_real":y_test,"y_pred_tunning":y_pred_tunning,"y_pred_boosting":y_pred_boosting})


    return print("el modelo de tunning tiene un MSE de: ", mean_squared_error(y_test,y_pred_tunning),
                 "\n Mientas que el modelo de boosting tiene un MSE de: ",mean_squared_error(y_test,y_pred_boosting)),print(test_df) 