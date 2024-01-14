from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np 
import pandas as pd 

def tunning(df:pd.DataFrame,y:str)->str:
    
    X = df.drop(columns=y)
    y = df[y]

    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=4)

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
    y_pred = gridmodel.best_estimator_.predict(X_test)
    return print("los mejores parámetros son:",gridmodel.best_params_,"\nEl mejor valor de MSE aplicando Grid Search es ", mean_squared_error(y_test,y_pred)),y_pred

def boosting(df:pd.DataFrame,y:str,loss:str)->{str,np}:
    
    X = df.drop(columns=y)
    y = df[y]

    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=4)
    boosting = GradientBoostingRegressor(loss=loss,n_estimators=500, max_depth=100, random_state=4).fit(X_train,y_train)
    boosting_y_pred = boosting.predict(X_test)
    return print("El MSE despues de aplicado el boosting es de: ", mean_squared_error(y_test,boosting_y_pred)),boosting_y_pred, y_test,boosting