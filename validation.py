from cleaning import read_data,merge,clean
from model import transform, tunning, boosting, test_models
from deploy import save_model
import pandas as pd

"""
INFORMACIÓN IMPORTANTE
La variable objetivo escogida fue "Rating:float"

WARNING
Recuera que el path te arroja los sepradores "\" pero se deben cambiar por "/"
"""
gps_rating = read_data("C:/Users/jovillalba/Desktop/myproject/data/googleplaystore.csv")
gps_reviews = read_data("C:/Users/jovillalba/Desktop/myproject/data/googleplaystore_user_reviews.csv")

gps_merge = merge(gps_rating,gps_reviews,"left","App")

gps_merge_clean = clean(gps_merge)

gps_merge_clean.drop(columns="index",inplace=True)

X_train,X_test,y_train,y_test = transform(gps_merge_clean,"Rating")


model_tunning = tunning(X_train,y_train)
model_boosting = boosting(X_train, y_train)

test_models(X_test,y_test,model_tunning,model_boosting)


save_model(model_boosting)