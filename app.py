
from cleaning import read_data,merge,clean
from model import tunning, boosting
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

tunning_message, tunning_y_pred = tunning(gps_merge_clean,"Rating")

boosting_message, boost_y_pred, y_test,boosting_model = boosting(gps_merge_clean,"Rating","squared_error")

print(tunning_message, "\n",boosting_message)
print(pd.DataFrame({"y_real":y_test,"y_pred_tunning":tunning_y_pred,"y_pred_boosting":boost_y_pred}))

save_model(boosting_model)