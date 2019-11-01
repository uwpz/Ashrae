# ######################################################################################################################
# Score
# ######################################################################################################################

# General libraries, parameters and functions
from os import getcwd
import sys; sys.path.append(getcwd() + "\\code") #not needed if code is marked as "source" in pycharm
from initialize import *

# Load pipelines
with open("productive.pkl", "rb") as file:
    d_pipelines = pickle.load(file)

# Read scoring data
df_test = pd.read_csv(dataloc + "test.csv", parse_dates=["timestamp"], dtype={'meter': object})
with open("0_etl.pkl", "rb") as file:
    d_vars = pickle.load(file)
df_weather = d_vars["df_weather"]
df_building = d_vars["df_building"]
df = (df_test.merge(df_building, how="left", on=["building_id"])
      .merge(df_weather, how="left", on=["site_id", "timestamp"]))

# Transform
from datetime import datetime
print(datetime.now())
df = d_pipelines["pipeline_etl"].transform(df)
print(datetime.now())

# Fit
yhat = scale_predictions(d_pipelines["pipeline_fit"].predict(df))
print(datetime.now())

df[["row_id"]].assign(meter_reading=np.exp(yhat) - 1).to_csv(dataloc + "score_3rd_try.csv", index=False)
