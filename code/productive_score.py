# ######################################################################################################################
# Score
# ######################################################################################################################

# General libraries, parameters and functions
'''
import os, sys
os.chdir("../Ashrae")
sys.path.append(os.getcwd() + "\\code") #not needed if code is marked as "source" in pycharm
'''
from initialize import *

# Specific parameters
n_jobs = 14
plt.ion(); matplotlib.use('TkAgg')

# Load pipelines
with open("productive.pkl", "rb") as file:
    d_pipelines = pickle.load(file)
step_scale = d_pipelines["pipeline_etl"].named_steps["target_scale"]

# Read scoring data
df_test = pd.read_csv(dataloc + "test.csv", parse_dates=["timestamp"], dtype={'meter': object})
#df_test = df_test.sample(n=int(1e5)).reset_index(drop = True)
with open("0_etl.pkl", "rb") as file:
    d_vars = pickle.load(file)
df_weather = d_vars["df_weather"]
df_building = d_vars["df_building"]
df = (df_test.merge(df_building, how="left", on=["building_id"])
      .merge(df_weather, how="left", on=["site_id", "timestamp"]))
df = df.merge(step_scale.df_scaleinfo,
              how = "left", on = step_scale.group_cols)

# Transform
start = time.time()
df = d_pipelines["pipeline_etl"].transform(df)
print((time.time()-start)/60)

# Fit
df[step_scale.target_newname] = d_pipelines["pipeline_fit"].predict(df)
df[step_scale.target_newname].hist(bins = 50)
df[step_scale.target] = df[step_scale.target_newname] * df["std_target"] + df["mean_target"]
df[step_scale.target].hist(bins = 50)
print((time.time()-start)/60)

# Write
(df[["row_id"]].assign(meter_reading=np.round(np.exp(df[step_scale.target].values) - 1, 4))
               .to_csv(dataloc + "score.csv", index=False))


df[step_scale.target].describe()