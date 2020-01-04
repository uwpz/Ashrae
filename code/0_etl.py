
# ######################################################################################################################
#  Initialize: Libraries, functions, parameters
# ######################################################################################################################

# General libraries, parameters and functions
'''
import os, sys
os.chdir("../Ashrae")
sys.path.append(os.getcwd() + "\\code") #not needed if code is marked as "source" in pycharm
'''
from initialize import *

# Specific libraries


# ######################################################################################################################
#  ETL
# ######################################################################################################################

# --- Training -----------------------------------------------------------------------------------------------------

# Read
df_train = pd.read_csv(dataloc + "train.csv", parse_dates=["timestamp"], dtype={'meter': object})
#df_test = pd.read_csv(dataloc + "test.csv", parse_dates=["timestamp"], dtype={'meter': object})

# Sample
df_tmp = pd.read_csv(dataloc + "building_metadata.csv", dtype={'floor_count': object})[["site_id", "building_id"]]
df_tmp.groupby("site_id").nunique()
#df_train = df_train.merge(df_tmp.query("site_id == 4")[["building_id"]], how="inner", on="building_id")
#df_train.shape
np.random.seed(123)
n_buildings = 100
buildings = df_train["building_id"].sample(n_buildings).values
df_train = df_train[df_train["building_id"].isin(buildings)]

# Timeseries processing
'''
# Old solution, but I do not need to resample as no time based FE can be done (whole 2017-2018 is test data)
df_train = (df_train.replace({"meter_reading": {0: np.nan}})  # Replace 0 with missing
           .set_index(['timestamp', 'building_id', 'meter']).unstack(['building_id', 'meter'])
           [["meter_reading"]]
          .ffill()   # fill missing (but not at beginning), alternative: interpolate("linear")
          .resample("H").ffill()  # fill gaps (but not at beginning)
          .stack(['building_id', 'meter']).reset_index())  # removes also na at beginning
'''

# FE
df_train["hour"] = df_train["timestamp"].dt.hour.astype("str").str.zfill(2)
df_train["dayofweek"] = df_train["timestamp"].dt.dayofweek.astype("str")
df_train["weekend"] = np.where(df_train.dayofweek.isin(["5", "6"]), 1, 0).astype("str")
df_train["week"] = df_train["timestamp"].dt.week
df_train["month"] = df_train["timestamp"].dt.month.astype("str").str.zfill(2)
df_train["dayofyear"] = df_train["timestamp"].dt.dayofyear


# --- Target -----------------------------------------------------------------------------------------------------

# Define target
df_train["target"] = np.log(df_train["meter_reading"] + 1)
df_train["target_iszero"] = np.where(df_train["meter_reading"] == 0, 1, 0)

# Scale (before winsorizing) target
df_train["target"].hist(by=df_train["meter"], bins=50)
group_cols = ["building_id", "meter"]
df_train["mean_target"] = df_train.groupby(group_cols)["target"].transform("mean")
df_train["std_target"] = df_train.groupby(group_cols)["target"].transform("std")
df_train["target_zscore"] = (df_train["target"] - df_train["mean_target"]) / df_train["std_target"]
df_train["target_zscore"].describe().round(2)
df_train["target_zscore"].hist(bins=50)

# Winsorize
tmp_quantile = np.quantile(df_train["target_zscore"], np.array([0.01, 0.99]))
df_train["target_zscore"] = df_train["target_zscore"].clip(lower = tmp_quantile[0], upper = tmp_quantile[1])
df_train["target_zscore"].hist(bins=50)


# --- Weather -----------------------------------------------------------------------------------------------------
# Read
df_weather = pd.concat([pd.read_csv(dataloc + "weather_train.csv", parse_dates=["timestamp"],
                                    dtype={'cloud_coverage': object, 'precip_depth_1_hr': object}),
                        pd.read_csv(dataloc + "weather_test.csv", parse_dates=["timestamp"],
                                    dtype={'cloud_coverage': object, 'precip_depth_1_hr': object})])

# Convert to local time
d_site2hourdiff = {0: 4, 1: 0, 2: 7, 3: 4, 4: 7, 5: 0, 6: 4, 7: 4, 8: 4,
                   9: 5, 10: 7, 11: 4, 12: 0, 13: 5, 14: 4, 15: 4}
df_weather["timestamp"] = df_weather["timestamp"] - \
                          pd.to_timedelta(df_weather["site_id"].map(d_site2hourdiff), "hour")

# Timeseries processing: Fill missings and gaps
df_weather = (df_weather.set_index(['timestamp', 'site_id']).unstack('site_id')
              [['air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_direction', 'wind_speed']]
              .ffill().bfill()  # fill missing
              .resample("H").ffill().bfill()  # fill gaps
              .stack().reset_index()
              .merge(df_weather[['site_id', 'timestamp', 'cloud_coverage', 'precip_depth_1_hr']],
                     how="left", on=['site_id', 'timestamp']))  # add (non-imputed) categorical columns

# FE
# TODO for air_temperature: lag1h, lag2h, avg12h, min12h, max12h
'''
sales_lag1d = df_train[ids + ["zsales_log", "onpromotion"]]\
    .rename(columns={"zsales_log": "zsales_log_lag1d", "onpromotion": "onpromotion_lag1d"})\
    .set_index(ids, append=True)
sales_lag7d = df_train[ids + ["zsales_log", "onpromotion"]]\
    .shift(shift, "D")\
    .rename(columns={"zsales_log": "zsales_log_lag7d", "onpromotion": "onpromotion_lag7d"})\
    .set_index(ids, append=True)
sales_avg7d = df_train[ids + ["zsales_log", "onpromotion"]]\
    .set_index(ids, append=True).unstack(ids)\
    .rolling("6D").mean().stack(ids)\
    .rename(columns={"zsales_log": "zsales_log_avg7d", "onpromotion": "onpromotion_avg7d"})
sales_avg4sameweekdays = df_train.groupby("dayofweek").apply(
    lambda x: x[ids + ["zsales_log", "onpromotion"]].set_index(ids, append=True).unstack(ids)
              .shift(shift, "D").rolling("21D").mean().stack(ids)) \
    .reset_index("dayofweek", drop=True) \
    .rename(columns={"zsales_log": "zsales_log_avg4sameweekdays", "onpromotion": "onpromotion_avg4sameweekdays"})
sales_avg12sameweekdays = df_train.groupby("dayofweek").apply(
    lambda x: x[ids + ["zsales_log", "onpromotion"]].set_index(ids, append=True).unstack(ids)
              .shift(shift, "D").rolling("77D").mean().stack(ids)) \
    .reset_index("dayofweek", drop=True) \
    .rename(columns={"zsales_log": "zsales_log_avg12sameweekdays", "onpromotion": "onpromotion_avg12sameweekdays"})
'''

# --- Building -----------------------------------------------------------------------------------------------------

df_building = pd.read_csv(dataloc + "building_metadata.csv", dtype={'floor_count': object})
df_building['sq_floor'] = df_building['square_feet'] / df_building['floor_count'].astype("float")


########################################################################################################################
# Prepare final data
########################################################################################################################

# Merge all together
df = (df_train.merge(df_building, how="left", on=["building_id"])
      .merge(df_weather, how="left", on=["site_id", "timestamp"]))

'''
# Some checks
#fig, ax = plt.subplots(4,4)
(sns.heatmap(df.groupby(["dayofyear", "meter", "site_id", "building_id", "meter"])["target_iszero"].mean()
    .unstack("dayofyear")))
df.groupby("site_id")["building_id"].nunique().sort_values()
for site_id in range(16):
    fig, ax = plt.subplots(1, 1)
    sns.heatmap(df.loc[df["site_id"] == site_id]
                .groupby(["dayofyear", "site_id", "meter", "building_id"])["target_iszero"].mean().unstack("dayofyear"))
# site_id=0: <=141, site_id=15: >= 42 <=88 -> remove

df.groupby(["building_id", "meter"])["target"].count().reset_index()[["target"]].boxplot()
(df.loc[df["building_id"].isin(df["building_id"].sample(10).values), :]
    .boxplot("target", by = ["building_id", "meter"], showmeans = True))

'''

# --- Read metadata (Project specific) -----------------------------------------------------------------------------
df_meta = (pd.read_excel(dataloc + "DATAMODEL_ashrae.xlsx")
           .assign(variable=lambda x: x["variable"].str.strip()))

# Check
print(setdiff(df.columns.values, df_meta["variable"].values))
print(setdiff(df_meta["variable"].values, df.columns.values))
print(setdiff(df_meta.loc[df_meta["status"] == "ready", "variable"].values, df.columns.values))

# Filter on "ready"
df_meta_sub = df_meta.loc[df_meta["status"].isin(["ready"])].reset_index()


# --- Save image ------------------------------------------------------------------------------------------------------

# Serialize
with open("0_etl.pkl", "wb") as file:
    pickle.dump({"df": df,
                 "df_meta_sub": df_meta_sub,
                 "df_weather": df_weather,
                 "df_building": df_building},
                file)
