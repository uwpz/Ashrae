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
from sklearn.pipeline import Pipeline
import xgboost as xgb

# Specific parameters
n_jobs = 4
plt.ion(); matplotlib.use('TkAgg')


# ######################################################################################################################
# Fit
# ######################################################################################################################

# --- Read data  ----------------------------------------------------------------------------------------

# df_train
df_train = pd.read_csv(dataloc + "train.csv", parse_dates=["timestamp"], dtype={'meter': object})

# Add weather and building
with open("0_etl.pkl", "rb") as file:
    d_vars = pickle.load(file)
df_weather = d_vars["df_weather"]
df_building = d_vars["df_building"]
df = (df_train.merge(df_building, how="left", on=["building_id"])
      .merge(df_weather, how="left", on=["site_id", "timestamp"]))

# Define Target
df["target"] = np.log(df["meter_reading"] + 1)
# df["target_iszero"] = np.where(df["meter_reading"] == 0, 1, 0)

# Define data to be used for target encoding
df["week"] = df["timestamp"].dt.week
np.random.seed(999)
tmp = np.random.permutation(df["week"].unique())
weeks_util = tmp[:5]
weeks_test = tmp[5:15]
weeks_train = tmp[15:]
df["fold"] = np.where(np.isin(df["week"].values, weeks_util), "util",
                      np.where(np.isin(df["week"], weeks_test), "test", "train"))
print(df.fold.value_counts())
df["encode_flag"] = df["fold"].map({"train": 0, "test": 0, "util": 1})  # Used for encoding


# --- Get Metadata  ----------------------------------------------------------------------------------------

# Read metadata
df_meta_sub = d_vars["df_meta_sub"]

# Get variable types
metr = df_meta_sub.loc[(df_meta_sub["type"] == "metr") & (df_meta_sub["exclude"] != 1), "variable"].values
cate = df_meta_sub.loc[(df_meta_sub["type"] == "cate") & (df_meta_sub["exclude"] != 1), "variable"].values


# --- ETL -------------------------------------------------------------------------------------------------

# df_save = df.copy()
#df = df_save.copy()
df = df.sample(n=int(1e4)).reset_index(drop = True)

# Remove "main gaps"
df["dayofyear"] = df["timestamp"].dt.dayofyear
mask = ~(((df["site_id"] == 0) & (df["dayofyear"] <= 141)) |
         ((df["site_id"] == 15) & (df["dayofyear"].between(42, 88))))
df = df.loc[mask].reset_index(drop = True)

'''
df["building_id_copy"] = df["building_id"]
pipeline_tmp = Pipeline([
    ("feature_engineering", FeatureEngineeringAshrae(derive_fe=True)),  # feature engineering
    ("target_scale", ScaleTarget(target = "target", target_newname = "target_zscore",
                                 group_cols = ["building_id", "meter"], winsorize_quantiles = [0.001, 0.999]))
])
df_tmp = pipeline_tmp.fit_transform(df, df["target"].values)
df_tmp["target"].hist(bins=50)
df_tmp["target_zscore"].hist(bins=50)
df_tmp.building_id.value_counts()
'''

# ETL
pipeline_etl = Pipeline([
    ("feature_engineering", FeatureEngineeringAshrae(derive_fe=True)),  # feature engineering
    ("target_scale", ScaleTarget(target = "target", target_newname = "target_zscore",
                                 group_cols = ["building_id", "meter"], winsorize_quantiles = [0.001, 0.999])),
    ("metr_convert", Convert(features=metr, convert_to="float")),  # convert metr to "float"
    ("metr_imp", DfSimpleImputer(features=metr, strategy="median", verbose=1)),  # impute metr with median
    ("cate_convert", Convert(features=cate, convert_to="str")),  # convert cate to "str"

    ("cate_imp", DfSimpleImputer(features=cate, strategy="constant", fill_value="(Missing)")),  # impute cate with const
    ("cate_map_nonexist", MapNonexisting(features=cate)),  # transform non-existing values: Collect information
    ("cate_enc", TargetEncoding(features=cate, encode_flag_column="encode_flag",
                                target="target_zscore")),  # target-encoding
    ("cate_map_toomany", MapToomany(features=cate, n_top=30))  # reduce "toomany-members" categorical features
])
df = pipeline_etl.fit_transform(df, df["target"].values, cate_map_nonexist__transform=False)
df["target"].hist(bins=50)
df["target_zscore"].hist(bins=50)
metr = np.append(metr, pipeline_etl.named_steps["cate_map_toomany"]._toomany + "_ENCODED")


'''
#import dill
#dill.dump_session("tmp.pkl")
#dill.load_session("tmp.pkl")
                
df["site_id"].value_counts()

from sklearn.model_selection import GridSearchCV, PredefinedSplit

#df_tune = df.query("site_id == '9'").sample(n=int(1e5)).reset_index(drop = True)
df_tune = df.sample(n=int(1e3)).reset_index(drop = True)

TARGET_TYPE = "REGR"
target = "target_zscore"
metric = "spear"

split_index = PredefinedSplit(df_tune["fold"].map({"train": -1, "test": 0}).values)

start = time.time()
fit = (GridSearchCV_xlgb(xgb.XGBRegressor(verbosity = 0),
                         {"n_estimators": [x for x in range(100, 3100, 500)], "learning_rate": [0.01],
                          "max_depth": [6, 9], "min_child_weight": [10],
                          "colsample_bytree": [0.7], "subsample": [0.7]},
                         cv = split_index.split(df_tune),
                         refit = False,
                         scoring = d_scoring[TARGET_TYPE],
                         return_train_score = True,
                         n_jobs = n_jobs)
       .fit(CreateSparseMatrix(metr = metr, cate = cate, df_ref = df_tune).fit_transform(df_tune),
            df_tune[target]))
print(time.time()-start)
pd.DataFrame(fit.cv_results_)
plot_cvresult(fit.cv_results_, metric = metric,
              x_var = "n_estimators", color_var = "max_depth", column_var = "min_child_weight")
         
'''


# --- Fit -----------------------------------------------------------------------------------------------

# OLD TODO: by my_site_id: lump site_id: 11,7,10,12,6,1 to my_site_id = 99
#df.groupby("site_id")["building_id"].nunique().sort_values()
# Fit
pipeline_fit = Pipeline([
    ("create_sparse_matrix", CreateSparseMatrix(metr=metr, cate=cate, df_ref=df)),
    ("clf", xgb.XGBRegressor(n_estimators=2600, learning_rate=0.01,
                             max_depth=6, min_child_weight=10,
                             colsample_bytree=0.7, subsample=0.7,
                             gamma=0,
                             n_jobs=n_jobs))
])
fit = pipeline_fit.fit(df, df["target_zscore"])
#yhat = pipeline_fit.predict(df)  # Test it


# --- Save ----------------------------------------------------------------------------------
with open("productive.pkl", "wb") as f:
    pickle.dump({"pipeline_etl": pipeline_etl,
                 "pipeline_fit": pipeline_fit}, f)
