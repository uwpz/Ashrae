# ######################################################################################################################
#  Initialize: Libraries, functions, parameters
# ######################################################################################################################

# General libraries, parameters and functions
from os import getcwd, chdir
chdir("../Ashrae")
import sys; sys.path.append(getcwd() + "\\code") #not needed if code is marked as "source" in pycharm
from initialize import *

# Specific libraries
from sklearn.pipeline import Pipeline
import xgboost as xgb


# Specific parameters


# ######################################################################################################################
# Fit
# ######################################################################################################################

# --- Read data  ----------------------------------------------------------------------------------------
# ABT
df_train = pd.read_csv(dataloc + "train.csv", parse_dates=["timestamp"], dtype={'meter': object})
df_train = df_train.set_index("timestamp")
df_train["week"] = df_train.index.week
df_train = df_train.reset_index()
with open("0_etl.pkl", "rb") as file:
    d_vars = pickle.load(file)
df_weather = d_vars["df_weather"]
df_building = d_vars["df_building"]
df = (df_train.merge(df_building, how="left", on=["building_id"])
      .merge(df_weather, how="left", on=["site_id", "timestamp"]))

# Read metadata
df_meta_sub = d_vars["df_meta_sub"]

# Define Target
df["target"] = np.log(df["meter_reading"] + 1)
df["target_iszero"] = np.where(df["meter_reading"] == 0, 1, 0)

# Define data to be used for target encoding
np.random.seed(1)
tmp = np.random.permutation(df["week"].unique())
weeks_util = tmp[:5]
weeks_test = tmp[5:15]
weeks_train = tmp[15:]
df["fold"] = np.where(np.isin(df["week"].values, weeks_util), "util",
                      np.where(np.isin(df["week"], weeks_test), "test", "train"))
print(df.fold.value_counts())
df["fold_num"] = df["fold"].map({"train": 0, "util": 0, "test": 1})  # Used for pedicting test data
df["encode_flag"] = df["fold"].map({"train": 0, "test": 0, "util": 1})  # Used for encoding




# Filter
#df = df.query("target_iszero==0")

df = df.sample(n=int(1e7))
# df_save = df.copy()
# df = df_save.copy()

# Get variable types
metr = df_meta_sub.loc[(df_meta_sub["type"] == "metr") & (df_meta_sub["exclude"] != 1), "variable"].values
cate = df_meta_sub.loc[(df_meta_sub["type"] == "cate") & (df_meta_sub["exclude"] != 1), "variable"].values


# --- ETL -------------------------------------------------------------------------------------------------
pipeline_etl = Pipeline([
    ("feature_engineering", FeatureEngineeringAshrae(derive_fe=True)),  # feature engineering
    ("metr_convert", Convert(features=metr, convert_to="float")),  # convert metr to "float"
    ("metr_imp", DfSimpleImputer(features=metr, strategy="median", verbose=1)),  # impute metr with median
    ("cate_convert", Convert(features=cate, convert_to="str")) , # convert cate to "str"
    ("cate_imp", DfSimpleImputer(features=cate, strategy="constant", fill_value="(Missing)")),  # impute cate with const
    ("cate_map_nonexist", MapNonexisting(features=cate)),  # transform non-existing values: Collect information
    ("cate_enc", TargetEncoding(features=cate, encode_flag_column="encode_flag",
                                target="target")),  # target-encoding
    ("cate_map_toomany", MapToomany(features=cate, n_top=30))  # reduce "toomany-members" categorical features
])
df = pipeline_etl.fit_transform(df, df["target"].values, cate_map_nonexist__transform=False)
metr = np.append(metr, pipeline_etl.named_steps["cate_map_toomany"]._toomany + "_ENCODED")

'''
from sklearn.model_selection import GridSearchCV, PredefinedSplit
df_tune = df.sample(n=int(1e6))
scoring = {"spear": make_scorer(spearman_loss_func, greater_is_better=True),
           "rmse": make_scorer(rmse, greater_is_better=False)}
metric = "rmse"

split_index = PredefinedSplit(df_tune["fold"].map({"train": -1, "test": 0}).values)
fit = GridSearchCV(xgb.XGBRegressor(n_jobs = 7),
                   [{"n_estimators": [x for x in range(100, 5100, 1000)], "learning_rate": [0.01],
                     "max_depth": [9,12], "min_child_weight": [10],
                     "subsample": [1], "colsample_by_tree": [0.5],
                     "gamma": [0]}],
                   cv=split_index.split(df_tune),
                   refit=False,
                   scoring=scoring,
                   return_train_score=True,
                   n_jobs=2) \
    .fit(CreateSparseMatrix(metr=metr,
                            cate=cate,
                            df_ref=df_tune).fit_transform(df_tune), df_tune["target"])
df_fitres = pd.DataFrame.from_dict(fit.cv_results_)
df_fitres.mean_fit_time.values.mean()
sns.FacetGrid(df_fitres, col="param_min_child_weight", margin_titles=True) \
    .map(sns.lineplot, "param_n_estimators", "mean_test_" + metric,  # do not specify x= and y=!
         hue="#" + df_fitres["param_max_depth"].astype('str'),  # needs to be string not starting with "_"
         style=df_fitres["param_learning_rate"],
         marker="o").add_legend()
'''


# --- Fit -----------------------------------------------------------------------------------------------
# Fit
pipeline_fit = Pipeline([
    ("create_sparse_matrix", CreateSparseMatrix(metr=metr, cate=cate, df_ref=df)),
    ("clf", xgb.XGBRegressor(n_estimators=5100, learning_rate=0.01,
                             max_depth=15, min_child_weight=10,
                             colsample_bytree=0.5, subsample=1,
                             gamma=0,
                             n_jobs=14))
])
fit = pipeline_fit.fit(df, df["target"].values)
# yhat = pipeline_fit.predict_proba(df)  # Test it


# --- Save ----------------------------------------------------------------------------------
with open("productive.pkl", "wb") as f:
    pickle.dump({"pipeline_etl": pipeline_etl,
                 "pipeline_fit": pipeline_fit}, f)
