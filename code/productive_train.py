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
# site_id filter

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
df = df.set_index("timestamp")
df["week"] = df.index.week
df = df.reset_index()
np.random.seed(1)
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

# df = df.sample(n=int(1e7))
# df_save = df.copy()
# df = df_save.copy()

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

# Filter "main gaps"
df = df.set_index("timestamp")
df["dayofyear"] = df.index.dayofyear
df = df.reset_index()
mask = ~(((df["site_id"] == 0) & (df["dayofyear"] <= 141)) |
         ((df["site_id"] == 15) & (df["dayofyear"].between(42, 88))))
df = df.loc[mask]

'''
#import dill
#dill.dump_session("tmp.pkl")
#dill.load_session("tmp.pkl")
                
df["site_id"].value_counts()

site_id = "9"
from sklearn.model_selection import GridSearchCV, PredefinedSplit
df_tune = df.query("site_id == '"+ site_id + "'").sample(n=int(1e6))
scoring = {"spear": make_scorer(spearman_loss_func, greater_is_better=True),
           "rmse": make_scorer(rmse, greater_is_better=False)}
metric = "spear"

split_index = PredefinedSplit(df_tune["fold"].map({"train": -1, "test": 0}).values)
fit = GridSearchCV(xgb.XGBRegressor(n_jobs = 7),
                   [{"n_estimators": [x for x in range(50, 650, 100)], "learning_rate": [0.01],
                     "max_depth": [9], "min_child_weight": [10],
                     "subsample": [1], "colsample_by_tree": [1],
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
fig = sns.FacetGrid(df_fitres, col="param_colsample_by_tree", margin_titles=True) \
    .map(sns.lineplot, "param_n_estimators", "mean_test_" + metric,  # do not specify x= and y=!
         hue="#" + df_fitres["param_max_depth"].astype('str'),  # needs to be string not starting with "_"
         style=df_fitres["param_learning_rate"],
         marker="o").add_legend()
fig.savefig(plotloc + "tune_" + site_id + "_" + metric + ".pdf")

         
'''


# --- Fit -----------------------------------------------------------------------------------------------

# TODO: by my_site_id: lump site_id: 11,7,10,12,6,1 to my_site_id = 99
#df.groupby("site_id")["building_id"].nunique().sort_values()
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
