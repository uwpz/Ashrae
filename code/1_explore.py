
# ######################################################################################################################
#  Initialize: Libraries, functions, parameters
# ######################################################################################################################

# General libraries, parameters and functions
from os import getcwd, chdir
chdir("../Ashrae")
import sys; sys.path.append(getcwd() + "\\code") #not needed if code is marked as "source" in pycharm
from initialize import *

# Specific libraries
from scipy.stats.mstats import winsorize

# Main parameter
TARGET_TYPE = "REGR"

# Specific parameters
ylim = (0,10)
cutoff_corr = 0
cutoff_varimp = 0.52
color = None

# Load results from exploration
with open("0_etl.pkl", "rb") as file:
    d_vars = pickle.load(file)
df, df_meta_sub = d_vars["df"], d_vars["df_meta_sub"]

# Train/Test fold: usually split by time
np.random.seed(1)
tmp = np.random.permutation(df["week"].unique())
weeks_util = tmp[:5]
weeks_test = tmp[5:15]
weeks_train = tmp[15:]
df["fold"] = np.where(np.isin(df["week"].values, weeks_util), "util",
                      np.where(np.isin(df["week"], weeks_test), "test", "train"))
#df["fold"] = np.random.permutation(pd.qcut(np.arange(len(df)), q=[0, 0.1, 0.8, 1], labels=["util", "train", "test"]))
print(df.fold.value_counts())
df["fold_num"] = df["fold"].map({"train": 0, "util": 0, "test": 1})  # Used for pedicting test data
df["encode_flag"] = df["fold"].map({"train": 0, "test": 0, "util": 1})  # Used for encoding

# Define the id
df["id"] = np.arange(len(df)) + 1


# ######################################################################################################################
# Metric variables: Explore and adapt
# ######################################################################################################################

# --- Define metric covariates -------------------------------------------------------------------------------------
metr = df_meta_sub.loc[df_meta_sub["type"] == "metr", "variable"].values
df = Convert(features=metr, convert_to="float").fit_transform(df)
df[metr].describe()


# --- Create nominal variables for all metric variables (for linear models) before imputing -------------------------
df[metr + "_BINNED"] = df[metr].apply(lambda x: char_bins(x))

# Convert missings to own level ("(Missing)")
df[metr + "_BINNED"] = df[metr + "_BINNED"].replace("nan", np.nan).fillna("(Missing)")
print(create_values_df(df[metr + "_BINNED"], 11))

# Get binned variables with just 1 bin (removed later)
onebin = (metr + "_BINNED")[df[metr + "_BINNED"].nunique() == 1]
print(onebin)

# --- Missings + Outliers + Skewness ---------------------------------------------------------------------------------
# Remove covariates with too many missings from metr
misspct = df[metr].isnull().mean().round(3)  # missing percentage
misspct.sort_values(ascending=False)  # view in descending order
remove = misspct[misspct > 0.95].index.values  # vars to remove
print(remove)
metr = setdiff(metr, remove)  # adapt metadata

'''

# Check for outliers and skewness
df[metr].describe()
plot_distr(df), metr, target_type=TARGET_TYPE, color=color, ylim=ylim,
           ncol=3, nrow=2, w=18, h=12, pdf=plotloc + TARGET_TYPE + "_distr_metr.pdf")

# Winsorize
df[metr] = df[metr].apply(lambda x: x.clip(x.quantile(0.02),
                                           x.quantile(0.98)))  # hint: plot again before deciding for log-trafo

# Log-Transform
tolog = np.array(["xxx"], dtype="object")
df[tolog + "_LOG_"] = df[tolog].apply(lambda x: np.log(x - min(0, np.min(x)) + 1))
metr = np.where(np.isin(metr, tolog), metr + "_LOG_", metr)  # adapt metadata (keep order)
df.rename(columns=dict(zip(tolog + "_BINNED", tolog + "_LOG_" + "_BINNED")), inplace=True)  # adapt binned version
'''

# --- Final variable information ------------------------------------------------------------------------------------
# Univariate variable importance
varimp_metr = calc_imp(df, metr, target_type=TARGET_TYPE)
print(varimp_metr)
varimp_metr_binned = calc_imp(df, metr + "_BINNED", target_type=TARGET_TYPE)
print(varimp_metr_binned)

# Plot
plot_distr(df, features=np.hstack(zip(metr, metr + "_BINNED")),
           varimp=pd.concat([varimp_metr, varimp_metr_binned]), target_type=TARGET_TYPE, color=color, ylim=ylim,
           ncol=3, nrow=2, w=18, h=12, pdf=plotloc + TARGET_TYPE + "_distr_metr_final.pdf")



# --- Removing variables -------------------------------------------------------------------------------------------
# Remove leakage features
remove = ["xxx", "xxx"]
metr = setdiff(metr, remove)

# Remove highly/perfectly (>=98%) correlated (the ones with less NA!)
df[metr].describe()
plot_corr(df, metr, cutoff=cutoff_corr, pdf=plotloc + TARGET_TYPE + "_corr_metr.pdf")
remove = ["xxx", "xxx"]
metr = setdiff(metr, remove)


# --- Predict target_iszereo -------------------------------------------------------------------------------------------
# Univariate variable importance (again ONLY for non-missing observations!)
varimp_metr_iszero = calc_imp(df, metr, "target_iszero")

# Plot
plot_distr(df, metr, "target_iszero", varimp=varimp_metr_iszero,
           target_type="CLASS",
           ncol=3, nrow=2, w=18, h=12, pdf=plotloc + "distr_metr_target_iszero.pdf")


# --- Missing indicator and imputation (must be done at the end of all processing)------------------------------------
miss = metr[df[metr].isnull().any().values]  # alternative: [x for x in metr if df[x].isnull().any()]
df["MISS_" + miss] = pd.DataFrame(np.where(df[miss].isnull(), "miss", "no_miss"))
df["MISS_" + miss].describe()

# Impute missings with randomly sampled value (or median, see below)
df = DfSimpleImputer(features=miss, strategy="median").fit_transform(df)
df[miss].isnull().sum()


# ######################################################################################################################
# Categorical  variables: Explore and adapt
# ######################################################################################################################

# --- Define categorical covariates -----------------------------------------------------------------------------------
# Nominal variables
cate = df_meta_sub.loc[df_meta_sub.type.isin(["cate"]), "variable"].values
df[cate] = df[cate].astype("str").replace("nan", np.nan)
df[cate].describe()

# Merge categorical variable (keep order)
cate = np.append(cate, ["MISS_" + miss])


# --- Handling factor values ----------------------------------------------------------------------------------------
# Convert "standard" features: map missings to own level
df[cate] = df[cate].fillna("(Missing)")
df[cate].describe()

# Get "too many members" columns and copy these for additional encoded features (for tree based models)
topn_toomany = 30
levinfo = df[cate].apply(lambda x: x.unique().size).sort_values(ascending=False)  # number of levels
print(levinfo)
toomany = levinfo[levinfo > topn_toomany].index.values
print(toomany)
toomany = setdiff(toomany, ["xxx", "xxx"])  # set exception for important variables

# Create encoded features (for tree based models), i.e. numeric representation
df = TargetEncoding(features=cate, encode_flag_column="encode_flag", target="target").fit_transform(df)

# Convert toomany features: lump levels and map missings to own level
df = MapToomany(features=toomany, n_top=topn_toomany).fit_transform(df)

# Univariate variable importance
varimp_cate = calc_imp(df, cate, target_type=TARGET_TYPE)
print(varimp_cate)

# Check
plot_distr(df, cate, varimp=varimp_cate, target_type=TARGET_TYPE,
           color=color, ylim=ylim,
           ncol=3, nrow=2, w=18, h=12, pdf=plotloc + TARGET_TYPE + "_distr_cate.pdf")


# --- Removing variables ---------------------------------------------------------------------------------------------

# Remove highly/perfectly (>=99%) correlated (the ones with less levels!)
plot_corr(df, cate, cutoff=cutoff_corr, n_cluster=5,
          w=12, h= 12, pdf=plotloc + TARGET_TYPE + "_corr_cate.pdf")


# --- Predict target_iszereo -------------------------------------------------------------------------------------------
# Univariate variable importance (again ONLY for non-missing observations!)
varimp_cate_iszero = calc_imp(df, cate, "target_iszero")

# Plot: only variables with with highest importance
plot_distr(df, cate, "target_iszero", varimp=varimp_cate_iszero, target_type="CLASS",
           ncol=3, nrow=2, w=18, h=12, pdf=plotloc + "distr_cate_target_iszero.pdf")


########################################################################################################################
# Prepare final data
########################################################################################################################

# --- Define final features ----------------------------------------------------------------------------------------

exclude = ["dayofweek", "week", "month"]
exclude = exclude + [x + "_ENCODED" for x in exclude] + [x + "_BINNED" for x in exclude]
features_lasso = dict(metr=setdiff(toomany + "_ENCODED", exclude),
                      cate=setdiff(np.concatenate([setdiff(metr + "_BINNED", onebin), setdiff(cate, "MISS_" + miss)]),
                                   exclude))
features_xgb = dict(metr=setdiff(np.concatenate([metr, toomany + "_ENCODED"]),exclude),
                    cate=setdiff(cate, exclude))
features_lgbm = dict(metr=setdiff(metr, exclude),
                     cate=setdiff(cate + "_ENCODED", exclude))

# features = np.concatenate([metr, cate, toomany + "_ENCODED"])
# features_binned = np.concatenate([setdiff(metr + "_BINNED", onebin),
#                                   setdiff(cate, "MISS_" + miss),
#                                   toomany + "_ENCODED"])  # do not need indicators for binned
# features_lgbm = np.append(metr, cate + "_ENCODED")


# --- Remove burned data ----------------------------------------------------------------------------------------
df = df.query("fold != 'util'")


# --- Save image ------------------------------------------------------------------------------------------------------
plt.close(fig="all")  # plt.close(plt.gcf())

# Serialize
with open(TARGET_TYPE + "_1_explore.pkl", "wb") as file:
    pickle.dump({"df": df,
                 "features_lasso": features_lasso,
                 "features_xgb": features_xgb,
                 "features_lgbm": features_lgbm},
                file)
