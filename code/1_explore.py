
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

# Main parameter
TARGET_TYPE = "REGR"

# Specific parameters
ylim = (0, 10)
cutoff_corr = 0
cutoff_varimp = 0.52
color = None

# Load results from etl
with open("0_etl.pkl", "rb") as file:
    d_vars = pickle.load(file)
df, df_meta_sub = d_vars["df"], d_vars["df_meta_sub"]


# ######################################################################################################################
# Etl
# ######################################################################################################################

# Train/Test fold: usually split by time
np.random.seed(999)
tmp = np.random.permutation(df["week"].unique())
weeks_test = tmp[:10]
weeks_train = tmp[10:]
df["fold"] = np.where(np.isin(df["week"], weeks_test), "test", "train")
np.random.seed(999)
df["fold"].iloc[np.random.choice(np.arange(len(df)), 10)] = "util"
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

df[metr + "_BINNED"] = df[metr]
df = Binning(features = metr + "_BINNED").fit_transform(df)

# Convert missings to own level ("(Missing)")
df[metr + "_BINNED"] = df[metr + "_BINNED"].fillna("(Missing)")
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


# --- Final variable information ------------------------------------------------------------------------------------

# Univariate variable importance
varimp_metr = calc_imp(df, metr, target_type=TARGET_TYPE)
print(varimp_metr)
varimp_metr_binned = calc_imp(df, metr + "_BINNED", target_type=TARGET_TYPE)
print(varimp_metr_binned)
varimp_metr_z = calc_imp(df, metr, "target_zscore", target_type=TARGET_TYPE)
print(varimp_metr_z)
varimp_metr_binned_z = calc_imp(df, metr + "_BINNED",  "target_zscore", target_type=TARGET_TYPE)
print(varimp_metr_binned_z)

# Plot
plot_distr(df, features = np.column_stack((metr, metr + "_BINNED")).ravel(),
           varimp=pd.concat([varimp_metr, varimp_metr_binned]), target_type=TARGET_TYPE, color=color, ylim=ylim,
           ncol=3, nrow=2, w=18, h=12, pdf=plotloc + TARGET_TYPE + "_distr_metr_final.pdf")
plot_distr(df, features = np.column_stack((metr, metr + "_BINNED")).ravel(), target = "target_zscore",
           varimp=pd.concat([varimp_metr_z, varimp_metr_binned_z]), target_type=TARGET_TYPE, color=color,
           ncol=3, nrow=2, w=18, h=12, pdf=plotloc + TARGET_TYPE + "_distr_metr_final_zscore.pdf")


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
df = Convert(features = cate, convert_to = "str").fit_transform(df)
df[cate].describe()


# --- Handling factor values ----------------------------------------------------------------------------------------

# Convert "standard" features: map missings to own level
df[cate] = df[cate].fillna("(Missing)")
df[cate].describe()

# Get "too many members" columns and copy these for additional encoded features (for tree based models)
topn_toomany = 30
levinfo = df[cate].nunique().sort_values(ascending = False)  # number of levels
print(levinfo)
toomany = levinfo[levinfo > topn_toomany].index.values
print(toomany)
toomany = setdiff(toomany, ["xxx", "xxx"])  # set exception for important variables

# Create encoded features (for tree based models), i.e. numeric representation
df = TargetEncoding(features=cate, encode_flag_column="encode_flag", target="target_zscore").fit_transform(df)
df["MISS_" + miss + "_ENCODED"] = df["MISS_" + miss].apply(lambda x: x.map({"no_miss": 0, "miss": 1}))

# Convert toomany features: lump levels and map missings to own level
df = MapToomany(features=toomany, n_top=topn_toomany).fit_transform(df)

# Univariate variable importance
varimp_cate = calc_imp(df, np.append(cate, ["MISS_" + miss]), target_type=TARGET_TYPE)
print(varimp_cate)
varimp_cate_z = calc_imp(df, np.append(cate, ["MISS_" + miss]), "target_zscore", target_type=TARGET_TYPE)
print(varimp_cate_z)

# Check
plot_distr(df, np.append(cate, ["MISS_" + miss]), target_type=TARGET_TYPE,
           varimp=varimp_cate,
           color=color, ylim=ylim,
           ncol=3, nrow=2, w=18, h=12, pdf=plotloc + TARGET_TYPE + "_distr_cate.pdf")
plot_distr(df, np.append(cate, ["MISS_" + miss]), target = "target_zscore", target_type=TARGET_TYPE,
           varimp=varimp_cate_z,
           color=color,
           ncol=3, nrow=2, w=18, h=12, pdf=plotloc + TARGET_TYPE + "_distr_cate_zscore.pdf")

# --- Removing variables ---------------------------------------------------------------------------------------------

# Remove highly/perfectly (>=99%) correlated (the ones with less levels!)
plot_corr(df, np.append(cate, ["MISS_" + miss]), cutoff=cutoff_corr, n_cluster=5,
          w=12, h= 12, pdf=plotloc + TARGET_TYPE + "_corr_cate.pdf")


# --- Predict target_iszero -------------------------------------------------------------------------------------------

# Univariate variable importance (again ONLY for non-missing observations!)
varimp_cate_iszero = calc_imp(df, np.append(cate, ["MISS_" + miss]), "target_iszero")

# Plot
plot_distr(df, np.append(cate, ["MISS_" + miss]), "target_iszero", varimp=varimp_cate_iszero, target_type="CLASS",
           ncol=3, nrow=2, w=18, h=12, pdf=plotloc + "distr_cate_target_iszero.pdf")


########################################################################################################################
# Prepare final data
########################################################################################################################

# --- Define final features ----------------------------------------------------------------------------------------

# Exclude some features
exclude = df_meta_sub.query("exclude == 1")["variable"].values
metr = setdiff(metr, exclude)
cate = setdiff(cate, exclude)
toomany = setdiff(toomany, exclude)
miss = setdiff(miss, exclude)

# Standard: for xgboost or Lasso
metr_standard = np.append(metr, toomany + "_ENCODED")
cate_standard = np.append(cate, "MISS_" + miss)

# Binned: for Lasso
metr_binned = np.array([])
cate_binned = np.append(setdiff(metr + "_BINNED", onebin), cate)

# Encoded: for Lightgbm or DeepLearning
metr_encoded = np.concatenate([metr, cate + "_ENCODED", "MISS_" + miss + "_ENCODED"])
cate_encoded = np.array([])

# Check
all_features = np.unique(np.concatenate([metr_standard, cate_standard, metr_binned, cate_binned, metr_encoded]))
setdiff(all_features, df.columns.values.tolist())
setdiff(df.columns.values.tolist(), all_features)


# --- Remove burned data ----------------------------------------------------------------------------------------

df = df.query("fold != 'util'").reset_index(drop = True)


# --- Save image ----------------------------------------------------------------------------------------------------

# Clean up
plt.close(fig = "all")  # plt.close(plt.gcf())

# Serialize
target_labels = "target"
with open(TARGET_TYPE + "_1_explore.pkl", "wb") as file:
    pickle.dump({"df": df,
                 "target_labels": target_labels,
                 "metr_standard": metr_standard,
                 "cate_standard": cate_standard,
                 "metr_binned": metr_binned,
                 "cate_binned": cate_binned,
                 "metr_encoded": metr_encoded,
                 "cate_encoded": cate_encoded},
                file)
