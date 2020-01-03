
# ######################################################################################################################
#  Initialize: Libraries, functions, parameters
# ######################################################################################################################

# General libraries, parameters and functions
from os import getcwd, chdir
chdir("../Ashrae")
import sys; sys.path.append(getcwd() + "\\code") #not needed if code is marked as "source" in pycharm
from initialize import *

# Specific libraries
from sklearn.model_selection import cross_validate
from sklearn.base import clone
import xgboost as xgb
import lightgbm as lgbm

# Main parameter
TARGET_TYPE = "REGR"

# Specific parameters
scoring = {"spear": make_scorer(spearman_loss_func, greater_is_better=True),
           "rmse": make_scorer(rmse, greater_is_better=False)}
metric = "spear"
importance_cut = 99
topn = 10
ylim_res = (-1.5, 1.5)

# Load results from exploration
with open(TARGET_TYPE + "_1_explore.pkl", "rb") as file:
    d_vars = pickle.load(file)
df, features_lasso, features_xgb, features_lgbm = \
    d_vars["df"], d_vars["features_lasso"], d_vars["features_xgb"], d_vars["features_lgbm"]
metr = features_xgb["metr"]
cate = features_xgb["cate"]
features = np.append(metr, cate)


# ######################################################################################################################
# Prepare
# ######################################################################################################################

# Tuning parameter to use (for xgb)
n_estimators = 2100
learning_rate = 0.01
max_depth = 6
num_leaves = 32
min_child_weight = 10
min_child_samples = 10
colsample_bytree = 1
subsample = 1
gamma = 0


# --- Sample data ----------------------------------------------------------------------------------------------------
df_train = df.query("fold == 'train'").sample(n=min(df.query("fold == 'train'").shape[0], int(1e5)))
b_sample = None
b_all = None

# Test data
df_test = df.query("fold == 'test'")  # .sample(300) #ATTENTION: Do not sample in final run!!!

# Combine again
df_traintest = pd.concat([df_train, df_test])

# Folds for crossvalidation and check
split_my5fold = TrainTestSep(5, "cv")
for i_train, i_test in split_my5fold.split(df_traintest):
    print("TRAIN-fold:", df_traintest["fold"].iloc[i_train].value_counts())
    print("TEST-fold:", df_traintest["fold"].iloc[i_test].value_counts())
    print("##########")


# ######################################################################################################################
# Performance
# ######################################################################################################################

# --- Do the full fit and predict on test data -------------------------------------------------------------------

# Fit
clf = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                       max_depth=max_depth, min_child_weight=min_child_weight,
                       colsample_bytree=colsample_bytree, subsample=subsample,
                       gamma=0,
                       n_jobs=12)
# clf = lgbm.LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
#                          num_leaves=num_leaves, min_child_samples=min_child_samples)

X_train = CreateSparseMatrix(metr=metr,
                             cate=cate,
                             df_ref=df_traintest).fit_transform(df_train)
fit = clf.fit(X_train, df_train["target"].values)

# Predict
X_test = CreateSparseMatrix(metr=metr,
                            cate=cate,
                            df_ref=df_traintest).fit_transform(df_test)
yhat_test = fit.predict(X_test)
pd.DataFrame(yhat_test).describe()
print(spearman_loss_func(df_test["target"].values, yhat_test))
print(rmse(df_test["target"].values, yhat_test))

# Plot performance
plot_all_performances(df_test["target"], yhat_test, target_type=TARGET_TYPE, ylim=None,
                      pdf=plotloc + TARGET_TYPE + "_performance.pdf")


# --- Check performance for crossvalidated fits ---------------------------------------------------------------------
d_cv = cross_validate(clf,
                      CreateSparseMatrix(metr=metr,
                                         cate=cate,
                                         df_ref=df_traintest).fit_transform(df_traintest),
                      df_traintest["target"].values,
                      cv=split_my5fold.split(df_traintest),  # special 5fold
                      scoring=scoring,
                      return_estimator=True,
                      n_jobs=1)
# Performance
print(d_cv["test_spear"])
print(d_cv["test_rmse"])


# --- Most important variables (importance_cum < 95) model fit ------------------------------------------------------
# Variable importance (on train data!)
df_varimp_train = calc_varimp_by_permutation(df_train, df_traintest, fit, "target", metr, cate, target_type=TARGET_TYPE,
                                             b_sample=b_sample, b_all=b_all, n_jobs=12)

# Top features (importances sum up to 95% of whole sum)
features_top = df_varimp_train.loc[df_varimp_train["importance_cum"] < importance_cut, "feature"].values

# Fit again only on features_top
X_train_top = CreateSparseMatrix(metr[np.in1d(metr, features_top)], cate[np.in1d(cate, features_top)],
                                 df_ref=df_traintest).fit_transform(df_train)
fit_top = clone(clf).fit(X_train_top, df_train["target"])

# Plot performance
X_test_top = CreateSparseMatrix(metr[np.in1d(metr, features_top)], cate[np.in1d(cate, features_top)],
                                df_ref=df_traintest).fit_transform(df_test)
yhat_top = fit_top.predict(X_test_top)
print(spearman_loss_func(df_test["target"].values, yhat_top))

plot_all_performances(df_test["target"], yhat_top, target_type=TARGET_TYPE,
                      pdf=plotloc + TARGET_TYPE + "_performance_top.pdf")


# ######################################################################################################################
# Diagnosis
# ######################################################################################################################

# ---- Check residuals --------------------------------------------------------------------------------------------

# Residuals
df_test["yhat"] = yhat_test

df_test["residual"] = df_test["target"] - df_test["yhat"]
df_test["abs_residual"] = df_test["residual"].abs()
df_test["residual"].describe()


# For non-regr tasks one might want to plot it for each target level (df_test.query("target == 0/1"))
plot_distr(df_test, features, target="residual", target_type="REGR", ylim=ylim_res,
           ncol=3, nrow=2, w=18, h=12, pdf=plotloc + TARGET_TYPE + "_diagnosis_residual.pdf")
plt.close(fig="all")


# Absolute residuals
if TARGET_TYPE in ["CLASS", "REGR"]:
    plot_distr(df_test, features, target="abs_residual", target_type="REGR", ylim=(0,ylim_res[1]),
               ncol=3, nrow=2, w=18, h=12, pdf=plotloc + TARGET_TYPE + "_diagnosis_absolute_residual.pdf")


# ---- Explain bad predictions ------------------------------------------------------------------------------------

# TODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO


# ######################################################################################################################
# Variable Importance
# ######################################################################################################################

# --- Default Variable Importance: uses gain sum of all trees ----------------------------------------------------------
xgb.plot_importance(fit)


# --- Variable Importance by permuation argument -------------------------------------------------------------------
# Importance for "total" fit (on test data!)
df_varimp = calc_varimp_by_permutation(df_test, df_traintest, fit, "target", metr, cate, target_type=TARGET_TYPE,
                                       b_sample=b_sample, b_all=b_all)
topn_features = df_varimp["feature"].values[range(topn)]

# Add other information (e.g. special category): category variable is needed -> fill with at least with "dummy"
df_varimp["Category"] = pd.cut(df_varimp["importance"], [-np.inf, 10, 50, np.inf], labels=["low", "medium", "high"])

# Crossvalidate Importance: ONLY for topn_vars
# df_varimp_cv = pd.DataFrame()
# for i, (i_train, i_test) in enumerate(split_my5fold.split(df_traintest)):
#     df_tmp = calc_varimp_by_permutation(df_traintest.iloc[i_train, :], df_traintest, d_cv["estimator"][i],
#                                         "target", metr, cate, TARGET_TYPE,
#                                         b_sample, b_all,
#                                         features=topn_features)
#     df_tmp["run"] = i
#     df_varimp_cv = df_varimp_cv.append(df_tmp)


# Plot
fig, ax = plt.subplots(1, 1)
sns.barplot("importance", "feature", hue="Category", data=df_varimp.iloc[range(topn)],
            dodge=False, palette=sns.xkcd_palette(["blue", "orange", "red"]), ax=ax)
ax.plot("importance_cum", "feature", data=df_varimp.iloc[range(topn)], color="grey", marker="o")
ax.set_xlabel(r"importance / cumulative importance in % (-$\bullet$-)")
# noinspection PyTypeChecker
ax.set_title("Top{0: .0f} (of{1: .0f}) Feature Importances".format(topn, len(features)))
fig.tight_layout()
fig.savefig(plotloc + TARGET_TYPE + "_variable_importance.pdf")


# --- Compare variable importance for train and test (hints to variables prone to overfitting) -------------------------
fig, ax = plt.subplots(1, 1)
sns.barplot("importance_sumnormed", "feature", hue="fold",
            data=pd.concat([df_varimp_train.assign(fold="train"), df_varimp.assign(fold="test")], sort=False))


# ######################################################################################################################
# Partial Dependance
# ######################################################################################################################

df_pd = calc_partial_dependence(df_test, df_traintest, fit, "target", metr, cate, target_type=TARGET_TYPE,
                                b_sample=b_sample, b_all=b_all,
                                features=topn_features,
                                n_jobs=12)

# # Crossvalidate Depedence
# df_pd_cv = pd.DataFrame()
# for i, (i_train, i_test) in enumerate(split_my5fold.split(df_traintest)):
#     df_tmp = calc_partial_dependence(df_traintest.iloc[i_train, :], df_traintest, d_cv["estimator"][i],
#                                      "target", metr, cate, TARGET_TYPE,
#                                      b_sample, b_all,
#                                      features=topn_features)
#     df_tmp["run"] = i
#     df_pd_cv = df_pd_cv.append(df_tmp)


# ######################################################################################################################
# xgboost Explainer
# ######################################################################################################################

# TODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO

plt.close("all")
