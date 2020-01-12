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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # , GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, SGDRegressor, LogisticRegression, ElasticNet
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.regularizers import l2
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

#  from sklearn.tree import DecisionTreeRegressor, plot_tree , export_graphviz

# Main parameter
TARGET_TYPE = "REGR"
target = "target_zscore"
plt.ion(); matplotlib.use('TkAgg')

# Specific parameters
n_jobs = 14
metric = "spear"

# Load results from exploration
df = metr_standard = cate_standard = metr_binned = cate_binned = metr_encoded = cate_encoded = target_labels = None
with open(TARGET_TYPE + "_1_explore.pkl", "rb") as file:
    d_pick = pickle.load(file)
for key, val in d_pick.items():
    exec(key + "= val")


# ######################################################################################################################
# # Test an algorithm (and determine parameter grid)
# ######################################################################################################################

# --- Sample data ----------------------------------------------------------------------------------------------------

df_tune = df.sample(n = min(df.shape[0], int(1e6))).reset_index(drop = True)

# Scale "metr_enocded" features for DL
df_tune[metr_encoded + "_normed"] = ((df_tune[metr_encoded] - df_tune[metr_encoded].min()) /
                                     (df_tune[metr_encoded].max() - df_tune[metr_encoded].min()))



# --- Define some splits -------------------------------------------------------------------------------------------

#split_index = PredefinedSplit(df_tune["fold"].map({"train": -1, "test": 0}).values)
split_my1fold_cv = TrainTestSep(1)
#split_5fold = KFold(5, shuffle=False, random_state=42)
split_my5fold_cv = TrainTestSep(5)
split_my5fold_boot = TrainTestSep(5, "bootstrap")
'''
df_tune["fold"].value_counts()
mysplit = split_my1fold_cv.split(df_tune)
i_train, i_test = next(mysplit)
df_tune["fold"].iloc[i_train].describe()
df_tune["fold"].iloc[i_test].describe()
i_test.sort()
i_test
'''


# --- Fits -----------------------------------------------------------------------------------------------------------

# Lasso / Elastic Net
#fit = (GridSearchCV(SGDRegressor(penalty = "ElasticNet", warm_start = True),  # , tol=1e-2
fit = (GridSearchCV(ElasticNet(normalize=False, warm_start=True),  # , tol=1e-2
                    {"alpha": [2 ** x for x in range(-8, -24, -2)],
                     "l1_ratio": [1]},
                    cv = split_my1fold_cv.split(df_tune),
                    refit = False,
                    scoring = d_scoring[TARGET_TYPE],
                    return_train_score = True,
                    n_jobs = n_jobs)
       .fit(CreateSparseMatrix(metr = metr_binned, cate = cate_binned, df_ref = df_tune).fit_transform(df_tune),
            df_tune[target]))
plot_cvresult(fit.cv_results_, metric = metric, x_var = "alpha", color_var = "l1_ratio")
pd.DataFrame(fit.cv_results_)
# -> keep l1_ratio=1 to have a full Lasso


# XGBoost
start = time.time()
fit = (GridSearchCV_xlgb(xgb.XGBRegressor(verbosity = 0, n_jobs = n_jobs),
                         {"n_estimators": [x for x in range(100, 3100, 500)], "learning_rate": [0.02],
                          "max_depth": [12,15,17], "min_child_weight": [10],
                          "colsample_bytree": [0.7], "subsample": [0.7]},
                         cv = split_my1fold_cv.split(df_tune),
                         refit = False,
                         scoring = d_scoring[TARGET_TYPE],
                         return_train_score = True,
                         n_jobs = n_jobs)
       .fit(CreateSparseMatrix(metr = metr_standard, cate = cate_standard, df_ref = df_tune).fit_transform(df_tune),
            df_tune[target]))
print(time.time()-start)
pd.DataFrame(fit.cv_results_)
plot_cvresult(fit.cv_results_, metric = metric,
              x_var = "n_estimators", color_var = "max_depth", column_var = "min_child_weight")


# -> keep around the recommended values: max_depth = 6, shrinkage = 0.01, n.minobsinnode = 10


# LightGBM
# metr_encoded = setdiff(metr_encoded,'building_id_ENCODED')
start = time.time()
fit = (GridSearchCV_xlgb(lgbm.LGBMRegressor(n_jobs = n_jobs),
                         {"n_estimators": [x for x in range(100, 3100, 500)], "learning_rate": [0.02],
                          "num_leaves": [64,128,512], "min_child_samples": [10],
                          "colsample_bytree": [0.7], "subsample": [0.7]},
                         cv = split_my1fold_cv.split(df_tune),
                         refit = False,
                         scoring = d_scoring[TARGET_TYPE],
                         return_train_score = True,
                         n_jobs = n_jobs)
       .fit(df_tune[metr_encoded], df_tune[target],
            categorical_feature = [x for x in metr_encoded.tolist() if "_ENCODED" in x]))
print(time.time()-start)
plot_cvresult(fit.cv_results_, metric = metric,
              x_var = "n_estimators", color_var = "num_leaves",
              column_var = "colsample_bytree", row_var = "subsample",
              style_var = "min_child_samples")


# DeepL

# Keras wrapper for Scikit
def keras_model(input_dim, output_dim, target_type,
                size = "10",
                lambdah = None, dropout = None,
                lr = 1e-5,
                batch_normalization = False,
                activation = "relu"):
    model = Sequential()

    # Add dense layers
    for units in size.split("-"):
        model.add(Dense(units = int(units), activation = activation, input_dim = input_dim,
                        kernel_regularizer = l2(lambdah) if lambdah is not None else None,
                        kernel_initializer = "glorot_uniform"))
        # Add additional layer
        if batch_normalization is not None:
            model.add(BatchNormalization())
        if dropout is not None:
            model.add(Dropout(dropout))

    # Output
    if target_type == "CLASS":
        model.add(Dense(1, activation = 'sigmoid',
                        kernel_regularizer = l2(lambdah) if lambdah is not None else None))
        model.compile(loss = "binary_crossentropy", optimizer = optimizers.RMSprop(lr = lr), metrics = ["accuracy"])
    elif target_type == "MULTICLASS":
        model.add(Dense(output_dim, activation = 'softmax',
                        kernel_regularizer = l2(lambdah) if lambdah is not None else None))
        model.compile(loss = "categorical_crossentropy", optimizer = optimizers.RMSprop(lr = lr),
                      metrics = ["accuracy"])
    else:
        model.add(Dense(1, activation = 'linear',
                        kernel_regularizer = l2(lambdah) if lambdah is not None else None))
        model.compile(loss = "mean_squared_error", optimizer = optimizers.RMSprop(lr = lr),
                      metrics = ["mean_squared_error"])

    return model


# Fit
fit = (GridSearchCV(KerasRegressor(build_fn = keras_model,
                                   input_dim = metr_encoded.size,
                                   output_dim = 1,
                                   target_type = TARGET_TYPE,
                                   verbose = 0),
                    {"size": ["50","100-50-20"],
                     "lambdah": [None], "dropout": [None],
                     "batch_size": [100], "lr": [1e-3],
                     "batch_normalization": [True],
                     "activation": ["relu"],
                     "epochs": [20]},
                    cv = split_my1fold_cv.split(df_tune),
                    refit = False,
                    scoring = d_scoring[TARGET_TYPE],
                    return_train_score = False,
                    n_jobs = n_jobs)
       .fit(CreateSparseMatrix(metr = metr_encoded + "_normed", df_ref = df_tune).fit_transform(df_tune),
            df_tune[target]))

plot_cvresult(fit.cv_results_, metric = metric, x_var = "epochs", color_var = "lr",
              column_var = "size", row_var = "batch_size")


# ######################################################################################################################
# Evaluate generalization gap
# ######################################################################################################################

# Sample data (usually undersample training data)
df_gengap = df_tune.copy()

# Tune grid to loop over
param_grid = {"n_estimators": [x for x in range(100, 3100, 500)], "learning_rate": [0.01],
              "max_depth": [3, 6, 9], "min_child_weight": [10],
              "colsample_bytree": [0.7], "subsample": [0.7],
              "gamma": [10]}

# Calc generalization gap
fit = (GridSearchCV_xlgb(xgb.XGBRegressor(verbosity = 0),
                         param_grid,
                         cv = split_my1fold_cv.split(df_gengap),
                         refit = False,
                         scoring = d_scoring[TARGET_TYPE],
                         return_train_score = True,
                         n_jobs = n_jobs)
       .fit(CreateSparseMatrix(metr = metr_standard, cate = cate_standard, df_ref = df_gengap).fit_transform(df_gengap),
            df_gengap["target"]))
plot_gengap(fit.cv_results_, metric = metric,
            x_var = "n_estimators", color_var = "max_depth", column_var = "min_child_weight", row_var = "gamma",
            pdf = plotloc + TARGET_TYPE + "_xgboost_gengap.pdf")


# ######################################################################################################################
# Simulation: compare algorithms
# ######################################################################################################################

# Basic data sampling
df_modelcomp = df_tune.copy()


# --- Run methods ------------------------------------------------------------------------------------------------------

df_modelcomp_result = pd.DataFrame()  # intialize

# Lightgbm
cvresults = cross_validate(
    estimator = GridSearchCV_xlgb(
        lgbm.LGBMRegressor(),
        {"n_estimators": [x for x in range(100, 3100, 500)], "learning_rate": [0.01],
         "num_leaves": [64], "min_child_weight": [10],
         "colsample_bytree": [0.7], "subsample": [0.7]},
        cv = ShuffleSplit(1, 0.2, random_state = 999),  # just 1-fold for tuning
        refit = metric,
        scoring = d_scoring[TARGET_TYPE],
        return_train_score = False,
        n_jobs = n_jobs),
    X = df_tune[metr_encoded],
    y = df_tune[target],
    fit_params = {"categorical_feature": [x for x in metr_encoded.tolist() if "_ENCODED" in x]},
    cv = split_my5fold_cv.split(df_modelcomp),
    return_train_score = False,
    n_jobs = n_jobs)
df_modelcomp_result = df_modelcomp_result.append(pd.DataFrame.from_dict(cvresults).reset_index()
                                                 .assign(model = "Lightgbm"),
                                                 ignore_index = True)

# Xgboost
cvresults = cross_validate(
    estimator = GridSearchCV_xlgb(
        xgb.XGBRegressor(verbosity = 0),
        {"n_estimators": [x for x in range(100, 3100, 500)], "learning_rate": [0.01],
         "max_depth": [6], "min_child_weight": [10],
         "colsample_bytree": [0.7], "subsample": [0.7]},
        cv = ShuffleSplit(1, 0.2, random_state = 999),  # just 1-fold for tuning
        refit = metric,
        scoring = d_scoring[TARGET_TYPE],
        return_train_score = False,
        n_jobs = n_jobs),
    X = CreateSparseMatrix(metr = metr_standard, cate = cate_standard, df_ref = df_modelcomp).fit_transform(
        df_modelcomp),
    y = df_modelcomp[target],
    cv = split_my5fold_cv.split(df_modelcomp),
    return_train_score = False,
    n_jobs = n_jobs)
df_modelcomp_result = df_modelcomp_result.append(pd.DataFrame.from_dict(cvresults).reset_index()
                                                 .assign(model = "XGBoost"),
                                                 ignore_index = True)


# --- Plot model comparison ------------------------------------------------------------------------------

plot_modelcomp(df_modelcomp_result.rename(columns = {"index": "run", "test_score": metric}), scorevar = metric,
               pdf = plotloc + TARGET_TYPE + "_model_comparison.pdf")


# ######################################################################################################################
# Learning curve for winner algorithm
# ######################################################################################################################

# Basic data sampling
df_lc = df_tune.copy()

# Calc learning curve
n_train, score_train, score_test = learning_curve(
    estimator = GridSearchCV_xlgb(
        xgb.XGBRegressor(verbosity = 0) if TARGET_TYPE == "REGR" else xgb.XGBClassifier(verbosity = 0),
        {"n_estimators": [x for x in range(100, 3100, 500)], "learning_rate": [0.01],
         "max_depth": [6], "min_child_weight": [10],
         "colsample_bytree": [0.7], "subsample": [1]},
        cv = ShuffleSplit(1, 0.2, random_state = 999),  # just 1-fold for tuning
        refit = metric,
        scoring = d_scoring[TARGET_TYPE],
        return_train_score = False,
        n_jobs = 4),
    X = CreateSparseMatrix(metr = metr_standard, cate = cate_standard, df_ref = df_lc).fit_transform(df_lc),
    y = df_lc[target],
    train_sizes = np.append(np.linspace(0.05, 0.1, 5), np.linspace(0.2, 1, 5)),
    cv = split_my1fold_cv.split(df_lc),
    n_jobs = 4)

# Plot it
plot_learning_curve(n_train, score_train, score_test,
                    pdf = plotloc + TARGET_TYPE + "_learningCurve.pdf")

plt.close("all")
