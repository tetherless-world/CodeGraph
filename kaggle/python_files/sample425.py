#Thanks for Vettejeep's sharing(https://www.kaggle.com/vettejeep/masters-final-project-model-lb-1-392)
#I'm a beginner in data science and I just find these parameter turning got a better cv and lb score.
import os
import time
import warnings
import traceback
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
import scipy.signal as sg
import multiprocessing as mp
from scipy.signal import hann
from scipy.signal import hilbert
from scipy.signal import convolve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from tqdm import tqdm
warnings.filterwarnings("ignore")

DATA_DIR = '../input/'  # set for local environment

SIG_LEN = 150000
NUM_SEG_PER_PROC = 4000
NUM_THREADS = 6

NY_FREQ_IDX = 75000  # the test signals are 150k samples long, Nyquist is thus 75k.
CUTOFF = 18000
MAX_FREQ_IDX = 20000
FREQ_STEP = 2500


def join_mp_build():
    train_X = pd.read_csv('../input/masters-final-project/train_x_%d.csv' % 0)
    train_y = pd.read_csv('../input/masters-final-project/train_y_%d.csv' % 0)
    test_X = pd.read_csv('../input/masters-final-project/test_x.csv')
    for i in tqdm(range(1, NUM_THREADS)):
        temp = pd.read_csv('../input/masters-final-project/train_x_%d.csv' % i)
        train_X = train_X.append(temp)
        temp = pd.read_csv('../input/masters-final-project/train_y_%d.csv' % i)
        train_y = train_y.append(temp)
    return train_X, train_y, test_X
train_X, train_y, test_X = join_mp_build()
train_X['classic_sta_lta1_mean_0'].loc[~np.isfinite(train_X['classic_sta_lta1_mean_0'])] = \
train_X['classic_sta_lta1_mean_0'].loc[np.isfinite(train_X['classic_sta_lta1_mean_0'])].mean()

def scale_fields(train_X = train_X, test_X = test_X):
    try:
        train_X.drop(labels=['seg_id', 'seg_start', 'seg_end'], axis=1, inplace=True)
    except:
        pass
    print('start scaler')
    scaler = StandardScaler()
    scaler.fit(train_X)
    scaled_train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)
    scaled_test_X = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns)
    return scaled_train_X, scaled_test_X
scaled_train_X, scaled_test_X = scale_fields()

def xgb_trimmed_model(scaled_train_X = scaled_train_X, scaled_test_X = scaled_test_X, train_y = train_y):
    maes = []
    rmses = []
    tr_maes = []
    tr_rmses = []
    submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv', index_col='seg_id')

    pcol = []
    pcor = []
    pval = []
    y = train_y['time_to_failure'].values
    for col in scaled_train_X.columns:
        pcol.append(col)
        pcor.append(abs(pearsonr(scaled_train_X[col], y)[0]))
        pval.append(abs(pearsonr(scaled_train_X[col], y)[1]))
    df = pd.DataFrame(data={'col': pcol, 'cor': pcor, 'pval': pval}, index=range(len(pcol)))
    df.sort_values(by=['cor', 'pval'], inplace=True, ascending = False)
    df.dropna(inplace = True)
    df = df.iloc[: 500]

    drop_cols = []
    for col in scaled_train_X.columns:
        if col not in df['col'].tolist():
            drop_cols.append(col)

    scaled_train_X.drop(labels=drop_cols, axis=1, inplace=True)
    scaled_test_X.drop(labels=drop_cols, axis=1, inplace=True)

    predictions = np.zeros(len(scaled_test_X))
    preds_train = np.zeros(len(scaled_train_X))

    print('shapes of train and test:', scaled_train_X.shape, scaled_test_X.shape)

    n_fold = 6
    folds = KFold(n_splits=n_fold, shuffle=False, random_state=42)
    
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_train_X, train_y.values)):
        print('working fold %d' % fold_)
        strLog = "fold {}".format(fold_)
        print(strLog)
        X_tr, X_val = scaled_train_X.iloc[trn_idx], scaled_train_X.iloc[val_idx]
        y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]

        model = xgb.XGBRegressor(n_estimators=1000,
                                        learning_rate=0.005,
                                        max_depth=12,
                                        subsample=0.9,
                                        colsample_bytree=0.3,
                                        reg_lambda=1.0, # seems best within 0.5 of 2.0
                                        # gamma=1,
                                        random_state=777+fold_,
                                        n_jobs=-1,
                                        verbosity=2)
        model.fit(X_tr, y_tr)

        # predictions
        preds = model.predict(scaled_test_X)  #, num_iteration=model.best_iteration_)
        predictions += preds / folds.n_splits
        preds = model.predict(scaled_train_X)  #, num_iteration=model.best_iteration_)
        preds_train += preds / folds.n_splits
        preds = model.predict(X_val)  #, num_iteration=model.best_iteration_)

        # mean absolute error
        mae = mean_absolute_error(y_val, preds)
        print('MAE: %.6f' % mae)
        maes.append(mae)
        # root mean squared error
        rmse = mean_squared_error(y_val, preds)
        print('RMSE: %.6f' % rmse)
        rmses.append(rmse)
        # training for over fit
        preds = model.predict(X_tr)  #, num_iteration=model.best_iteration_)
        mae = mean_absolute_error(y_tr, preds)
        print('Tr MAE: %.6f' % mae)
        tr_maes.append(mae)
        rmse = mean_squared_error(y_tr, preds)
        print('Tr RMSE: %.6f' % rmse)
        tr_rmses.append(rmse)
    print('MAEs', maes)
    print('MAE mean: %.6f' % np.mean(maes))
    print('RMSEs', rmses)
    print('RMSE mean: %.6f' % np.mean(rmses))
    print('Tr MAEs', tr_maes)
    print('Tr MAE mean: %.6f' % np.mean(tr_maes))
    print('Tr RMSEs', rmses)
    print('Tr RMSE mean: %.6f' % np.mean(tr_rmses))
    submission.time_to_failure = predictions
    submission.to_csv('submission_xgb_pearson_6fold.csv')  # index needed, it is seg id
    pr_tr = pd.DataFrame(data=preds_train, columns=['time_to_failure'], index=range(0, preds_train.shape[0]))
    pr_tr.to_csv(r'preds_tr_xgb_pearson_6fold.csv', index=False)
    print('Train shape: {}, Test shape: {}, Y shape: {}'.format(scaled_train_X.shape, scaled_test_X.shape, train_y.shape))
xgb_trimmed_model()