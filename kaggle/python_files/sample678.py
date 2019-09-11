#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from joblib import Parallel, delayed
import os
import gc
import xgboost as xgb
from sklearn.model_selection import KFold
import scipy as sp
from sklearn import metrics
from tsfresh.feature_extraction import feature_calculators
print(os.listdir("../input"))

# In[ ]:


class FeatureGenerator(object):
    def __init__(self, dtype, n_jobs=1, chunk_size=None):
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.filename = None
        self.n_jobs = n_jobs
        self.test_files = []
        if self.dtype == 'train':
            self.filename = '../input/train.csv'
            self.total_data = int(629145481 / self.chunk_size)
        else:
            submission = pd.read_csv('../input/sample_submission.csv')
            for seg_id in submission.seg_id.values:
                self.test_files.append((seg_id, '../input/test/' + seg_id + '.csv'))
            self.total_data = int(len(submission))

    def read_chunks(self):
        if self.dtype == 'train':
            iter_df = pd.read_csv(self.filename, iterator=True, chunksize=self.chunk_size,
                                  dtype={'acoustic_data': np.float64, 'time_to_failure': np.float64})
            for counter, df in enumerate(iter_df):
                x = df.acoustic_data.values
                y = df.time_to_failure.values[-1]
                seg_id = 'train_' + str(counter)
                del df
                yield seg_id, x, y
        else:
            for seg_id, f in self.test_files:
                df = pd.read_csv(f, dtype={'acoustic_data': np.float64})
                x = df.acoustic_data.values[-self.chunk_size:]
                del df
                yield seg_id, x, -999

    def features(self, x, y, seg_id):
        feature_dict = dict()
        feature_dict['target'] = y
        feature_dict['seg_id'] = seg_id

        # create features here
        # numpy
        feature_dict['mean'] = np.mean(x)
        feature_dict['max'] = np.max(x)
        feature_dict['min'] = np.min(x)
        feature_dict['std'] = np.std(x)
        feature_dict['var'] = np.var(x)
        feature_dict['ptp'] = np.ptp(x)
        feature_dict['percentile_10'] = np.percentile(x, 10)
        feature_dict['percentile_20'] = np.percentile(x, 20)
        feature_dict['percentile_30'] = np.percentile(x, 30)
        feature_dict['percentile_40'] = np.percentile(x, 40)
        feature_dict['percentile_50'] = np.percentile(x, 50)
        feature_dict['percentile_60'] = np.percentile(x, 60)
        feature_dict['percentile_70'] = np.percentile(x, 70)
        feature_dict['percentile_80'] = np.percentile(x, 80)
        feature_dict['percentile_90'] = np.percentile(x, 90)

        # scipy
        feature_dict['skew'] = sp.stats.skew(x)
        feature_dict['kurtosis'] = sp.stats.kurtosis(x)
        feature_dict['kstat_1'] = sp.stats.kstat(x, 1)
        feature_dict['kstat_2'] = sp.stats.kstat(x, 2)
        feature_dict['kstat_3'] = sp.stats.kstat(x, 3)
        feature_dict['kstat_4'] = sp.stats.kstat(x, 4)
        feature_dict['moment_1'] = sp.stats.moment(x, 1)
        feature_dict['moment_2'] = sp.stats.moment(x, 2)
        feature_dict['moment_3'] = sp.stats.moment(x, 3)
        feature_dict['moment_4'] = sp.stats.moment(x, 4)
        
        feature_dict['abs_energy'] = feature_calculators.abs_energy(x)
        feature_dict['abs_sum_of_changes'] = feature_calculators.absolute_sum_of_changes(x)
        feature_dict['count_above_mean'] = feature_calculators.count_above_mean(x)
        feature_dict['count_below_mean'] = feature_calculators.count_below_mean(x)
        feature_dict['mean_abs_change'] = feature_calculators.mean_abs_change(x)
        feature_dict['mean_change'] = feature_calculators.mean_change(x)
        feature_dict['var_larger_than_std_dev'] = feature_calculators.variance_larger_than_standard_deviation(x)
        feature_dict['range_minf_m4000'] = feature_calculators.range_count(x, -np.inf, -4000)
        feature_dict['range_m4000_m3000'] = feature_calculators.range_count(x, -4000, -3000)
        feature_dict['range_m3000_m2000'] = feature_calculators.range_count(x, -3000, -2000)
        feature_dict['range_m2000_m1000'] = feature_calculators.range_count(x, -2000, -1000)
        feature_dict['range_m1000_0'] = feature_calculators.range_count(x, -1000, 0)
        feature_dict['range_0_p1000'] = feature_calculators.range_count(x, 0, 1000)
        feature_dict['range_p1000_p2000'] = feature_calculators.range_count(x, 1000, 2000)
        feature_dict['range_p2000_p3000'] = feature_calculators.range_count(x, 2000, 3000)
        feature_dict['range_p3000_p4000'] = feature_calculators.range_count(x, 3000, 4000)
        feature_dict['range_p4000_pinf'] = feature_calculators.range_count(x, 4000, np.inf)

        feature_dict['ratio_unique_values'] = feature_calculators.ratio_value_number_to_time_series_length(x)
        feature_dict['first_loc_min'] = feature_calculators.first_location_of_minimum(x)
        feature_dict['first_loc_max'] = feature_calculators.first_location_of_maximum(x)
        feature_dict['last_loc_min'] = feature_calculators.last_location_of_minimum(x)
        feature_dict['last_loc_max'] = feature_calculators.last_location_of_maximum(x)
        feature_dict['time_rev_asym_stat_10'] = feature_calculators.time_reversal_asymmetry_statistic(x, 10)
        feature_dict['time_rev_asym_stat_100'] = feature_calculators.time_reversal_asymmetry_statistic(x, 100)
        feature_dict['time_rev_asym_stat_1000'] = feature_calculators.time_reversal_asymmetry_statistic(x, 1000)
        feature_dict['autocorrelation_5'] = feature_calculators.autocorrelation(x, 5)
        feature_dict['autocorrelation_10'] = feature_calculators.autocorrelation(x, 10)
        feature_dict['autocorrelation_50'] = feature_calculators.autocorrelation(x, 50)
        feature_dict['autocorrelation_100'] = feature_calculators.autocorrelation(x, 100)
        feature_dict['autocorrelation_1000'] = feature_calculators.autocorrelation(x, 1000)
        feature_dict['c3_5'] = feature_calculators.c3(x, 5)
        feature_dict['c3_10'] = feature_calculators.c3(x, 10)
        feature_dict['c3_100'] = feature_calculators.c3(x, 100)
        feature_dict['fft_1_real'] = list(feature_calculators.fft_coefficient(x, [{'coeff': 1, 'attr': 'real'}]))[0][1]
        feature_dict['fft_1_imag'] = list(feature_calculators.fft_coefficient(x, [{'coeff': 1, 'attr': 'imag'}]))[0][1]
        feature_dict['fft_1_ang'] = list(feature_calculators.fft_coefficient(x, [{'coeff': 1, 'attr': 'angle'}]))[0][1]
        feature_dict['fft_2_real'] = list(feature_calculators.fft_coefficient(x, [{'coeff': 2, 'attr': 'real'}]))[0][1]
        feature_dict['fft_2_imag'] = list(feature_calculators.fft_coefficient(x, [{'coeff': 2, 'attr': 'imag'}]))[0][1]
        feature_dict['fft_2_ang'] = list(feature_calculators.fft_coefficient(x, [{'coeff': 2, 'attr': 'angle'}]))[0][1]
        feature_dict['fft_3_real'] = list(feature_calculators.fft_coefficient(x, [{'coeff': 3, 'attr': 'real'}]))[0][1]
        feature_dict['fft_3_imag'] = list(feature_calculators.fft_coefficient(x, [{'coeff': 3, 'attr': 'imag'}]))[0][1]
        feature_dict['fft_3_ang'] = list(feature_calculators.fft_coefficient(x, [{'coeff': 3, 'attr': 'angle'}]))[0][1]
        feature_dict['long_strk_above_mean'] = feature_calculators.longest_strike_above_mean(x)
        feature_dict['long_strk_below_mean'] = feature_calculators.longest_strike_below_mean(x)
        feature_dict['cid_ce_0'] = feature_calculators.cid_ce(x, 0)
        feature_dict['cid_ce_1'] = feature_calculators.cid_ce(x, 1)
        feature_dict['binned_entropy_5'] = feature_calculators.binned_entropy(x, 5)
        feature_dict['binned_entropy_10'] = feature_calculators.binned_entropy(x, 10)
        feature_dict['binned_entropy_20'] = feature_calculators.binned_entropy(x, 20)
        feature_dict['binned_entropy_50'] = feature_calculators.binned_entropy(x, 50)
        feature_dict['binned_entropy_80'] = feature_calculators.binned_entropy(x, 80)
        feature_dict['binned_entropy_100'] = feature_calculators.binned_entropy(x, 100)

        feature_dict['num_crossing_0'] = feature_calculators.number_crossing_m(x, 0)
        feature_dict['num_peaks_10'] = feature_calculators.number_peaks(x, 10)
        feature_dict['num_peaks_50'] = feature_calculators.number_peaks(x, 50)
        feature_dict['num_peaks_100'] = feature_calculators.number_peaks(x, 100)
        feature_dict['num_peaks_500'] = feature_calculators.number_peaks(x, 500)

        feature_dict['spkt_welch_density_1'] = list(feature_calculators.spkt_welch_density(x, [{'coeff': 1}]))[0][1]
        feature_dict['spkt_welch_density_10'] = list(feature_calculators.spkt_welch_density(x, [{'coeff': 10}]))[0][1]
        feature_dict['spkt_welch_density_50'] = list(feature_calculators.spkt_welch_density(x, [{'coeff': 50}]))[0][1]
        feature_dict['spkt_welch_density_100'] = list(feature_calculators.spkt_welch_density(x, [{'coeff': 100}]))[0][1]

        feature_dict['time_rev_asym_stat_1'] = feature_calculators.time_reversal_asymmetry_statistic(x, 1)
        feature_dict['time_rev_asym_stat_10'] = feature_calculators.time_reversal_asymmetry_statistic(x, 10)
        feature_dict['time_rev_asym_stat_100'] = feature_calculators.time_reversal_asymmetry_statistic(x, 100)        

        return feature_dict

    def generate(self):
        feature_list = []
        res = Parallel(n_jobs=self.n_jobs,
                       backend='threading')(delayed(self.features)(x, y, s)
                                            for s, x, y in tqdm(self.read_chunks(), total=self.total_data))
        for r in res:
            feature_list.append(r)
        return pd.DataFrame(feature_list)


training_fg = FeatureGenerator(dtype='train', n_jobs=10, chunk_size=150000)
training_data = training_fg.generate()

test_fg = FeatureGenerator(dtype='test', n_jobs=10, chunk_size=150000)
test_data = test_fg.generate()

# In[ ]:


X = training_data.drop(['target', 'seg_id'], axis=1)
X_test = test_data.drop(['target', 'seg_id'], axis=1)
test_segs = test_data.seg_id
y = training_data.target

# In[ ]:


folds = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(X), 1))
test_preds = np.zeros((len(X_test), 1))

# In[ ]:


params = {
    "learning_rate": 0.01,
    "max_depth": 4,
    "n_estimators": 10000,
    "min_child_weight": 1,
    "colsample_bytree": 0.9,
    "subsample": 1.0,
    "nthread": 12,
    "random_state": 42,
}

# In[ ]:


for fold_, (trn_, val_) in enumerate(folds.split(X)):
    print("Current Fold: {}".format(fold_))
    trn_x, trn_y = X.iloc[trn_], y.iloc[trn_]
    val_x, val_y = X.iloc[val_], y.iloc[val_]

    clf = xgb.XGBRegressor(**params)
    clf.fit(
        trn_x, trn_y,
        eval_set=[(trn_x, trn_y), (val_x, val_y)],
        eval_metric='mae',
        verbose=150,
        early_stopping_rounds=100
    )
    val_pred = clf.predict(val_x, ntree_limit=clf.best_ntree_limit)
    test_fold_pred = clf.predict(X_test, ntree_limit=clf.best_ntree_limit)
    print("MAE = {}".format(metrics.mean_absolute_error(val_y, val_pred)))
    oof_preds[val_, :] = val_pred.reshape((-1, 1))
    test_preds += test_fold_pred.reshape((-1, 1))
test_preds /= 5

oof_score = metrics.mean_absolute_error(y, oof_preds)
print("Mean MAE = {}".format(oof_score))

# In[ ]:


submission = pd.DataFrame(columns=['seg_id', 'time_to_failure'])
submission.seg_id = test_segs
submission.time_to_failure = test_preds
submission.to_csv('submission.csv', index=False)


# In[ ]:



