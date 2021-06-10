from scipy import stats
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif,mutual_info_regression, chi2, f_regression, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, log_loss
import multiprocessing
import time
from info_gain import info_gain
from pathos.multiprocessing import ProcessingPool
import tensorflow_hub as hub
import tensorflow as tf
path = "/data/home/hsaf/pycharm_projects/module/module_elmo/"
def import_elmo(path = "/data/home/hsaf/pycharm_projects/module/module_elmo/"):
    global elmo
    elmo = hub.load(path)

def embd_dict(sentense):
    input_text = tf.reshape(sentense, [-1])
    embeddings = elmo.signatures['default'](input_text)['default']
    embeddings = embeddings.numpy()
    return embeddings

import_elmo()

def compute_acc_with_cv(dataset, cand_dataset, folds,iterations,metric, target_name = 'target' ,
                        acc_decimals = 4,acc_dist=False,n_estimators=100,ccp_alpha = 0.0):
    ''' Received two datasets, execute RF model fols * iterations times and perform t test on the differences vector

        Args:
              dataset (DF) -  dataset
              cand_dataset (DF) -another dataset (the null hypothesis- this dataset is better)
              folds (int) - number of folds for cross validation
              iterations (int) - number of iterations of k folds CV.
              metric (str) - auc/acc
              target_name (str='target') =  the name of the target column
              acc_decimals (int = 4) = how to round the returned results

        Returns:
             mean difference, mean of the datasets, mean of the cand dataset, p value '''

    # convert to np
    y = np.array(dataset[target_name])
    # create X data
    X_dataset = dataset.drop([target_name], 1)
    X_cand_dataset = cand_dataset.drop([target_name], 1)
    X = np.array(pd.get_dummies(X_dataset))
    X_cand = np.array(pd.get_dummies(X_cand_dataset))
    # initializations
    # save classes (for the auc function)
    classes = dataset['target'].unique()
    # create dict which will hold the results
    scores = {'acc': [], 'acc_cand': []}
    # ccp_alpha = 0.0
    # if dataset.shape[0] <600: ccp_alpha = 0.05
    # else: ccp_alpha = 0.005
    # first chose ccp alpha
    rf = RandomForestClassifier(min_samples_leaf=1, n_estimators=n_estimators, n_jobs=6,
                                ccp_alpha=ccp_alpha)
    # Start iterations
    print('ccp_alpha is: ', ccp_alpha)
    for iter in range(iterations):
        sss = StratifiedKFold(n_splits=folds, shuffle=True, random_state=iter)#StratifiedShuffleSplit(n_splits=folds,random_state=iter)
        # 5 fold cross validation

        for train_idx, test_idx in sss.split(X, y):

            # split to train and test
            y_train, y_test = y[train_idx], y[test_idx]
            X_train, X_test = X[train_idx], X[test_idx]
            X_cand_train, X_cand_test = X_cand[train_idx], X_cand[test_idx]
            # train and fit values
            model = rf.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            acc_train = roc_auc_score(label_binarize(y_train, classes=classes),
                                      label_binarize(y_train_pred, classes=classes))

            # print('train auc: '+ '{},{}\n'.format(acc_train,dataset.shape[0]))

            # f = open("AUC_ccp0.001.txt", "a")
            # f.write('{},{}\n'.format(acc_train,dataset.shape[0]))
            # f.close()

            y_pred = model.predict(X_test)
            model_cand = rf.fit(X_cand_train, y_train)
            y_cand_pred = model_cand.predict(X_cand_test)
            # compute metric
            if metric == 'acc':
                acc, acc_cand = np.mean(y_pred == y_test), np.mean(y_cand_pred == y_test)
            elif metric == 'logloss':
                y_pred = model.predict_proba(X_test)
                y_cand_pred = model_cand.predict_proba(X_cand_test)
                acc = log_loss(y_true=y_test,y_pred = y_pred)
                acc_cand = log_loss(y_true=y_test,y_pred = y_cand_pred)

            else:
                acc = roc_auc_score(label_binarize(y_test,classes = classes),
                                label_binarize(y_pred,classes = classes))

                acc_cand = roc_auc_score(label_binarize(y_test,classes = classes),
                                     label_binarize(y_cand_pred,classes = classes))
            # save results
            scores['acc'].append(acc)
            scores['acc_cand'].append(acc_cand)

        # print(acc_train)
    # summarise results
    mean_acc, mean_acc_cand = np.array(scores['acc']).mean(), np.array(scores['acc_cand']).mean()
    diff_acc_mean = mean_acc_cand - mean_acc
    # t test
    t = stats.ttest_rel(scores['acc_cand'] , scores['acc'])[0]
    # compute one side p value
    df = folds*iterations-1
    p = (1 - stats.t.cdf(t, df))
    # return distribution of the acc (for dataset meta fetures)

    return  mean_acc.round(acc_decimals), np.std(scores['acc']).round(acc_decimals), np.min(scores['acc']).round(acc_decimals), np.max(scores['acc']).round(acc_decimals)


def statMetaFeat(cand_feature_name, candidate_features, target, exist_features_for_F, features_nulls_and_type_df, dataset_feat_exist,exist_features_dtype):
    features_statistic_meta_feat = {'feature_name': [], 'feature_type': [], 'dtype': [], 'discrete': [],
                                    'dist_min': [], 'dist_max': [], 'dist_mean': [], 'dist_median': [],
                                    'dist_sd': [],
                                    'gof_normal': [], 'gof_exp': [], 'gof_unif': [],
                                    'num_of_levels': [], 'min_rec_in_level': [], 'max_rec_in_level': [],
                                    'mean_rec_in_level': [], 'sd_rec_in_level': [],
                                    'ratio_of_nulls': [],
                                    'ig': [], 'mutual_ig': [], 'ig_ratio': [], 'intrinsic_value': [], 'mRMR': [],
                                    'redundancy': [],
                                    'target_f': [], 'target_f_p': [], 'target_chi': [],
                                    'target_chi_p': [],
                                    'exi_feat_sd_mutual_ig': [], 'exi_feat_min_mutual_ig': [],
                                    'exi_feat_median_mutual_ig': [], 'exi_feat_mean_mutual_ig': [],
                                    'exi_feat_max_mutual_ig': [],
                                    'exi_feat_sd_f': [], 'exi_feat_min_f': [], 'exi_feat_median_f': [],
                                    'exi_feat_mean_f': [], 'exi_feat_max_f': []}
    print(cand_feature_name)

    # save type and ratio of nulls
    nulls_and_type = features_nulls_and_type_df.loc[features_nulls_and_type_df['feature_name'] == cand_feature_name, :]
    features_statistic_meta_feat['feature_type'].append(nulls_and_type['feature_type'].values[0])
    features_statistic_meta_feat['ratio_of_nulls'].append(nulls_and_type['ratio_of_nulls'].values[0])

    # chose the current feature
    cand_feature = candidate_features[cand_feature_name]

    # save it dtype
    cand_feature_dtype = str(cand_feature.dtype)
    # for Binary features change dtype to binary
    dtype_meta_feture = cand_feature_dtype
    if 'BINARY_' in str(cand_feature_name):
        dtype_meta_feture = 'Binary'

    # for ig computation determine if discrete or not
    if dtype_meta_feture in ['object', 'Binary']:
        discrete_features = True
    else:
        discrete_features = False

    cand_as_array = cand_feature.copy()
    """NUMERIC FEAT  - GOF AND DIST"""
    if discrete_features == False:
        cand_as_array = np.array(cand_as_array)
        # compute dist componnets
        dist_sd, dist_min, dist_median, dist_mean, dist_max = cand_as_array.std(), cand_as_array.min(), np.median(
            cand_as_array), cand_as_array.mean(), cand_as_array.max()

        # compute gof for sits
        stats.kstest(cand_as_array, 'norm')
        gof_normal = stats.kstest(cand_as_array, 'norm')[1]
        gof_exp = stats.kstest(cand_as_array, 'expon')[1]
        gof_unif = stats.kstest(cand_as_array, 'uniform')[1]
        # gof_lognormal= stats.kstest(cand_as_array,'lognorm')[1]
    else:
        """for discrete FEAT  - GOF AND DIST"""
        dist_sd, dist_min, dist_median, dist_mean, dist_max, gof_normal, gof_exp, gof_unif, = -1, -1, -1, -1, -1, -1, -1, -1

    """DISCRETE FEAT  - NUM OF VALUES AND DIST"""
    if discrete_features == True:

        num_of_levels = len(cand_feature.unique())
        num_of_records_in_level = cand_feature.value_counts()
        sd_rec_in_level, min_rec_in_level, mean_rec_in_level, max_rec_in_level = num_of_records_in_level.std(), \
                                                                                 num_of_records_in_level.min(), num_of_records_in_level.mean(), num_of_records_in_level.max()

    else:
        """for numeric FEAT  - NUM OF VALUES AND DIST"""
        num_of_levels, min_rec_in_level, max_rec_in_level, mean_rec_in_level, sd_rec_in_level = 1, -1, -1, -1, -1

    """IG, F, CHI AGAINST TARGET"""
    # if its a categorical get dummies
    if cand_feature_dtype == "object":

        cand_as_array = np.array(pd.get_dummies(cand_as_array))
        X_for_ig = cand_as_array.copy()
        X_for_chi = cand_as_array.copy()
    else:

        cand_as_array = np.array(cand_as_array)
        cand_as_array = cand_as_array.reshape(-1, 1)  # convert to list of lists

        X_for_ig = cand_as_array.copy()
        X_for_chi = cand_as_array.copy()
        X_for_chi[X_for_chi < 0] = 0  # negative values not allowed

    # ig versions against target
    ig = info_gain.info_gain(cand_feature.values, target)
    intrinsic_value = info_gain.intrinsic_value(cand_feature.values, target)
    ig_ratio = info_gain.info_gain_ratio(cand_feature.values, target)
    # mutual ig, f, chi against target
    mutual_ig = mutual_info_classif(X_for_ig, target, random_state=0, discrete_features=discrete_features)

    # chose the max ig (relevant for dummy variables only)
    mutual_ig = mutual_ig.max()
    # f test against target
    target_f, target_f_p = f_classif(X_for_ig, target)
    target_f = target_f.max()
    target_f_p = target_f_p.min()
    # chi test
    target_chi, target_chi_p = chi2(X_for_chi, target)
    target_chi = target_chi.max()
    target_chi_p = target_chi_p.min()

    """IG and F test against the original features (only if there are exist features)"""

    if dataset_feat_exist == True:
        start = time.time()
        if cand_feature_dtype == "object":
            # mutual ig

            ex_feat_mutual_ig = mutual_info_classif(exist_features_for_F, cand_feature.values.astype(str),
                                                    random_state=0,discrete_features=exist_features_dtype)
            # perform F test
            f_values, p_values = f_classif(exist_features_for_F, cand_feature.values.astype(str))
        else:
            # ig
            ex_feat_mutual_ig = mutual_info_regression(exist_features_for_F, cand_feature.values, random_state=0,discrete_features=exist_features_dtype)
            # F
            f_values, p_values = f_regression(exist_features_for_F, cand_feature.values)

        sd_mutual_ig, min_mutual_ig, median_mutual_ig, mean_mutual_ig, max_mutual_ig = ex_feat_mutual_ig.std(), ex_feat_mutual_ig.min(), np.median(
            ex_feat_mutual_ig), ex_feat_mutual_ig.mean(), ex_feat_mutual_ig.max()

        f_values[f_values == np.inf] = 200
        f_values[f_values == -np.inf] = 0
        sd_f, min_f, median_f, mean_f, max_f = f_values.std(), f_values.min(), np.median(
            f_values), f_values.mean(), f_values.max()
        redundancy = ex_feat_mutual_ig.sum() / (exist_features_for_F.shape[1] + 1)
        mRMR = ig - redundancy
    # if there is no exist features f mean and f max are 0
    else:
        sd_f, min_f, median_f, mean_f, max_f = 0, 0, 0, 0, 0
        sd_mutual_ig, min_mutual_ig, median_mutual_ig, mean_mutual_ig, max_mutual_ig = 0, 0, 0, 0, 0
        redundancy, mRMR = 0, 0

    # assign to dictionary
    features_statistic_meta_feat['feature_name'].append(cand_feature_name)
    features_statistic_meta_feat['dtype'].append(dtype_meta_feture)
    features_statistic_meta_feat['discrete'].append(discrete_features)
    features_statistic_meta_feat['dist_min'].append(dist_min)
    features_statistic_meta_feat['dist_max'].append(dist_max)
    features_statistic_meta_feat['dist_mean'].append(dist_mean)
    features_statistic_meta_feat['dist_median'].append(dist_median)
    features_statistic_meta_feat['dist_sd'].append(dist_sd)
    features_statistic_meta_feat['gof_normal'].append(gof_normal)
    features_statistic_meta_feat['gof_exp'].append(gof_exp)
    features_statistic_meta_feat['gof_unif'].append(gof_exp)
    features_statistic_meta_feat['num_of_levels'].append(num_of_levels)
    features_statistic_meta_feat['min_rec_in_level'].append(min_rec_in_level)
    features_statistic_meta_feat['mean_rec_in_level'].append(mean_rec_in_level)
    features_statistic_meta_feat['max_rec_in_level'].append(max_rec_in_level)
    features_statistic_meta_feat['sd_rec_in_level'].append(sd_rec_in_level)
    features_statistic_meta_feat['mutual_ig'].append(mutual_ig)
    features_statistic_meta_feat['ig'].append(ig)
    features_statistic_meta_feat['intrinsic_value'].append(intrinsic_value)
    features_statistic_meta_feat['ig_ratio'].append(ig_ratio)
    features_statistic_meta_feat['mRMR'].append(mRMR)
    features_statistic_meta_feat['redundancy'].append(redundancy)
    features_statistic_meta_feat['target_f'].append(target_f)
    features_statistic_meta_feat['target_f_p'].append(target_f_p)
    features_statistic_meta_feat['target_chi'].append(target_chi)
    features_statistic_meta_feat['target_chi_p'].append(target_chi_p)
    features_statistic_meta_feat['exi_feat_sd_mutual_ig'].append(sd_mutual_ig)
    features_statistic_meta_feat['exi_feat_min_mutual_ig'].append(min_mutual_ig)
    features_statistic_meta_feat['exi_feat_mean_mutual_ig'].append(mean_mutual_ig)
    features_statistic_meta_feat['exi_feat_median_mutual_ig'].append(median_mutual_ig)
    features_statistic_meta_feat['exi_feat_max_mutual_ig'].append(max_mutual_ig)
    features_statistic_meta_feat['exi_feat_sd_f'].append(sd_f)
    features_statistic_meta_feat['exi_feat_min_f'].append(min_f)
    features_statistic_meta_feat['exi_feat_mean_f'].append(mean_f)
    features_statistic_meta_feat['exi_feat_median_f'].append(median_f)
    features_statistic_meta_feat['exi_feat_max_f'].append(max_f)
    # statistic metafeatures into df
    features_statistic_meta_feat_df = pd.DataFrame.from_dict(features_statistic_meta_feat)
    return features_statistic_meta_feat_df

def meta_features_extraction(original_dataset, statistic_features_df, categories_fearures_df, properties_fearures_df, ratio_of_nulls):
    """Recives original dataset and external features, returns dataset- meta-features and features-meta-features"""
    # join all cand features together
    candidate_features = statistic_features_df.join(categories_fearures_df).join(properties_fearures_df)
    candidate_features = candidate_features.fillna('other')

    # create a dictionary which will hold the vlaues of the meta features
    features_nulls_and_type = { 'feature_type': [],'ratio_of_nulls': []}

    # add to features dict
    features_nulls_and_type['ratio_of_nulls'] = ratio_of_nulls
    """save cand features type: statistic/property etc (one of the meta features)"""
    # properties: count/property
    properties_fearures_type = []
    for featrue_name in properties_fearures_df.columns:
        if 'BINARY_' in featrue_name:
            type = 'Property_B'
        elif featrue_name == 'wikiPageOutDegree':
            type = 'Statistic'
        else:
            type = 'Property'
        properties_fearures_type.append(type)

    # categories features type
    categories_fearures_type = []
    for featrue_name in categories_fearures_df.columns:
        featrue_name = str(featrue_name)
        if 'Category:' in featrue_name:
            type = 'Category'
        elif 'LINK_' in featrue_name:
            type = 'Link'
        elif 'BOW_BINARY_' in featrue_name:
            type = 'BOW_B'
        elif 'BOW_' in featrue_name:
            type = 'BOW'
        else:
            type = 'EMBD'
        categories_fearures_type.append(type)

    # statastic
    statistic_fearures_type = ['Statistic'] * statistic_features_df.shape[1]
    # concatenate all
    features_type = statistic_fearures_type + \
                    categories_fearures_type + \
                    properties_fearures_type
    # save to meta features dict
    features_nulls_and_type['feature_type'] = features_type

    ''' compute IG, f statistic and chi squar test'''
    # exclude the target from the dataset
    target = np.array(original_dataset['target'])
    exist_features = original_dataset.drop(['target', 'lookup'], 1)


    # if there is features - get there dummies
    if exist_features.shape[1] > 0:
        dataset_feat_exist = True
        exist_features_with_dummies = pd.get_dummies(exist_features)
        exist_features_for_F = np.array(exist_features_with_dummies)
        # for mutual info computations
        exist_features_dtype = (exist_features_with_dummies.dtypes == 'uint8').values
    else:
        dataset_feat_exist = False
        exist_features_for_F = None
        exist_features_dtype = None

    cand_features_list = candidate_features.columns.values

    # convert to df
    features_nulls_and_type_df = pd.DataFrame.from_dict(features_nulls_and_type)
    features_nulls_and_type_df['feature_name'] = cand_features_list

    candidate_features.sort_index(inplace=True)  # sort for faster retrieval

    # extract features-meta-features on each feature
    workers = multiprocessing.cpu_count()-20
    if 'pool' not in globals():
        print('Create new pool for meta features')
    else:
        print('Pool already exist for meta features')

    pool = ProcessingPool(workers)

    features_statistic_df_list = pool.map(lambda x: statMetaFeat(x, candidate_features, target, exist_features_for_F,features_nulls_and_type_df,dataset_feat_exist,exist_features_dtype),cand_features_list)
    pool.close()
    pool.join()
    pool.terminate()
    pool.clear()

    features_statistic_meta_feat_df = pd.concat(features_statistic_df_list)
    # fill na with 0
    features_statistic_meta_feat_df = features_statistic_meta_feat_df.fillna(0)
    # fill inf with max*2 values
    def covert_inf_to_max(column):
        print(column.name)
        new_column = column
        if str(column.dtype) != 'object':
            max = column.replace([np.inf], np.nan).dropna().max()
            new_column = column.replace([np.inf], max * 2)
        return new_column

    features_statistic_meta_feat_df = features_statistic_meta_feat_df.apply(covert_inf_to_max)
    '''Dataset meta-features'''
    num_of_candidates = candidate_features.shape[1]
    # initial auc/logloss distribution
    mean_auc, sd_auc, min_auc, max_auc = compute_acc_with_cv(original_dataset, original_dataset,
                                                             metric='auc', folds=2,
                                                             acc_decimals=4,
                                                             iterations=15, acc_dist=True)
    mean_loss, sd_loss, min_loss, max_loss = compute_acc_with_cv(original_dataset, original_dataset,
                                                             metric='logloss', folds=2,
                                                             acc_decimals=4,
                                                             iterations=15, acc_dist=True)


    """ig, chi squar and f tests dist - exist features against target"""
    exist_features = original_dataset.drop(['target', 'lookup'], 1)
    if exist_features.shape[1] > 0:
        columns_for_dummy = original_dataset.drop(['target', 'lookup'], 1).loc[
                            :, original_dataset.dtypes == 'object'].columns.values


        exist_features_for_f_ig = np.array(pd.get_dummies(original_dataset.drop(['target', 'lookup'], 1), columns=columns_for_dummy))
        # compute f
        dataset_f = f_classif(exist_features_for_f_ig, target)[0]
        dataset_f[dataset_f==np.inf] = 200
        dataset_f[dataset_f == -np.inf] = 0

        sdFforDiscreteFeat, meanFforDiscreteFeat, maxFforDiscreteFeat, minFforDiscreteFeat = \
            dataset_f.std(), dataset_f.mean(), dataset_f.max(), dataset_f.min()

        # compute ig
        dataset_ig = mutual_info_classif(exist_features_for_f_ig, target, random_state=0)
        sdIGforDiscreteFeat, meanIGforDiscreteFeat, maxIGforDiscreteFeat, minIGforDiscreteFeat = \
            dataset_ig.std(), dataset_ig.mean(), dataset_ig.max(), dataset_ig.min()

        if len(columns_for_dummy) > 0:
            exist_features_for_chi = np.array(pd.get_dummies(original_dataset[columns_for_dummy]))
            dataset_chi2 = chi2(exist_features_for_chi, target)[0]
            sdChiforDiscreteFeat, meanChiforDiscreteFeat, maxChiforDiscreteFeat, minChiforDiscreteFeat = \
                dataset_chi2.std(), dataset_chi2.mean(), dataset_chi2.max(), dataset_chi2.min()

        else: sdChiforDiscreteFeat,meanChiforDiscreteFeat,maxChiforDiscreteFeat,minChiforDiscreteFeat = 0,0,0,0
    else:

        sdFforDiscreteFeat, meanFforDiscreteFeat, maxFforDiscreteFeat, minFforDiscreteFeat = 0, 0, 0, 0
        sdIGforDiscreteFeat, meanIGforDiscreteFeat, maxIGforDiscreteFeat, minIGforDiscreteFeat = 0, 0, 0, 0
        sdChiforDiscreteFeat, meanChiforDiscreteFeat, maxChiforDiscreteFeat, minChiforDiscreteFeat = 0, 0, 0, 0



    # numeric features bol series
    numeric_feat = (original_dataset.drop('target', 1).dtypes != 'object')

    dataset_meta_feat = {'numOfInstances': [original_dataset.shape[0]] * num_of_candidates,
                         'numOfFeatures': [original_dataset.shape[1]] * num_of_candidates,
                         'numOfClasses': [len(original_dataset['target'].unique())] * num_of_candidates,
                         'meanAUC': [mean_auc] * num_of_candidates,
                         "sdvAUC": [sd_auc] * num_of_candidates,
                         "maxAUC": [max_auc] * num_of_candidates,
                         "minAUC": [min_auc] * num_of_candidates,
                         'meanLogLoss': [mean_loss] * num_of_candidates,
                         "sdvLogLoss": [sd_loss] * num_of_candidates,
                         "maxLogLoss": [max_loss] * num_of_candidates,
                         "minLogLoss": [min_loss] * num_of_candidates,
                         'sdFforDiscreteFeat':[sdFforDiscreteFeat] * num_of_candidates,
                         'meanFforDiscreteFeat':[meanFforDiscreteFeat] * num_of_candidates,
                         'maxFforDiscreteFeat':[maxFforDiscreteFeat] * num_of_candidates,
                         'minFforDiscreteFeat':[minFforDiscreteFeat] * num_of_candidates,
                         'sdIGforDiscreteFeat':[sdIGforDiscreteFeat] * num_of_candidates,
                         'meanIGforDiscreteFeat':[meanIGforDiscreteFeat] * num_of_candidates,
                         'maxIGforDiscreteFeat':[maxIGforDiscreteFeat] * num_of_candidates,
                         'minIGforDiscreteFeat':[minIGforDiscreteFeat] * num_of_candidates,
                         'sdChiforDiscreteFeat':[sdChiforDiscreteFeat] * num_of_candidates,
                         'meanChiforDiscreteFeat':[meanChiforDiscreteFeat] * num_of_candidates,
                         'maxChiforDiscreteFeat':[maxChiforDiscreteFeat] * num_of_candidates,
                         'minChiforDiscreteFeat':[minChiforDiscreteFeat] * num_of_candidates,
                         'featuresInDataset': [original_dataset.shape[1] > 2] * num_of_candidates,
                         'numOfNumericFeat': [numeric_feat.sum()] * num_of_candidates,
                         'numOfDiscreteFeat': [(~numeric_feat).sum()] * num_of_candidates,
                         'ratioOfNumericFeat': [numeric_feat.mean().round(3)] * num_of_candidates,
                         'ratioOfDiscreteFeat': [(~numeric_feat).mean().round(3)] * num_of_candidates,
                         'numOfIntities': [len(original_dataset['lookup'].unique())] * num_of_candidates,
                         'ratioOfIntitiesInstances':[(len(original_dataset['lookup'].unique()))/original_dataset.shape[0]] * num_of_candidates

                         }

    # to df
    dataset_meta_feat = pd.DataFrame.from_dict(dataset_meta_feat)


    # join dataset metafeatures and features meta features
    meta_features = features_statistic_meta_feat_df.join(dataset_meta_feat).reset_index(drop=True)

    print('=======Meta features extraction done=========')

    return meta_features


