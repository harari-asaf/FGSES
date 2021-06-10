from models import neural_net
import numpy as np
import pandas as pd
import joblib
def feature_eng(data):
    """
    This func apply preprocessing on the data and return new df
    :param data: df of the train or test data
    :return: df after pre processing
    """
    ### train pre processing
    # add binary feat to the highest 1% of ig, target f and target chi
    records_list = []
    for dataset in data['dataset'].unique():
        # isolate current dataset
        dataset_records = data.loc[data['dataset'] == dataset, :]
        # compute 99 quantile of ig f chi
        ig_f_chi_99_quntile = dataset_records[['ig', 'target_f', 'target_chi']].quantile(q=0.999)
        # create 3 new binary  features low/high 99 quntile
        dataset_records[['ig_99_per', 'target_f_99_per', 'target_chi_99_per']] = dataset_records[
                                                                                     ['ig', 'target_f',
                                                                                      'target_chi']] >= ig_f_chi_99_quntile
        records_list.append(dataset_records)

    data = pd.concat(records_list)
    return data

def prepare_data(data):
    none_feat_cols = ['dataset', 'feature_name']
    # create dummy
    columns_to_remove = [ 'target_chi', 'target_chi_p','target_chi_99_per']
    v = ~data.columns.isin(columns_to_remove)
    data = data.loc[:,v]
    # update featlist after preprocessing
    features = list(data.drop(none_feat_cols, 1).columns.values)
    new_feature_type = data.feature_type.astype(pd.api.types.CategoricalDtype(
        categories=['Statistic', 'BOW', 'BOW_B', 'EMBD', 'Link', 'Property_B','Property', 'Category'], ordered=False))
    data = data.assign(feature_type=new_feature_type)
    new_dtype = data.dtype.astype(pd.api.types.CategoricalDtype(
         categories=['int64', 'float64', 'Binary', 'object'], ordered=False))
    data = data.assign(dtype=new_dtype)


    columns_for_dummy = ['feature_type','dtype']
    data_X = pd.get_dummies(data[features], columns=columns_for_dummy).astype(int)
    # print(data_X.columns)
    # np.save('mModel_feta_importance/meta_feat_names',data_X.columns)
    data_X  = np.array(data_X)
    final_metaFeat = np.load('models/final_metaFeat.npy')
    scaler = joblib.load('models/scaler.pkl')
    scaled_data_X = scaler.transform(data_X)[:, final_metaFeat]

    # return np array with dummies and no target
    return scaled_data_X


def candidate_feature_ranking(meta_features):
    """Receives meta features, process them and infer about each cand feature. return sorted candidate features"""
    # meta fetures preprocessing
    meta_features_eng = feature_eng(meta_features)
    meta_data_X = prepare_data(meta_features_eng)
    # load pre-trained meta-model
    meta_model = neural_net.neural_net(input_shape=meta_data_X.shape[1])
    # assign score to each candidate features by the meta-features
    scores = meta_model.predict(meta_data_X)
    # sort candidate features by score
    meta_features[['score']] = scores
    ranked_cand_features = meta_features[:['feature_name','score']].sort_values('score', ascending=False)
    # return
    return ranked_cand_features