from entity_matching import entity_matching
from candidate_features_extraction import candidate_features_extraction
from meta_features_extraction import meta_features_extraction
from candidate_feature_ranking import candidate_feature_ranking
from evaluation import evaluate_dataset


def FGSES(dataset,dataset_name, DBpedia_DB_connection,maxIter,folds):
    """Entity Matching"""
    # Match each dataset entity with DBpedia entity
    dataset_after_matching = entity_matching(dataset,dataset_name,DBpedia_DB_connection)
    """Candidate Features Extraction"""
    # Extract candidate features from each DBpedia entity
    candidate_features, categories_features ,statistic_features ,properties_features, ratio_of_nulls = candidate_features_extraction(dataset_after_matching, DBpedia_DB_connection)

    """Candidate_Feature_Evaluation & Selection"""
    rank_cand_features = True # Rank candidate features only in the first iteration and after each candidate feature addition
    evaluated_cand_features = [] # save each evaluated feature
    selected_cand_features = [] # save final set of candidate features
    dataset_initial_AUC = evaluate_dataset(dataset,folds)
    last_iter_AUC = dataset_initial_AUC
    for i in maxIter:
        """Candidate_Feature_Ranking"""
        if rank_cand_features:
            meta_features = meta_features_extraction(dataset, statistic_features, categories_features,
                                                     properties_features,
                                                     ratio_of_nulls)

            rank_cand_features = False
            # Rank candidate features acording to the meta-features
            ranked_cand_features = candidate_feature_ranking(meta_features)

        # remove candidates which already been evaluated
        v = ~ranked_cand_features['feature_name'].isin(evaluated_cand_features)
        # pop best cand
        best_cand_for_now = ranked_cand_features.loc[v, 'feature_name'].values.astype(str)[0]
        evaluated_cand_features.append(best_cand_for_now)
        # evaluate the best cand
        dataset_for_evaluation = dataset.join(candidate_features[best_cand_for_now])
        current_AUC = evaluate_dataset(dataset_for_evaluation,folds)
        error_reduc = current_AUC - last_iter_AUC

        # if the error reduction is greater then the threshold, add the current candidate feature to the dataset
        if error_reduc >0.01:
            selected_cand_features.append(best_cand_for_now)
            rank_cand_features = True
            dataset = dataset_for_evaluation
            last_iter_AUC = current_AUC

    # return final set of candidate features
    return selected_cand_features

if __name__ == '__main__':

    selected_cand_features = FGSES(dataset, dataset_name, DBpedia_DB_connection, maxIter, folds)





