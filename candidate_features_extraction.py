import pandas as pd
import pymysql # connect to mysql in linux
import re
import os
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
ps = PorterStemmer()
from collections import Counter
from timeit import default_timer as timer
import tensorflow_hub as hub
import tensorflow as tf

def import_elmo(path = "/data/home/hsaf/pycharm_projects/module/module_elmo/"):
    global elmo
    elmo = hub.load(path)


def embd_dict(sentense):
    input_text = tf.reshape(sentense, [-1])
    embeddings = elmo.signatures['default'](input_text)['default']
    embeddings = embeddings.numpy()
    return embeddings


# query result into df
def queryRes(cur):
    results = pd.DataFrame(list(cur.fetchall()))
    if len(results)>1:
        results.columns = [i[0] for i in cur.description]
    return results


def update_feat_dict(new_names, new_values, features_dict,fill_empty_cells_with, records_num):
    ''' Received entity  features (list of features names, values) and add to the dict of the exist features (add one element to each list.
        add new feature if necessary, add "count"  feature to each feature,
        if feature appear more than once his name will be name_i.

    Args:
          new_names (list) - list of the names of the new features
          new_values (list) - list of the values of the new features
          features_dict (dict of lists) -  each key and list is a feature
          fill_empty_cells_with (int/none) - How to fill former features which is not exist in the current entity
          records_num (int)- the index of the row of the entity in the dataset.

    Returns:
         features_dict (dict) - updated dict of lists, each key and list is a feature'''
    # initialize list of added  properties
    added_feat_names = []
    for name, value in zip(new_names,new_values):
        # print(name, value)
        add_i = 0
        while True:
            # for new feature -
            if name not in features_dict:
                # insert to dict and fill all former records with none or 0 (for count/binary features)
                if any(w in name for w in ['BOW','LINK','BINARY_']):
                    features_dict[name] = [0] * records_num
                else: features_dict[name] = [fill_empty_cells_with]*records_num
                # add the current value in the end
                features_dict[name].append(value)

            # if the name is duplicated in the same entity add _i
            elif name in added_feat_names:
                name = name +'_'+ str(add_i)
                name = name.replace('_' + str(add_i - 1),'')
                add_i += 1
                # for count feature increase value by 1
                name_before_addi = name.split('_'+str(add_i-1))[0]
                # increase first count feature
                if '_count' in name_before_addi:
                    features_dict[name_before_addi][records_num] = add_i
                continue

            # if the feature name exist (been added in former eintity)
            else:
                # add the value to his list
                features_dict[name].append(value)
            # save the name which been added
            added_feat_names.append(name)
            # go to next feature
            break

    # fill properties that not exist in current entity with Nones (count features with 0 )
    for key in features_dict.keys():
        if key not in added_feat_names:
            if any(w in key for w in ['BOW', 'LINK', 'BINARY_']):
                features_dict[key].append(0)
            else:
                features_dict[key].append(fill_empty_cells_with)

    return features_dict


def string_to_tokens_count(string):
    """Recived string, remove special characters and count tokens
    return dict token:count """
    # remove special chars
    parsed_string = re.sub(r'[^\.a-zA-Z0-9 ]', r'', string).replace(",", "").replace(".", "")
    # tokenize
    parsed_string_tokens = word_tokenize(parsed_string)
    # stem and remove stop words
    stop_words = set(stopwords.words('english'))
    parsed_string_tokens = [ps.stem(w) for w in parsed_string_tokens if not w.lower() in stop_words]
    # count tokens
    count_tokens = Counter(parsed_string_tokens)
    return count_tokens

def covert_nan_to_other_or_mean(column):
    """for add_properties_func. received panda series and fill nans.
        numeric with mean, categorical with other"""

    if column.dtype == 'object':
        new_column = column.fillna('other')
    else:
        avg = column.mean()
        new_column = column.fillna(avg)
    return new_column

def add_categories_feat(dataset,want_ratio_of_nulls = False, permitted_ratio_of_nulls = 0.96):
    ''' Retrieve all wikilinks and abstract  add them as binary(object)/count(int64) features
    wikilinks as is and as BOW abstract as BOW, removes wikilinks with null ratio greater than defined

    Args:
          dataset (DF) - Dataset with 'lookup' column containing wiki_page_names
          want_ratio_of_nulls (bol = False) - If to return a list with missing values ratio to each feature
          permitted_ratio_of_nulls (int = 0.96) - Feature with ratio of nulls grater than this will be removed

    Returns: DF with the new features (and list of null ratio)'''
    unique_lookups = dataset['lookup'].unique()
    print('Add links and BOW')
    # create dict which will hold all categories features
    features_dict = {}
    embeddings_list = []
    n = 0
    for lookup_value in unique_lookups:
        print(lookup_value)
        start = timer()
        # import links
        query = """select  linked_page  from Tbl_page_wikilinks
                    where  wiki_page_name = """ + '"' + lookup_value + '"' # linked_page LIKE 'Category:%' AND
        cur.execute(query)
        wikilinks = list(cur.fetchall())
        entity_wikilinks_names, entity_wikilinks_values = [], []
        # list all wikilinks which been found and list of ones
        parsed_wikilinks = [wikilink[0].replace('_', ' ').strip() for wikilink in wikilinks]
        for i in parsed_wikilinks:
            # TODO: Change references from 'link ' to LINK_
            entity_wikilinks_names.append('LINK_'+i)
            entity_wikilinks_values.append(str(1))

        print('links been extracted ', timer()-start)
        # import abstract
        query = """select  abstract  from Tbl_page_abstract
                                       where  wiki_page_name = """ + '"' + lookup_value + '"'
        num_of_records = cur.execute(query)
        abstract = ''
        if num_of_records > 0:
            abstract = list(cur.fetchone())[0]
        """Create abstract embeddings"""
        print('abstract been extracted ', timer()-start)
        abstract_for_elmo = process_abstract_for_elmo(abstract)

        abstracts_embeddings = embd_dict([abstract_for_elmo]).flatten()
        print('time for elmo calculations: ', timer()-start)
        embeddings_list.append(abstracts_embeddings)
        """create BOW from the wikilinks (not categories) list and abstract"""
        concaten_wikilinks =  ' '.join(parsed_wikilinks)

        wikilinks_abstract_tokens_count = string_to_tokens_count(concaten_wikilinks+abstract)
        num_of_words = len(wikilinks_abstract_tokens_count)

        entity_BOW_names = []
        entity_BOW_values = []
        idf_dict ={}
        # list all tokens which been from the wikilinks and acbstract
        for key,value in wikilinks_abstract_tokens_count.items():
            if len(key) > 1:
                # TODO: Change references to BOW_ and BOW_BINARY_
                # count feature
                entity_BOW_names.append(('BOW_' + key))
                Tf = value/num_of_words
                entity_BOW_values.append(Tf)
                # if 'BOW_' + key in idf_dict: idf_dict['BOW_' + key]

                # binary feature
                entity_BOW_names.append(('BOW_BINARY_' + key))
                entity_BOW_values.append(str(1))

        # concatenate wikilinks and BOW features
        entities_features = entity_wikilinks_names+entity_BOW_names
        entities_values = entity_wikilinks_values+ entity_BOW_values

        """Create BOW from categories"""
        # update the dict with the categories
        features_dict = update_feat_dict(new_names=entities_features,
                                         new_values=entities_values,
                                         features_dict=features_dict,
                                         fill_empty_cells_with=0,
                                         records_num = n)
        n += 1
        print('loop been done: ', timer() - start)
    # convert to df
    categories_fearures_df_with_BOW_B = pd.DataFrame.from_dict(features_dict)


    # compute ratio of nulls
    ratio_of_nulls = (categories_fearures_df_with_BOW_B==0).mean()

    # remove features with lots of nulls
    bol_less_than_r = ratio_of_nulls<permitted_ratio_of_nulls
    categories_fearures_df_with_BOW_B_filterd = categories_fearures_df_with_BOW_B.loc[:,bol_less_than_r.values]

    categories_fearures_df_only_BOW = categories_fearures_df_with_BOW_B_filterd.loc[:,
                                      ~categories_fearures_df_with_BOW_B_filterd.columns.str.contains('BOW_BINARY_')].filter(
        like='BOW')
    # compute tfidf
    idf = 1 / categories_fearures_df_with_BOW_B_filterd.filter(like='BOW_BINARY_').astype(int).mean(0)
    categories_fearures_df_only_tfidf = categories_fearures_df_only_BOW.apply(
        lambda x: x * idf['BOW_BINARY_' + x.name.split('BOW_')[1]])


    # filter out binary BOW and count BOW from original features
    categories_fearures_df = categories_fearures_df_with_BOW_B_filterd.loc[:,
                             ~categories_fearures_df_with_BOW_B_filterd.columns.str.contains('BOW_')]

    # join with tfidf features
    categories_fearures_df = categories_fearures_df.join(categories_fearures_df_only_tfidf).join(categories_fearures_df_with_BOW_B_filterd.filter(like='BOW_BINARY_'))

    embeddings_df = pd.DataFrame(embeddings_list)
    categories_fearures_df = categories_fearures_df.join(embeddings_df)
    # compute new ratio of nulls
    ratio_of_nulls = (categories_fearures_df==0).mean()  # actually  ratio of zeroes

    print('Num of links and word extracted: ' + str(categories_fearures_df.shape[1]))

    # expend to full dataset with duplicated values
    categories_fearures_df['lookup'] = unique_lookups

    categories_fearures_df_full = \
        dataset[['lookup']].reset_index().merge(categories_fearures_df, on='lookup', sort=False). \
            sort_values('index').drop(['lookup', 'index'], 1).reset_index(drop=True)



    # lookup_df = dataset.set_index('lookup', inplace=True)
    # lookup_df.head()
    # categories_fearures_df_lookup = categories_fearures_df.reset_index()[['lookup']].set_index('lookup')
    # start_join = timer
    # # categories_fearures_df_full = \
    # #     lookup_df.join(categories_fearures_df, on='lookup').reset_index()
    # print(timer - start_join)
    #
    # import dask.dataframe as dd
    # from dask.diagnostics import ProgressBar
    # ddata = dd.from_pandas(dataset[['lookup']].set_index('lookup'), npartitions=20)
    # with ProgressBar():
    #     categories_fearures_df_full = \
    #         ddata.merge(categories_fearures_df, on='lookup').compute()  # .reset_index()
    # categories_fearures_df_full = categories_fearures_df_full.reset_index()
    # categories_fearures_df_full.head()
    # return
    if want_ratio_of_nulls == True: return categories_fearures_df_full, list(ratio_of_nulls)
    return categories_fearures_df_full



"""Add Properties and there counts as features"""
def add_properties_feat(dataset,want_ratio_of_nulls = False,permitted_ratio_of_nulls = 0.96):
    ''' Create properties features , removes features with null ratio greater than defined

    Args:
          dataset (DF) - Dataset with 'lookup' column containing wiki_page_names
          want_ratio_of_nulls (bol = False) - If to return a list with missing values ratio to each feature
          permitted_ratio_of_nulls (int = 0.96) - Feature with ratio of nulls grater than this will be removed

    Returns: DF with the new features (and list of null ratio)'''
    unique_lookups = dataset['lookup'].unique()
    print('Add properties')
    features_dict = {}
    n = 0
    for lookup_value in unique_lookups:
        # properties
        # print('lookup_value: '+ lookup_value)
        query = """select property, value  from Tbl_page_properties where wiki_page_name = """ + '"' + lookup_value + '"'
        num_of_records = cur.execute(query)
        # print('Num of properties been found: ' + str(num_of_records))
        properties = list(cur.fetchall())
        # save property name and value, and set counts to 1
        entity_prop_names ,entity_prop_values  = [], []
        # create list of names and values, for each propertie create 'count' feature
        for i in properties:
            name = i[0].replace('_', ' ').strip()
            entity_prop_names.append(name)
            entity_prop_values.append(i[1].strip())
            # save counts for each propertie
            entity_prop_names.append('BINARY_'+name)
            entity_prop_values.append(str(1))

        # update features dict
        features_dict = update_feat_dict(new_names=entity_prop_names,
                                         new_values=entity_prop_values,
                                         features_dict=features_dict,
                                         fill_empty_cells_with=float('NaN'),
                                         records_num = n)
        n += 1

    properties_fearures_df = pd.DataFrame.from_dict(features_dict)
    " remove features with lots of nulls"
    # save how many nulls in each feature
    ratio_of_nulls = (properties_fearures_df.fillna(0)==0).mean()
    # remove features with lots of nulls
    bol_less_than_r = ratio_of_nulls<permitted_ratio_of_nulls
    properties_fearures_df_top_r = properties_fearures_df.loc[:,bol_less_than_r.values]
    ratio_of_nulls = (properties_fearures_df_top_r.fillna(0)==0).mean()

    print('Num of features extracted: ' + str(properties_fearures_df.shape[1]))
    print('Num of features after removing features with ratio nulls ratio greater than '+str(permitted_ratio_of_nulls)+' ', str(properties_fearures_df_top_r.shape[1]))

    # fill nulls with 'other' (categorical) and mean (numeric).
    properties_fearures_df_top_r = properties_fearures_df_top_r.apply(lambda x: pd.to_numeric(x,errors = 'ignore'))
    # convert all nans to other or mean
    properties_fearures_df_top_r = properties_fearures_df_top_r.apply(covert_nan_to_other_or_mean)

    # expend to full dataset with duplicated values
    properties_fearures_df_top_r['lookup'] = unique_lookups
    properties_fearures_df_top_r_full =\
        dataset[['lookup']].reset_index().merge(properties_fearures_df_top_r, on='lookup', sort=False).\
            sort_values('index').drop(['lookup','index'], 1).reset_index(drop=True)

    # return
    if want_ratio_of_nulls == True: return properties_fearures_df_top_r_full, list(ratio_of_nulls)
    return properties_fearures_df_top_r_full



""" ADD NEW FEATURES FROM STATISTIC TABLE"""
def add_statistic_feat(dataset,want_ratio_of_nulls = False):
    ''' Create statistic features

    Args:
          dataset (DF) - Dataset with 'lookup' column containing wiki_page_names
          want_ratio_of_nulls (bol = False) - If to return a list with missing values ratio to each feature

    Returns: DF with the new features'''
    unique_lookups = dataset['lookup'].unique()

    print('Add statistic')
    statistic_features_dict = {'abstract':[],'abstract_length':[],'external_links':[],'properties':[],'wikilinks':[],'in_wikilinks':[] }
    values_without_results = 0
    n = 0
    # len_dataset = 5#len(dataset)
    for lookup_value in unique_lookups:
        n += 1
        # print(lookup_value)
        query = """select *  from Tbl_page_statistic where wiki_page_name = """+'"'+ lookup_value +'"'
        # save how many records has been returned
        num_of_records = cur.execute(query)
        # if there is no match, preduce nulls
        if num_of_records == 0:
            # print('no match!')
            values_without_results +=1
            statistic_values = [0] * len(statistic_features_dict.keys())
        else:
            # extract the statistic
            statistic_values = list(cur.fetchall()[0])
            statistic_values = statistic_values[3:]

        # add to dictionary
        for col, val in zip(statistic_features_dict,statistic_values):
            statistic_features_dict[col].append(val)

    # save the new records into df
    statistic_features_df = pd.DataFrame.from_dict(statistic_features_dict)
    # expend to full dataset with duplicated values
    statistic_features_df['lookup'] = unique_lookups

    statistic_features_df_full = \
        dataset[['lookup']].reset_index().merge(statistic_features_df, on='lookup', sort=False). \
            sort_values('index').drop(['lookup', 'index'], 1).reset_index(drop=True)

    if want_ratio_of_nulls == True: return statistic_features_df_full, [0]*statistic_features_df_full.shape[1]
    return statistic_features_df_full


def candidate_features_extraction(dataset_after_matching,cur=cur):
    """recives dataset after matching to google entities and DB object, returns external features features """
    categories_fearures_df, categ_ratio_on_nulls = add_categories_feat(dataset_after_matching, want_ratio_of_nulls=True)
    statistic_features_df, stat_ratio_on_nulls = add_statistic_feat(dataset_after_matching, want_ratio_of_nulls=True)
    properties_fearures_df, prop_ratio_on_nulls_top_r = add_properties_feat(dataset_after_matching, want_ratio_of_nulls=True)
    ratio_of_nulls = stat_ratio_on_nulls + categ_ratio_on_nulls + prop_ratio_on_nulls_top_r
    candidate_features = statistic_features_df.join(categories_fearures_df).join(properties_fearures_df)

    return candidate_features, categories_fearures_df,statistic_features_df,properties_fearures_df, ratio_of_nulls


