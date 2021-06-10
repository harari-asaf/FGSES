# TODO: fix dbpedia parsing : "Chevrolet_Chevy_II_/_Nova" become "Nova" becouse the /
import pandas as pd
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)
import pymysql # connect to mysql in linux
import numpy as np
from timeit import default_timer as timer
from nltk.stem import PorterStemmer
from gensim.models import KeyedVectors
import os
import tensorflow_hub as hub
import tensorflow as tf
ps = PorterStemmer()

def queryRes(cur):
    """ convert query result to df"""
    results = pd.DataFrame(list(cur.fetchall()))
    if len(results)>0:
        results.columns = [i[0] for i in cur.description]
    return results

def isEnglish(s):
    """Check if string have non english chars"""
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def import_elmo(path):
    """Load pertained elmo using tensorflow hub"""
    global elmo
    elmo = hub.load(path)

def embd_dict(sentense):
    input_text = tf.reshape(sentense, [-1])
    embeddings = elmo.signatures['default'](input_text)['default']
    embeddings = embeddings.numpy()
    return embeddings


def elmo_for_each_page(pages):
    """recives all wikipedia pages and return embedding of each"""
    # compute embeddings to each page (sum the words)
    print('compute embeddings to each page, num of pages: ', pages.shape)

    """ Compute embeddings to each wiki page"""
    np.set_printoptions(threshold=1000000)
    # preprocess each page name
    process_pages = pages['wiki_page_name'].apply(
        lambda x: process_pages_for_elmo(x, domain='')).to_list()

    start_comute_embd = timer()

    pages_embds_ls = [embd_dict([page_name]) for page_name in tqdm(process_pages)]
    pages['embd'] = pages_embds_ls
    pages['embd'] = pages['embd'].apply(lambda x: x[0])

    print('dim of the df: ', pages.shape)
    return pages


# function whicc computes cosine sim
def cos_sim(a,b):
    value = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return value

def process_pages_for_elmo(sentence, domain):
    # print('sentence: '+sentence)
    """ split sentence to tokens, compute embd and sum them
    sentence (str)
    embeddings (dict)"""
    sentence = domain + ' ' + sentence
    parse_sentence = sentence.replace('_',' ').replace('-',' ').replace('(','').replace(')','').replace('.','').strip()
    if len(parse_sentence) > 0: sentence = parse_sentence
    return sentence

def compute_embd(sentence,embeddings, domain):
    # print('sentence: '+sentence)
    """ parse sentese for elmo
    sentence (str)
    embeddings (dict)"""
    sentence = domain + ' ' + sentence
    parse_sentence = sentence.replace('_',' ').replace('-',' ').replace('(','').replace(')','').replace('.',' ').replace("  "," ").strip().title()

    if len(parse_sentence) > 0: sentence = parse_sentence
    embd = embeddings([sentence])
    return embd

def embds_to_matrix_and_norm(embd_series):
    # convert categories embeddings to matrix
    pages_embeddings = np.array(list(embd_series.apply(list)))
    # compute the norm of the categories embeddings
    # transpose matrix
    pages_embeddings_T = pages_embeddings.transpose()
    # compute norm
    # compute diag of dot product (sum the inner squers)
    embed_inner_dot = np.einsum('ij,ji->i', pages_embeddings,pages_embeddings_T)
    # sqrt
    pages_norm = np.sqrt(embed_inner_dot)
    return pages_embeddings, pages_norm # np.arrays

def term_pages_similarity(term_embd,pages_norm,pages_embeddings):
    """"Compute cosine sim componnets"""
    # compute terme norm
    term_norm = np.linalg.norm(term_embd)
    # create terme norm vec  (same value in each cell)
    """"Compute cosine sim"""
    # print('compute similarity')
    # compute dot product between term amd category (vector[num_of_categories])
    pages_term_dot_vec = np.dot(pages_embeddings,term_embd)
    # compute similarity between term embed and categories embd
    term_pages_sim_vec = pages_term_dot_vec/(term_norm*pages_norm)
    return term_pages_sim_vec # np.array

"""filter by stemmed categories"""
def stem_sentence(sentence):
    tokenaized_sentence = sentence.split('_')
    # tokenaized_sentence = word_tokenize(sentence)
    stemmed_tokenaized_sentence = [ps.stem(token) for token in tokenaized_sentence]
    stemmed_sentence = ' '.join(stemmed_tokenaized_sentence)
    return stemmed_sentence


def retrieve_wiki_entity_for_term_from_all(lookup_value,domain,all_pages,pages_abstr,domain_pages, embd_dict,domain_pages_norm, domain_pages_embeddings,cur=cur):
    '''recived lookup term and return wiki_entity from entities list'''

    # initializations
    retType = 'dirHit'
    wiki_entity = ''
    cand_wiki_entity_abstr = '-'
    lookup_value_U = lookup_value.replace('_',' ').strip().lower()
    lookup_value_U_dis = lookup_value_U+' (disambiguation)'
    # compute term embeddings
    term_embd = compute_embd(lookup_value_U, embd_dict,  domain="").flatten()

    print('lookup: '+lookup_value)
    # check if there is a direct hit in the pages list
    wiki_entity_array = all_pages.loc[all_pages['wiki_page_name_lower'] == lookup_value_U][
        'wiki_page_name'].values
    wiki_entity_dis_array = all_pages.loc[all_pages['wiki_page_name_lower'] == lookup_value_U_dis][
        'wiki_page_name'].values

    if len(wiki_entity_dis_array) == 1:
        cand_wiki_entity = wiki_entity_dis_array[0]
    elif len(wiki_entity_array) == 1:
        cand_wiki_entity = wiki_entity_array[0]

    # if there is no direct hit in the list, use embeddings
    else:
        # print('Match using embeddings')
        retType = 'embd'
        # insert to domain pages df
        domain_pages['sim_to_term'] = term_pages_similarity(term_embd, domain_pages_norm, domain_pages_embeddings)
        # return the most similar page
        cand_wiki_entity = list(domain_pages.sort_values(['sim_to_term'], ascending=False)['wiki_page_name'])[0]

    cand_wiki_entity_abstr_arry = list(pages_abstr.loc[pages_abstr['wiki_page_name'] == cand_wiki_entity]['abstract'])

    if len(cand_wiki_entity_abstr_arry) == 1: cand_wiki_entity_abstr = cand_wiki_entity_abstr_arry[0]
    else: cand_wiki_entity_abstr = 'no abstract'


    # if the chosen page is disambiguate page: retrieve  the first link
    if ('may refer to:' in cand_wiki_entity_abstr) or \
            ('disambiguation' in cand_wiki_entity) or \
            (cand_wiki_entity_abstr == 'no abstract'):

        # import wikilinks of the page
        cur.execute(
            """SELECT linked_page FROM Tbl_page_wikilinks where wiki_page_name = """ + '"' + cand_wiki_entity + '"')
        wikilinks = queryRes(cur)
        wikilinks['wiki_page_name'] = wikilinks['linked_page'].apply(lambda x: x.replace('_', ' ').strip())
        # if the dis page is redirect page move to the redirect
        if (len(wikilinks)==1) and ('disambiguation' in cand_wiki_entity):
            retType = retType+'_disambi'
            cand_wiki_entity = list(wikilinks['wiki_page_name'])[0]
            cur.execute(
                """SELECT linked_page FROM Tbl_page_wikilinks where wiki_page_name = """ + '"' + cand_wiki_entity + '"')
            wikilinks = queryRes(cur)
            wikilinks['wiki_page_name'] = wikilinks['linked_page'].apply(lambda x: x.replace('_', ' ').strip())
        # if there is only one wikilinks he is the wiki entity
        if (len(wikilinks) == 1):
            wiki_entity = list(wikilinks['wiki_page_name'])[0]
        # if there lot of wikilinks chose one of them acording to the domain embd
        elif (len(wikilinks)>1):
            wikilinks = wikilinks.loc[~wikilinks['linked_page'].str.contains('Category:|File:|Image:|Wikipedia:|Help|Template|Portal:')]
            # retrieve the first result
            wiki_entity = list(wikilinks['wiki_page_name'])[0]
        else:
            wiki_entity = cand_wiki_entity

    # if this is a regular page return its name
    else: wiki_entity = cand_wiki_entity
    print('       = '+wiki_entity+ ' '+ retType)
    return (lookup_value,wiki_entity,retType)



def entity_matching(dataset,dataset_name,cur):

    print("==================\nStart entity retrival from wikipedia \n =================")
    """Recicves connection to DB  with Wikipedia entities, dataset and dataset name. return dataset with matches entities"""
    # import Wikipedia pages with abstract
    cur.execute("""SELECT wiki_page_name FROM Tbl_page_statistic where abstract = 1""")
    pages = queryRes(cur)
    no_english_pages = np.array(pages['wiki_page_name'].apply(isEnglish))
    pages = pages.loc[no_english_pages]

    # import all wikipedia pages
    cur.execute("""SELECT wiki_page_name FROM Tbl_page_id""")
    all_pages = queryRes(cur)
    all_pages['wiki_page_name_lower'] = all_pages['wiki_page_name'].apply(str.lower)
    # import the abstract of each page
    cur.execute("""SELECT wiki_page_name,abstract FROM Tbl_page_abstract""")
    # each page
    pages_abstr = queryRes(cur)


    # import dictionary of elmo vectores
    import_elmo()
    # compute embeddings of each wikipedia page - this is done off-line
    pages = elmo_for_each_page(pages)
    # compute the norm of each page anc convert to numpy matrix - this is done off-line
    pages_embeddings, pages_norm = embds_to_matrix_and_norm(pages['embd'])

    print('==== '+ dataset_name+ ' ====')

    domain_pages = pages.copy()
    domain_pages_norm = pages_norm
    domain_pages_embeddings = pages_embeddings
    print('Start iterate over dataset lookup values and find most similar entities')
    unique_lookups = dataset['lookup'].unique()
    retrieval_results = {'lookup_value':[],'wiki_entity':[]}

    print('Number of unique lookup values in dataset: ',len(unique_lookups))
    unique_lookups_temp = unique_lookups


    res = map(lambda lookup_value: retrieve_wiki_entity_for_term_from_all(lookup_value, all_pages,pages_abstr,
                                                                    domain_pages, embd_dict,
                                                                    domain_pages_norm,
                                                                    domain_pages_embeddings),
        unique_lookups_temp)


    retrieval_entities = list(res)

    retrieval_entities_for_stats = pd.DataFrame(retrieval_entities,columns=['lookup','entity','retType'])
    retrieval_entities_for_stats['dataset'] = dataset_name
    retrieval_entities_df = retrieval_entities_for_stats.loc[:,['lookup','entity']]

    # match each record with it retrieved entity
    dataset_with_ret_entities = dataset.merge(retrieval_entities_df,on ='lookup')

    # return the dataset with wiki entities
    dataset_after_matching = dataset_with_ret_entities.drop(['entity','lookup'],1)
    dataset_after_matching['lookup'] = dataset_with_ret_entities['entity']
    return dataset_after_matching