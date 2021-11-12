import os
import pandas as pd
import re
import string

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import gensim


def preprocessing_corpus(corpus):

    '''
    Takes string from news corpus, normalizes and tokenizes into sentences
    '''

    # Removal of white space
    step_1 = re.sub('\s+', ' ', corpus)
    # RFrom/To and subject line of email
    step_2 = re.sub(r'\bFrom: .*? writes: ', '', step_1)
    # Removal of digits
    step_3 = re.sub(r'\d+', '', step_2)
    # Remove other signs
    step_4 = re.sub(r'["_“”\'\`\-\*\(\)]', '', step_3)
    # Tokenize to sentences while punctuation is still in place
    tokens_news = sent_tokenize(step_4)
    # Convert each token to lowercase
    lower_token = list(map(lambda token: token.lower(), tokens_news))
    # Remove punctuation from lowercase token
    punct_less_token = list(map(lambda token:
                                token.translate(str.maketrans('', '', string.punctuation)), lower_token))

    return punct_less_token


def build_word2vec(list_of_list, dimension_size, window_size, min_obs, model_type, model_name):

    """
    Creates a model object
    Args:
        list_of_list (list): preprocessed text corpus
        dimension_size (int): size of dimensions in model
        window_size (int): window size used for training
        min_obs (int): minimum observed instances of a word to be considered
        model_type (binary 1 or 0): 1 = skipgram, 0 = CBOW
        model_name: name of object
    Returns:
        A trained word2vec model, that is saved as an object
    """

    new_model = gensim.models.Word2Vec(list_of_list, vector_size=dimension_size, window=window_size,
                                       sg=model_type, min_count=min_obs)
    new_model.save(model_name)

    return new_model


def evaluate_model_nn(model_object, index_model_name, evaluation_word_lists, top_n):
    """
    Identifies top n nearest words for words in a lit
    Args:
        model_object (obj): word2vec model
        index_model_name (str): name of model for row indices naming
        evaluation_word_lists (list): words to be used as reference to identify nearest neighbors
        top_n (int): top n number of nearest neighbors
    Returns:
        A pd dataframe summary
    """

    # Create blank df
    results_df = pd.DataFrame()

    for i, word in enumerate(evaluation_word_lists):
        result = model_object.wv.most_similar(word, topn=top_n)
        result = list(map(lambda x: str(x[0]) + ', ' + str(round(x[1],3)), result))
        results_df[word] = [result]

    results_df.index = [str(index_model_name)]

    return results_df


if __name__ == "__main__":

    # Reading in pre-processed gendered df
    gender_mentionned = pd.read_csv('yelp_gendered.csv')
    gender_mentionned_unique = gender_mentionned[(gender_mentionned['male_present'] +
                                                  gender_mentionned['female_present']) == 1]
    print("- Gendered CSV read, entries referencing both genders are removed")

    # Subset of full dataset
    subset = gender_mentionned_unique[0:100000]

    # Create a list of all words (for word2vec modeling)
    word_tokens_list_of_list = []
    for review in subset.text:
        tokenized_review = preprocessing_corpus(review)
        for sentence in tokenized_review:
            word_tokens_list = word_tokenize(sentence)
            word_tokens_list_of_list.append(word_tokens_list)
    print('- All words of corpus added to list of list')
    print('- READY FOR WORD2VEC MODELING!')

    # Create Skipgram model
    model_sg_100_5 = build_word2vec(word_tokens_list_of_list, 100, 5, 5, 1, "model_sg_100_5")
    print('+ Created skipgram models with 100 embeddings and window size 5')

    pd.set_option('max_colwidth', 500)
    evaluation_list = ['lady', 'sales', 'owner', 'employee', 'cashier', 'driver', 'saleswoman', 'salesman']
    agg_results = evaluate_model_nn(model_sg_100_5, 'model_sg_100_5', evaluation_list, 10)
    agg_results

    # Printing results
    for i in range(0, agg_results.shape[1]):
        print('NEAREST NEIGHBORS FOR: ' + agg_results.columns[i])
        result = agg_results.iloc[0, i]
        for j in result:
            print(j)
        print('---   ---   ---   ---   ---   ---   ---   ---')