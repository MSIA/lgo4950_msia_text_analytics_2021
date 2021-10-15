import os
import pandas as pd
import re
import string

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import gensim


def preprocessing_corpus_2(corpus):

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
        results_df[word] = [result]

    results_df.index = [str(index_model_name)]

    return results_df


if __name__ == '__main__':

    # Identify all paths to files for aggregation
    files_list = []
    path = os.getcwd() + '/20news-bydate-train/'
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            files_list.append(os.path.join(root, name))
    print('- File paths to all corpus elements collected')

    # Iteratively append news stories to a list (holding the full corpus
    corpus_2_list = []
    for i in files_list:
        with open(i, 'r', encoding="utf8", errors="ignore") as f:
            file = f.read()
            corpus_2_list.append(file)
    print('- All elements of corpus added to list')

    # Create a text file containing all the cleaned corpus
    textfile = open("corpus_2_text_file_final.txt", "w")
    for news_story in corpus_2_list:
        tokenized_story = preprocessing_corpus_2(news_story)
        for sentence in tokenized_story:
            textfile.write(sentence + "\n" + "\n")
    textfile.close()
    print('- All corpus preprocessed and added to text file')

    # Create a list of all words (for word2vec modeling)
    word_tokens_list_of_list = []
    for news_story in corpus_2_list:
        tokenized_story = preprocessing_corpus_2(news_story)
        for sentence in tokenized_story:
            word_tokens_list = word_tokenize(sentence)
            word_tokens_list_of_list.append(word_tokens_list)
    print('- All words of corpus added to list of list')
    print('- READY FOR WORD2VEC MODELING!')

    # Create Skipgram model - with different vector and window parameters
    model_sg_50_3 = build_word2vec(word_tokens_list_of_list, 50, 3, 5, 1, "model_sg_50_3")
    model_sg_50_5 = build_word2vec(word_tokens_list_of_list, 50, 5, 5, 1, "model_sg_50_5")
    model_sg_50_7 = build_word2vec(word_tokens_list_of_list, 50, 7, 5, 1, "model_sg_50_7")
    model_sg_100_3 = build_word2vec(word_tokens_list_of_list, 100, 3, 5, 1, "model_sg_100_3")
    model_sg_100_5 = build_word2vec(word_tokens_list_of_list, 100, 5, 5, 1, "model_sg_100_5")
    model_sg_100_7 = build_word2vec(word_tokens_list_of_list, 100, 7, 5, 1, "model_sg_100_7")
    model_sg_200_3 = build_word2vec(word_tokens_list_of_list, 200, 3, 5, 1, "model_sg_200_3")
    model_sg_200_5 = build_word2vec(word_tokens_list_of_list, 200, 5, 5, 1, "model_sg_200_5")
    model_sg_200_7 = build_word2vec(word_tokens_list_of_list, 200, 7, 5, 1, "model_sg_200_7")
    print('+ Created 9 skipgram models with embeddings [50,100,200] and window size [3,5,7]')

    # Create CBOW model - with different vector and window parameters
    model_cbow_50_3 = build_word2vec(word_tokens_list_of_list, 50, 3, 5, 1, "model_cbow_50_3")
    model_cbow_50_5 = build_word2vec(word_tokens_list_of_list, 50, 5, 5, 1, "model_cbow_50_5")
    model_cbow_50_7 = build_word2vec(word_tokens_list_of_list, 50, 7, 5, 1, "model_cbow_50_7")
    model_cbow_100_3 = build_word2vec(word_tokens_list_of_list, 100, 3, 5, 1, "model_cbow_100_3")
    model_cbow_100_5 = build_word2vec(word_tokens_list_of_list, 100, 5, 5, 1, "model_cbow_100_5")
    model_cbow_100_7 = build_word2vec(word_tokens_list_of_list, 100, 7, 5, 1, "model_cbow_100_7")
    model_cbow_200_3 = build_word2vec(word_tokens_list_of_list, 200, 3, 5, 1, "model_cbow_200_3")
    model_cbow_200_5 = build_word2vec(word_tokens_list_of_list, 200, 5, 5, 1, "model_cbow_200_5")
    model_cbow_200_7 = build_word2vec(word_tokens_list_of_list, 200, 7, 5, 1, "model_cbow_200_7")
    print('+ Created 9 CBOW models with embeddings [50,100,200] and window size [3,5,7]')

    # Evaluating all models at once and creating summary table
    list_model_objects = [model_sg_50_3, model_sg_50_5, model_sg_50_7,
                          model_sg_100_3, model_sg_100_5, model_sg_100_7,
                          model_sg_200_3, model_sg_200_5, model_sg_200_7,
                          model_cbow_50_3, model_cbow_50_5, model_cbow_50_7,
                          model_cbow_100_3, model_cbow_100_5, model_cbow_100_7,
                          model_cbow_200_3, model_cbow_200_5, model_cbow_200_7]

    list_index_names = ['sg_50_3', 'sg_50_5', 'sg_50_7',
                          'sg_100_3', 'sg_100_5', 'sg_100_7',
                          'sg_200_3', 'sg_200_5', 'sg_200_7',
                         'cbow_50_3', 'cbow_50_5', 'cbow_50_7',
                          'cbow_100_3', 'cbow_100_5', 'cbow_100_7',
                          'cbow_200_3', 'cbow_200_5', 'cbow_200_7']
    evaluation_list = ['government', 'army', 'happy', 'food', 'pride',
                       'wealth', 'overwhelming', 'education', 'family', 'computer']
    agg_results = evaluate_model_nn(list_model_objects[0], list_index_names[0], evaluation_list, 3)

    # Add rows to summary table
    for i in range(1, 18):
        new_row = pd.DataFrame(evaluate_model_nn(list_model_objects[i], list_index_names[i], evaluation_list, 3))
        agg_results = agg_results.append(new_row)
    print('+ Nearest neighbors generated for 18 models')

    # Save results
    agg_results.to_csv('results_final.csv')
    print('+ NEAREST NEIGHBOR RESULTS SAVED TO CSV!')
