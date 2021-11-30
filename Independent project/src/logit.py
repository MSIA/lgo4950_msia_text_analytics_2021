import numpy as np
import pandas as pd
import regex as re
import json
import sklearn
import eli5

from nltk.tokenize import word_tokenize
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


def convert_json_review_to_pd(json_path):
    ''' Takes jason path, reads in all entries, then returns pd dataframe object'''

    # Read in json file
    reviews = []
    with open(json_path) as json_file:
        for rec in json_file:
            dic = json.loads(rec)
            reviews.append(dic)

    # Return pd dataframe object
    reviews_pd = pd.DataFrame.from_records(reviews)

    return reviews_pd


def prepare_text(text):
    '''Cleaning and tokenizing text for analysis'''

    text = str(text)

    # Replace contractions with full text form
    text = text.split()
    new_text = []
    for word in text:
        if word in contractions:
            new_text.append(contractions[word])
        else:
            new_text.append(word)
    text = " ".join(new_text)

    # Removal of white space
    text = re.sub('\s+', ' ', text)
    # Removal of digits
    text = re.sub(r'\d+', '', text)

    # Convert words to lower case
    text = text.lower()

    # Remove other signs including punctuation
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'\'', ' ', text)

    return text


def preprocess_no_shuffle(df):
    ''' Takes raw pd dataframe object, keeps only rating and star review, then balances set for equal balance
    between 1/2 star reviews and 4/5 star reviews '''

    # Select only subset of columns
    df = df[['overall', 'reviewText']]

    # Remove average ratings (3 stars)
    df = df[df['overall'] != 3]

    # Group poor and strong ratings together (1 and 2 vs 4 and 5)
    df['overall'] = np.where(df['overall'] <= 2, 1, 5)

    # Split low and high scores
    low_scores = df[df['overall'] == 1].reset_index()
    high_scores = df[df['overall'] == 5].reset_index()

    # Select number of rows such that dataset is balanced
    min_rows = min(low_scores.shape[0], high_scores.shape[0]) - 1
    low_scores = low_scores
    high_scores = high_scores

    # Re-merge low and high scores
    df = low_scores.append(high_scores)

    # Complete pre-processing step
    df['clean_text'] = list(map(prepare_text, df['reviewText']))
    df = df.drop(['index'], axis=1)

    return df


def preprocess(df):
    ''' Takes raw pd dataframe object, keeps only rating and star review, then balances set for equal balance
    between 1/2 star reviews and 4/5 star reviews '''

    # Select only subset of columns
    df = df[['overall', 'reviewText']]

    # Remove average ratings (3 stars)
    df = df[df['overall'] != 3]

    # Group poor and strong ratings together (1 and 2 vs 4 and 5)
    df['overall'] = np.where(df['overall'] <= 2, 1, 5)

    # Split low and high scores
    low_scores = df[df['overall'] == 1].reset_index()
    high_scores = df[df['overall'] == 5].reset_index()

    # Select number of rows such that dataset is balanced
    min_rows = min(low_scores.shape[0], high_scores.shape[0]) - 1
    low_scores = low_scores[0:min_rows]
    high_scores = high_scores[0:min_rows]

    # Re-merge low and high scores
    df = low_scores.append(high_scores)

    # Shuffle resulting df
    df = shuffle(df).reset_index()

    # Complete pre-processing step
    df['clean_text'] = list(map(prepare_text, df['reviewText']))

    return df


contractions = {"ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
                "'cause": "because", "could've": "could have", "couldn't": "could not", "couldn't've": "could not have",
                "didn't": "did not", "doesn't": "does not", "don't": "do not",
                "hadn't": "had not", "hadn't've": "had not have", "hasn't": "has not",
                "haven't": "have not", "he'd": "he would", "he'd've": "he would have",
                "he'll": "he will", "he's": "he is", "how'd": "how did", "how'll": "how will",
                "how's": "how is", "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
                "isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is", "let's": "let us",
                "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                "must've": "must have", "mustn't": "must not", "needn't": "need not", "oughtn't": "ought not",
                "shan't": "shall not", "sha'n't": "shall not", "she'd": "she would", "she'll": "she will",
                "she's": "she is", "should've": "should have", "shouldn't": "should not", "that'd": "that would",
                "that's": "that is", "there'd": "there had", "there's": "there is", "they'd": "they would",
                "they'll": "they will", "they're": "they are", "they've": "they have", "wasn't": "was not",
                "we'd": "we would", "we'll": "we will", "we're": "we are", "we've": "we have",
                "weren't": "were not", "what'll": "what will", "what're": "what are", "what's": "what is",
                "what've": "what have", "where'd": "where did", "where's": "where is", "who'll": "who will",
                "who's": "who is", "won't": "will not", "wouldn't": "would not", "you'd": "you would",
                "you'll": "you will", "you're": "you are"}


def run_logistic_reg(df, product_category_name):
    '''Takes in pd df, then creates embeddings, trains logistic regression and returns most important parameters '''

    # Split train and test set
    training_data, test_data = train_test_split(df, train_size=0.8, random_state=123)
    te_y = test_data['overall']

    # Creating unigram + bigram embeddings
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 1),
                                 min_df=3, lowercase=True, max_features=100000)
    bow_representation = vectorizer.fit_transform(training_data['clean_text'])
    bow_representation_test = vectorizer.transform(test_data['clean_text'])

    best_logit = LogisticRegression(C=1, solver='liblinear',
                                    penalty='l1', max_iter=1000).fit(bow_representation, training_data['overall'])

    # predict
    y_test_pred = best_logit.predict(bow_representation_test)

    # Evaluate model
    c_matrix_test = confusion_matrix(te_y, y_test_pred)
    # Accuracy
    acc = np.round(sklearn.metrics.accuracy_score(te_y, y_test_pred), 5)
    # Precision
    prec = np.round(sklearn.metrics.precision_score(te_y, y_test_pred, average=None), 3)
    prec_micro = np.round(sklearn.metrics.precision_score(te_y, y_test_pred, average='micro'), 5)
    # Recall
    rec = np.round(sklearn.metrics.recall_score(te_y, y_test_pred, average=None), 3)
    rec_micro = np.round(sklearn.metrics.recall_score(te_y, y_test_pred, average='micro'), 5)
    # F1
    f1 = np.round(sklearn.metrics.f1_score(te_y, y_test_pred, average=None), 3)
    f1_micro = np.round(sklearn.metrics.f1_score(te_y, y_test_pred, average='micro'), 5)

    # Print model results
    print('Acc: ', acc, ' Prec: ', prec, ' Rec: ', rec, ' f1: ', f1)

    # Extract vector names
    # feature_names = vectorizer.get_feature_names_out()
    feature_names = vectorizer.get_feature_names()

    # Create summary pd
    top_x, top_y = 50, 50
    weights = eli5.show_weights(estimator=best_logit, top=(top_x, top_y),
                                target_names=training_data['overall'])
    result = pd.read_html(weights.data)[0]
    result = result.drop([top_x, (top_x + 1)], axis=0)
    result['feature_number'] = list(map(lambda x: int(x[1:]), result.Feature))
    result['feature_name'] = list(map(lambda x: feature_names[x], result.feature_number))
    result['weight_num'] = list(
        map(lambda x: np.where(x[0] == "+", float(x[1:]), float(x[1:]) * -1), result['Weight?']))
    result['category'] = product_category_name

    return result


def generate_results_for_category(category_name, path_to_data, product_file_path):
    ''' Reads data, preprocesses it, runs logistic regression classification model, returns results'''

    print('- Generating results for: ' + category_name)

    # Read in data
    df = convert_json_review_to_pd(path_to_data + product_file_path)

    # Preprocess
    df = preprocess(df)
    df['tokenized_words'] = list(map(word_tokenize, df['clean_text']))
    df['number_words'] = list(map(lambda n: len(n), df['tokenized_words']))

    # Print dataset stats
    print('Number of reviews: ' + str(df.shape[0]))
    print('Total words: ' + str(df['number_words'].sum()))

    # Run logistic regression
    reg_results = run_logistic_reg(df, category_name)
    del df
    return reg_results