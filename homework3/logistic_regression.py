import pandas as pd
import numpy as np
import re
import datetime
import sklearn
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

# A list of contractions from
# http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
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


def prepare_text(text):
    '''Cleaning and tokenizing text for analysis'''

    # Convert words to lower case
    text = text.lower()

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
    # Remove other signs including punctuation
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'\'', ' ', text)

    return text


def create_bow_representations(train_clean, test_clean, ngram_rep, min_obs, top_features):
    ''' Creates unigram representations from column of word lists'''

    text_transformer = TfidfVectorizer(stop_words='english', ngram_range=ngram_rep,
                                       min_df=min_obs, lowercase=True, max_features=top_features)
    bow_rep_train = text_transformer.fit_transform(train_clean)
    bow_rep_test = text_transformer.transform(test_clean)

    return bow_rep_train, bow_rep_test


def evaluate_logistic_regression(tr_x, tr_y, te_x, te_y,
                                 solver_name, penalty_type, reg_parameter):
    ''' times the running of a logistic regression and computes performance metrics'''

    # Start timer
    start = datetime.datetime.now()

    # train model
    model = LogisticRegression(C=reg_parameter, solver=solver_name,
                               penalty=penalty_type, max_iter=1000).fit(tr_x, tr_y)

    # predict
    y_test_pred = model.predict(te_x)
    y_train_pred = model.predict(tr_x)

    # produce confusion matrix (training set & test set)
    c_matrix_test = confusion_matrix(te_y, y_test_pred)
    c_matrix_train = confusion_matrix(tr_y, y_train_pred)

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

    # Stop timer
    finish = datetime.datetime.now()

    # Compute time for operation
    total_time = np.round((finish - start).total_seconds(), 3)

    print('Time: ', total_time, ' Acc: ', acc)

    return total_time, acc, prec, prec_micro, rec, rec_micro, f1, f1_micro


def run_logistic_regression():
    ''' Test different parameter combinations and returns summary table'''

    # Creating different parameter combinations
    dataset_used = ['uni', 'uni_bi']
    regularization_parameter = [0.1, 1, 10]
    penalty_used = ['l1', 'l2']

    # Creating empty lists to store results
    bow_representation, penalty, regularization, \
    total_time, acc, prec, prec_micro, \
    rec, rec_micro, f1, f1_micro = [], [], [], [], [], [], [], [], [], [], []

    # Testing every model combination
    for data in dataset_used:
        for reg_par in regularization_parameter:
            for pen in penalty_used:
                if data == 'uni':
                    total_time_iter, acc_iter, \
                    prec_iter, prec_micro_iter, \
                    rec_iter, rec_micro_iter, \
                    f1_iter, f1_micro_iter = evaluate_logistic_regression(x_train_bow,
                                                                          y_train, x_test_bow,
                                                                          y_test, 'liblinear',
                                                                          pen, reg_par)
                else:
                    total_time_iter, acc_iter, \
                    prec_iter, prec_micro_iter, \
                    rec_iter, rec_micro_iter, \
                    f1_iter, f1_micro_iter = evaluate_logistic_regression(x_train_bow_uni_bi,
                                                                          y_train, x_test_bow_uni_bi,
                                                                          y_test, 'liblinear',
                                                                          pen, reg_par)

                bow_representation.append(data)
                penalty.append(pen)
                regularization.append(reg_par)
                total_time.append(total_time_iter)
                acc.append(acc_iter)
                prec.append(prec_iter)
                prec_micro.append(prec_micro_iter)
                rec.append(rec_iter)
                rec_micro.append(rec_micro_iter)
                f1.append(f1_iter)
                f1_micro.append(f1_micro_iter)

    summary = pd.DataFrame({'BOW representation': bow_representation,
                            'penalty': penalty,
                            'regularization': regularization,
                            'total_time': total_time,
                            'accuracy': acc,
                            'precision': prec,
                            'micro precision': prec_micro,
                            'recall': rec,
                            'micro recall': rec_micro,
                            'F1': f1,
                            'micro F1': f1_micro})

    return summary


if __name__ == '__main__':
    # Reading in data
    final_subset = pd.read_csv('balanced_df.csv')[['stars', 'text']]
    print("- Balanced data loaded with (%d rows)" % (final_subset.shape[0]))

    # Preprocessing text
    final_subset['clean_text'] = list(map(prepare_text, final_subset.text))
    print("- Text has been preprocessed")

    # Creating test and training sets
    training_data, test_data = train_test_split(final_subset, train_size=0.8, random_state=123)
    y_train = training_data['stars']
    y_test = test_data['stars']
    print("- Training set (%d rows) and Test set (%d rows) split" % (training_data.shape[0], test_data.shape[0]))

    # Creating bag of word representations
    x_train_bow, x_test_bow = create_bow_representations(training_data['clean_text'], test_data['clean_text'],
                                                         (1, 1), 3, 100000)
    print("- 1gram bag-of-word representations created")
    x_train_bow_uni_bi, x_test_bow_uni_bi = create_bow_representations(training_data['clean_text'],
                                                                       test_data['clean_text'],
                                                                       (1, 2), 3, 100000)
    print("- 1gram+2gram bag-of-word representations created")

    # Running logistic regression
    print("- Ready for LOGISTIC REGRESSION")
    results_summary = run_logistic_regression()
    print("- Logistic regression results:")
    print(results_summary[['BOW representation', 'penalty', 'regularization', 'total_time', 'accuracy',
                           'precision', 'recall', 'F1']])
    results_summary.to_csv('results_logistic_regression_experiment.csv')
    print("- Logistic regression results stored in csv")

    # Training and saving best model to pickle
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2),
                                 min_df=3, lowercase=True, max_features=100000)
    bow_representation = vectorizer.fit_transform(final_subset['clean_text'])
    best_logit = LogisticRegression(C=1, solver='liblinear',
                                    penalty='l1', max_iter=1000).fit(bow_representation, final_subset['stars'])
    # Saving model
    with open('model_logistic_reg.pickle', 'wb') as f:
        pickle.dump(best_logit, f)
    print("- Best logit model saved in Pickle")

