import numpy as np
import pandas as pd
import json
import sklearn
import eli5
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


def visualize_gender_references_frequency():
    ''' Returns matplotlib plot summarizing frequency of gender mentions '''

    # set the figure size
    plt.figure(figsize=(12, 6))

    # Extract percentage
    percentage_male_female = (summary_gender.pct_male + summary_gender.pct_female) / 100
    df_perc = pd.DataFrame({'stars': ['1 star', '2 stars', '3 stars', '4 stars', '5 stars'],
                            'Gender references': round(percentage_male_female * 100, 3)})
    df_total = pd.DataFrame({'stars': ['1 star', '2 stars', '3 stars', '4 stars', '5 stars'],
                             'Gender references': [100, 100, 100, 100, 100]})

    # Create 2 bars
    bar1 = sns.barplot(x="stars", y="Gender references", data=df_total, color='Gray')
    bar2 = sns.barplot(x="stars", y="Gender references", data=df_perc, color='Black')

    # add legend
    top_bar = mpatches.Patch(color='Gray', label='Gender reference = No')
    bottom_bar = mpatches.Patch(color='Black', label='Gender reference = Yes')
    plt.legend(handles=[top_bar, bottom_bar])

    # show the graph
    return plt.show()


def visualize_gender_references_split():
    ''' Returns matplotlib plot summarizing split of genders '''

    # set the figure size
    plt.figure(figsize=(12, 6))

    # Extract percentage
    percentage_male = summary_gender.pct_male / (summary_gender.pct_male + summary_gender.pct_female)
    df_male = pd.DataFrame({'stars': ['1 star', '2 stars', '3 stars', '4 stars', '5 stars'],
                            'Gender': round(percentage_male * 100, 3)})
    df_total = pd.DataFrame({'stars': ['1 star', '2 stars', '3 stars', '4 stars', '5 stars'],
                             'Gender': [100, 100, 100, 100, 100]})

    # Create 2 bars
    bar1 = sns.barplot(x="stars", y="Gender", data=df_total, color='Pink')
    bar2 = sns.barplot(x="stars", y="Gender", data=df_male, color='Blue')

    # add legend
    top_bar = mpatches.Patch(color='Pink', label='Female')
    bottom_bar = mpatches.Patch(color='Blue', label='Male')
    plt.legend(handles=[top_bar, bottom_bar])

    # show the graph
    return plt.show()


def run_logistic_reg(df):
    '''Takes in pd df, then creates embeddings, trains logistic regression and returns most important parameters '''

    # Split train and test set
    training_data, test_data = train_test_split(df, train_size=0.8, random_state=123)
    te_y = test_data['stars']

    # Creating unigram + bigram embeddings
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2),
                                 min_df=3, lowercase=True, max_features=100000)
    bow_representation = vectorizer.fit_transform(training_data['clean_text'])
    bow_representation_test = vectorizer.transform(test_data['clean_text'])

    best_logit = LogisticRegression(C=1, solver='liblinear',
                                    penalty='l1', max_iter=1000).fit(bow_representation, training_data['stars'])

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
                                target_names=training_data['stars'])
    result = pd.read_html(weights.data)[0]
    result = result.drop([top_x, (top_x + 1)], axis=0)
    result['feature_number'] = list(map(lambda x: int(x[1:]), result.Feature))
    result['feature_name'] = list(map(lambda x: feature_names[x], result.feature_number))
    result['weight_num'] = list(
        map(lambda x: np.where(x[0] == "+", float(x[1:]), float(x[1:]) * -1), result['Weight?']))

    return result


def visualize_feature_importance(df, review_index):
    ''' Returns word by word importance for one given review'''
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2),
                                 min_df=3, lowercase=True, max_features=100000)
    bow_representation = vectorizer.fit_transform(df['clean_text'])
    bow_representation_test = vectorizer.transform(df['clean_text'])

    best_logit = LogisticRegression(C=1, solver='liblinear',
                                    penalty='l1', max_iter=1000).fit(bow_representation,
                                                                     df['stars'])

    class_names = {1: '1_star', 5: '5_star'}
    LIME_explainer = LimeTextExplainer(class_names=class_names)
    c = make_pipeline(vectorizer, best_logit)

    LIME_exp = LIME_explainer.explain_instance(female_df.text[review_index], c.predict_proba)
    # print results
    print('Document id: %d' % review_index)
    print('Review: ', female_df.text[review_index])
    print('Probability 5 star =', c.predict_proba([female_df.text[review_index]]).round(3)[0, 1])
    print('True class: %s' % class_names.get(female_df.stars[review_index]))

    return LIME_exp.show_in_notebook(text=True)


if __name__ == "__main__":

    # Read in reviews
    reviews = []
    with open('/Users/louisgenereux/Desktop/Term 4/Text_analytics/yelp_dataset/' \
              'yelp_academic_dataset_review.json') as json_file:
        for rec in json_file:
            dic = json.loads(rec)
            reviews.append(dic)
    print("- JSON format review data has been read")

    # Convert to pd
    reviews_df = pd.DataFrame.from_records(reviews)
    print("- Data converted to pd format")

    # Produce summary of df
    label = reviews_df['stars'].value_counts().index
    observed = reviews_df['stars'].value_counts()
    pct = reviews_df['stars'].value_counts() / len(reviews_df['stars'])
    summary = pd.DataFrame({'stars': label,
                            'observed': observed,
                            'pct': round(pct, 4) * 100})
    summary = summary.sort_values(['stars'])

    # Reading in pre-processed gendered df
    gender_mentionned = pd.read_csv('yelp_gendered.csv')
    gender_mentionned_unique = gender_mentionned[(gender_mentionned['male_present'] +
                                                  gender_mentionned['female_present']) == 1]
    print("- Gendered CSV read, entries referencing both genders are removed")

    # Producing summary
    summary_gender = gender_mentionned_unique.groupby(['stars']).agg({
                                               'male_present': [('sum')],
                                                'female_present': [('sum')]})
    summary_gender['pct_male'] = round(100*summary_gender['male_present']['sum']/
                                        list(summary['observed'].values),3)
    summary_gender['pct_female'] = round(100*summary_gender['female_present']['sum']/
                                          list(summary['observed'].values),3)
    print("- Summary of corpus: ")
    print(summary_gender)

    # Produce visuals of dataset stats
    visualize_gender_references_frequency()
    visualize_gender_references_split()

    # Create male and female only corpa
    male_df = gender_mentionned_unique[(gender_mentionned_unique['male_present'] == 1) &
                                       (gender_mentionned_unique['stars'].isin([1, 5]))]
    female_df = gender_mentionned_unique[(gender_mentionned_unique['female_present'] == 1) &
                                         (gender_mentionned_unique['stars'].isin([1, 5]))]
    print("- Male only and female only gendered corpa created")

    # Identify top word predictors of star ratings in gendered corpus
    print("- Running LOGISTIC REGRESSION")
    print("- Testing result for men:")
    result_men = run_logistic_reg(male_df)
    print("- Testing result for women:")
    result_women = run_logistic_reg(female_df)
    print('- Top predictors of high ratings male')
    print(result_men[0:10])
    print('- Top predictors of low ratings female')
    print(result_women[90:100])

    # Extracting positive and negative words in the male and female situation
    results_pos_men = list(result_men[0:50].feature_name.values)
    results_pos_women = list(result_women[0:50].feature_name.values)
    results_neg_men = list(result_men[51:100].feature_name.values)
    results_neg_women = list(result_women[51:100].feature_name.values)

    # Identifying which words appear only in subset
    men_only_pos = list(set(results_pos_men) - set(results_pos_women))
    women_only_pos = list(set(results_pos_women) - set(results_pos_men))
    men_only_neg = list(set(results_neg_men) - set(results_neg_women))
    women_only_neg = list(set(results_neg_women) - set(results_neg_men))

    # Print unique words
    print('- Words that are POSITIVE predictors of strong ratings when MEN are mentioned only:')
    print(men_only_pos)
    print('- Words that are POSITIVE predictors of strong ratings when WOMEN are mentioned only:')
    print(women_only_pos)
    print('- Words that are NEGATIVE predictors of strong ratings when MEN are mentioned only:')
    print(men_only_neg)
    print('- Words that are NEGATIVE predictors of strong ratings when WOMEN are mentioned only:')
    print(women_only_neg)

    # Which data entries have the word attitude
    female_df['attitude'] = list(map(lambda x: np.where(('attitude' in x), 1, 0), female_df.text))

    # Visualize feature importance
    visualize_feature_importance(female_df, 762)



