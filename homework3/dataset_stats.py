import pandas as pd
import json
import random
import string
from nltk.tokenize import word_tokenize


def group_same_score_reviews(reviews_df):
    ''' Takes JSON formatted review data, classifies each review into 5 lists, based on score '''

    # Create new subset data
    reviews_one_star = []
    reviews_two_star = []
    reviews_three_star = []
    reviews_four_star = []
    reviews_five_star = []

    # Iterate through each review, classifying it in right category
    for i in range(0, len(reviews_df)):
        if reviews_df[i]['stars'] == 1:
            reviews_one_star.append(reviews_df[i])
        elif reviews_df[i]['stars'] == 2:
            reviews_two_star.append(reviews_df[i])
        elif reviews_df[i]['stars'] == 3:
            reviews_three_star.append(reviews_df[i])
        elif reviews_df[i]['stars'] == 4:
            reviews_four_star.append(reviews_df[i])
        elif reviews_df[i]['stars'] == 5:
            reviews_five_star.append(reviews_df[i])

    return reviews_one_star, reviews_two_star, reviews_three_star, \
           reviews_four_star, reviews_five_star


def evaluate_score_distribution(rev_one, rev_two, rev_three, rev_four, rev_five):
    ''' Takes 5 lists of reviews (each with their own score), analyses frequency and creates summary '''

    # Number of each review types
    one_star = len(rev_one)
    two_star = len(rev_two)
    three_star = len(rev_three)
    four_star = len(rev_four)
    five_star = len(rev_five)
    total = len(reviews)

    # Creating summary table
    distribution = pd.DataFrame({'stars': [1, 2, 3, 4, 5, 'total'],
                                 'observations': [one_star,
                                                  two_star,
                                                  three_star,
                                                  four_star,
                                                  five_star,
                                                  total],
                                 'pct share of total': [100 * one_star / total,
                                                        100 * two_star / total,
                                                        100 * three_star / total,
                                                        100 * four_star / total,
                                                        100 * five_star / total,
                                                        100 * total / total]})

    # Print summary
    print(distribution)

    return distribution


def create_balanced_subset(rev_one, rev_two, rev_three, rev_four, rev_five, obs_count):
    '''
    Takes 5 lists of reviews (rev_one - rev_five), then produces balanced data set
    with obs_count observations from each class
     '''

    # Shortened list of all reviews
    reviews_one_star = rev_one[0:obs_count]
    reviews_two_star = rev_two[0:obs_count]
    reviews_three_star = rev_three[0:obs_count]
    reviews_four_star = rev_four[0:obs_count]
    reviews_five_star = rev_five[0:obs_count]

    # Aggregating all subsets into 1 final dataset
    reviews_subset = reviews_one_star + reviews_two_star + \
                     reviews_three_star + reviews_four_star + \
                     reviews_five_star

    # Shuffling data set so that not all categories are ordered
    random.seed(123)
    random.shuffle(reviews_subset)

    # Converting to pd format
    clean_pd = pd.DataFrame(reviews_subset)[['stars', 'text']]

    return clean_pd


def tokenize_count_length(text_string):
    ''' Takes string, removes punctuation, tokenize and count words in string '''

    punct_less_token = text_string.translate(str.maketrans('', '', string.punctuation))
    tokenized = word_tokenize(punct_less_token)
    word_count = len(tokenized)

    return word_count


if __name__ == '__main__':

    # Reading in data
    reviews = []
    with open('/Users/louisgenereux/Desktop/Term 4/Text_analytics/yelp_dataset/' \
              'yelp_academic_dataset_review.json') as json_file:
        for rec in json_file:
            dic = json.loads(rec)
            reviews.append(dic)
    print("- JSON format review data has been read")

    # Splitting all reviews in 5 categories (based on their star ratings)
    one_str, two_str, three_str, four_str, five_str = group_same_score_reviews(reviews)
    print("- All reviews categorized based on their ratings")

    # Evaluating distribution of ratings
    print("- Distribution of ratings:")
    summary_distribution = evaluate_score_distribution(one_str, two_str, three_str, four_str, five_str)

    # Creating balanced subset of data
    final_subset = create_balanced_subset(one_str, two_str, three_str, four_str, five_str, 100000)
    print("- Balanced subset of reviews created, with %d observations per class" % (final_subset.shape[0] / 5))

    # Evaluating length of each document
    final_subset['word_count'] = list(map(tokenize_count_length, final_subset.text))
    print("- Description of word counts:")
    print(final_subset[['word_count']].describe())

    # Save balanced training set as csv
    final_subset[['stars', 'text']].to_csv('balanced_df.csv')
    print("- Balanced data set saved as csv")

