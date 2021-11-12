import re
import pandas as pd
import json
import nltk

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

import warnings

warnings.filterwarnings("ignore")


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


def stem_text(text):
    '''Takes text string and returns stemmed string'''

    # Tokenize_text
    word_tokens = nltk.word_tokenize(text)

    # Create stems
    #ps = PorterStemmer()
    stems = [ps.stem(w) for w in word_tokens]

    # Re-append to form one string
    text = ' '.join(stems)

    return text


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


def man_present_in_string(string):
    '''Checks whether string contains reference to man'''

    present = 0
    man_words = ['man', 'male', 'boy', 'guy', 'gentleman']
    woman_words = ['woman', 'female', 'girl', 'gal', 'lady']

    word_list = word_tokenize(string)

    for word in man_words:
        if word in word_list:
            present = 1

    return present


def woman_present_in_string(string):
    '''Checks whether string contains reference to woman'''

    present = 0
    man_words = ['man', 'male', 'boy', 'guy', 'gentleman']
    woman_words = ['woman', 'female', 'girl', 'gal', 'lady']

    word_list = word_tokenize(string)

    for word in woman_words:
        if word in word_list:
            present = 1

    return present


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
    reviews_df = pd.DataFrame.from_records(reviews[0:1000])
    print("- Data converted to pd format")

    # Identify presence of men and women in text review
    reviews_df['male_present'] = list(map(man_present_in_string, reviews_df.text))
    reviews_df['female_present'] = list(map(woman_present_in_string, reviews_df.text))
    print("- Gender references identified in reviews")

    # Create subset: only with reviews containing gender
    gender_mentionned = reviews_df[(reviews_df['male_present']==1) | (reviews_df['female_present']==1)]
    print("- Gendered df subset created")

    # Preprocess this text
    gender_mentionned['clean_text'] = list(map(prepare_text, gender_mentionned['text']))

    # Stem cleaned text
    ps = PorterStemmer()
    gender_mentionned['clean_text_stem'] = list(map(stem_text, gender_mentionned.clean_text))
    print("- Text pre-processed for gendered df")

    # Save text to csv
    gender_mentionned.to_csv('yelp_gendered_sample.csv')
    print("- DF saved to csv")
