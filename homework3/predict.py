import pickle
import pandas as pd
import re

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


def prepare_text(text):
    '''Cleaning and tokenizing text for analysis'''

    # Convert words to lower case
    text = text.lower()

    # Removal of white space
    text = re.sub('\s+', ' ', text)
    # Removal of digits
    text = re.sub(r'\d+', '', text)
    # Remove other signs including punctuation
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'\'', ' ', text)

    # Removal of stop words
    text = text.split()
    s_word = set(stopwords.words("english"))
    text = [word for word in text if not word in s_word]
    text = " ".join(text)

    return text


if __name__ == '__main__':
    # Reading in data
    final_subset = pd.read_csv('balanced_df.csv')[['stars', 'text']]
    print("- Balanced data loaded with (%d rows)" % (final_subset.shape[0]))

    # Preprocessing text
    final_subset['clean_text'] = list(map(prepare_text, final_subset.text))
    print("- Text has been preprocessed")

    # Loading model
    loaded_model = pickle.load(open('model_logistic_reg.pickle', 'rb'))
    print("- Model loaded from Pickle")

    # Creating bow represenations
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2),
                                 min_df=3, lowercase=True, max_features=100000)
    bow_representation = vectorizer.fit_transform(final_subset['clean_text'])
    print("- BOW representations created")

    # Make predictions
    predicted = loaded_model.predict(bow_representation)
    predicted_prob = loaded_model.predict_proba(bow_representation)
    results = pd.DataFrame({'Text': final_subset['text'],
                            'Actual': final_subset['stars'],
                            'Predicted': predicted,
                            'Probability': list(predicted_prob)})
    print("- Forecasts created")

    # Saving to json format
    results[0:1000].to_json("predicted__results.json")
    print("- Forecasts saved to JSON format")