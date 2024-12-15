import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import emoji

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split


# MÓDULO 1

def preprocess_post(text: str) -> str:
    '''
    input:
    - text (str): input text to be processed.    

    steps :
    1. convert the text to lowercase to standardize case.
    2. remove urls, user (@) and hashtags (#).
    3. strip html tags.
    4. remove emojis from the text.
    5. remove punctuation marks.
    6. eliminate extra whitespace.
    7. tokenize the text into individual words.
    8. remove stopwords to focus on meaningful words.
    9. apply stemming and lemmatization to reduce words to their base forms.

    output :
    - str: cleaned and normalized version of the input text.
    '''

    text = str(text)  
    # 1.
    text = text.lower()

    # 2. 
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)

    # 3.
    text = re.sub(r'<.*?>', '', text)

    # 4.
    text = emoji.replace_emoji(text, replace='')

    # 5. 
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 6. 
    text = re.sub(r'\s+', ' ', text).strip()

    # 7. 
    words = word_tokenize(text)

    # 8.
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # 9. 
    # stemmer = PorterStemmer() could be used as well, less accurate for sentiment analysis
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    text = ' '.join(words)
    return text


# df['clean_post'] = df['post'].apply(preprocess_post)
# df.to_csv("processed_dataset.csv", index=False, sep=',', quotechar='"')

# MÓDULO 2

class SubredditClassifier:
    def __init__(self, max_features=5000, ngram_range=(1, 2), max_iter=500, random_state=42):
        self.tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.model = LogisticRegression(max_iter=max_iter, random_state=random_state)

    def train(self, X_train, y_train):
        """
        Entrena el modelo utilizando TF-IDF y Logistic Regression.
        """
        X_train_tfidf = self.tfidf.fit_transform(X_train)
        self.model.fit(X_train_tfidf, y_train)

    def classify_subreddit(self, text):
        """
        Clasifica un texto en una de las categorías.
        """
        text_tfidf = self.tfidf.transform([text])
        return self.model.predict(text_tfidf)[0]

    def transform_data(self, X_data):
        """
        Vectoriza datos de entrada utilizando TF-IDF.
        """
        return self.tfidf.transform(X_data)
    

# MÓDULO 3

def find_subreddit_mentions(text: str) -> list:
    '''
    Extract all subreddit mentions in a given text.
    It searches for patterns like '/r/subreddit'.

    Input:
    - text (str): The input text to search for subreddit mentions.

    Output:
    - list: A list containing the mentioned subreddits.
    '''
    
    pattern = r'/r/\w+'
    
    matches = re.findall(pattern, text)
    
    return matches

def url_extraction(text: str) -> list:
    '''
    Extract all URLs in a given text.
    It searches for patterns that match common URL formats.

    Input:
    - text (str): The input text to search for URLs.

    Output:
    - list: A list containing all the extracted URLs.
    '''
    pattern = r'(https?://\S+|www\.\S+)'
    urls = re.findall(pattern, text)
    return urls

def phone_number_extraction(text: str) -> list:
    '''
    Extract phone numbers from a given text.
    It searches for patterns matching common phone number formats.

    Input:
    - text (str): The input text to search for phone numbers.

    Output:
    - list: A list containing all the extracted phone numbers.
    '''
    pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    phone_numbers = re.findall(pattern, text)
    return phone_numbers

def dates_extraction(text: str) -> list:
    '''
    Extract all dates from a given text.
    It searches for patterns matching common date formats.

    Input:
    - text (str): The input text to search for dates.

    Output:
    - list: A list containing all the extracted dates.
    '''
    pattern = r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s\w+\s\d{4}|\w+\s\d{1,2},\s\d{4})\b'
    dates = re.findall(pattern, text)
    return dates

def code_extraction(text: str) -> list:
    '''
    Extract code snippets from a given text
    It searches for:
    - Blocks of code delimited by triple backticks
    - Inline code delimited by single backticks
    - HTML code tags
    - Common programming patterns

    Input:
    - text (str): The input text to search for code snippets.

    Outut:
    - list: A list containing all the extracted code snippets.
    '''
    pattern = (
        r'```[\s\S]*?```|'  
        r'`[^`]+?`|' 
        r'<[^>]+>.*?</[^>]+>|'     
        r'\b(def\s+\w+\(.*?\):|'   # def()
        r'print\(.+?\)|'           # print()
        r'\w+\s*=\s*\[.*?\])\b'    #list []
    )

    code_snippets = [match.group() for match in re.finditer(pattern, text)]
    return code_snippets
