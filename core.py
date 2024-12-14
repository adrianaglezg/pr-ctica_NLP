import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import emoji


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

