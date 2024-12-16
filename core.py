import pandas as pd
import re
import string
import random
import numpy as np

import emoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import pos_tag
from nltk import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import gensim
from gensim.models import Word2Vec



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

# MÓDULO 4

# Método basado en lexicons (Funciones necesarias)
def convert_tag(pos_tag):
    if pos_tag.startswith('JJ'):
        return wn.ADJ
    elif pos_tag.startswith('NN'):
        return wn.NOUN
    elif pos_tag.startswith('RB'):
        return wn.ADV
    elif pos_tag.startswith('VB') or pos_tag.startswith('MD'):
        return wn.VERB
    return None
def get_sentiment(word, tag):
    wn_tag = convert_tag(tag)
    if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV): 
        return []
    lemma = wn.morphy(word, pos=wn_tag)
    if lemma is None:
        return []
    synsets = wn.synsets(lemma, pos=wn_tag)
    if not synsets:
        return []
    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())
    return [swn_synset.pos_score(), swn_synset.neg_score(), swn_synset.obj_score()]

# Método basado en palabras únicas de textos de cada tipo de sentimiento
def entrenamiento(set_post, set_sentiment):
    positive_vocab = []
    negative_vocab = []
    for i in set_post.index:
        post = set_post[i]
        if set_sentiment[i] == 'POSITIVE':
            positive_vocab.extend(post)
        elif set_sentiment[i] == 'NEGATIVE':
            negative_vocab.extend(post)
    # Construcción de la distribución de frecuencia
    fdist_pos = FreqDist(positive_vocab)
    fdist_neg = FreqDist(negative_vocab)
    
    return fdist_pos, fdist_neg
# Obtener el set de entrenamiento y set de test para los posts y sus etiquetas.
X_train, y_train, X_test, y_test = train_test_split(df['post'].fillna(''), df['sentiment'].fillna(''), stratify=df['sentiment'].fillna(''), test_size=.3, random_state=1)
# Diccionarios de vocabulario positivo y su frecuencia y vocabulario negativo y su frecuencia
pos, neg = entrenamiento(X_train, X_test)

# Método basado en Word embeddings + algoritmo de ML de clasificación
vectores_post = [] # Lista de vectores(posts)

for post in X_train: # Recorremos cada post
    sentences = sent_tokenize(post)
    tokenized_sentences = [word_tokenize(sent.lower()) for sent in sentences]
    vectores = []

    # Obtenemos los vectores de sus palabras
    model = Word2Vec(sentences=tokenized_sentences, vector_size=50, window=5, sg=1, min_count=1, workers=4)
    # Añadimos cada vector palabra a la lista de vectores
    model.build_vocab(tokenized_sentences)  # Construye el vocabulario a partir de las oraciones
    
    # Entrenar el modelo
    model.train(tokenized_sentences, total_examples=model.corpus_count, epochs=10)
    for palabra in word_tokenize(post.lower()):
        if palabra in model.wv:
            vectores.append(model.wv[palabra])
    # Obtenemos el promedio (vector post)
    if len(vectores) > 0:
        vector_promedio = np.mean(vectores, axis=0)
    else:
        vector_promedio = np.zeros(model.vector_size)
    # Añadimos el vector del post analizado
    vectores_post.append(vector_promedio)
# Conversión de las etiquetas de sentimiento a valor numérico
y = X_test.apply(lambda x: 1 if x=='POSITIVE' else 0)
# Creación del modelo y entrenamiento
clf=RandomForestClassifier(n_estimators=10)
clf.fit(vectores_post, y)

# Función final
def sentiment_analysis(text:str):
    # Basado en lexicons
    pos_tags = pos_tag(word_tokenize(text))
    pos_score = neg_score = obj_score = 0
    for w, t in pos_tags:
        sentiment = get_sentiment(w, t)
        if sentiment:
            pos_score += sentiment[0]
            neg_score += sentiment[1]
            obj_score += sentiment[2]
    resul_metodo1 = pos_score, neg_score, obj_score
    
    # Palabras únicas en textos de ambos sentimientos
    resul_metodo2 = ''
    pos_score = 0
    neg_score = 0
    for palabra in text:
        pos_score += pos[palabra]
        neg_score += neg[palabra]
    if pos_score > neg_score:
        resul_metodo2 = 'POSITIVE'
    else:
        resul_metodo2 = 'NEGATIVE'

    # Word embeddings
    # Conversión del texto a vector
    sentences = sent_tokenize(text)
    tokenized_sentences = [word_tokenize(sent.lower()) for sent in sentences]
    vectores = []
    # Obtenemos los vectores de sus palabras
    model = Word2Vec(sentences=tokenized_sentences, vector_size=50, window=5, sg=1, min_count=1, workers=4)
    # Añadimos cada vector palabra a la lista de vectores
    for palabra in word_tokenize(post.lower()):
        if palabra in model.wv:
            vectores.append(model.wv[palabra])
    # Obtenemos el promedio (vector post)
    if len(vectores) > 0:
        vector_promedio = np.mean(vectores, axis=0)
    else:
        vector_promedio = np.zeros(model.vector_size)

    # Aplicación del vector promedio obtenido al modelo clf
    resul_metodo3 = clf.predict([vector_promedio])
    if resul_metodo3[0] == '0':
        resul_metodo3 = 'NEGATIVE'
    else:
        resul_metodo3 = 'POSITIVE'

    print(resul_metodo1)
    print(resul_metodo2)
    print(resul_metodo3)


# MÓDULO 5
def post_summarisation(text:str):
    """Construcción del diccionario palabras-frecuencia"""
    # Limpieza del texto
    text_limpio = re.sub(r'\[[0-9]*\]', ' ', text)
    text_limpio = re.sub(r'\s+', ' ', text_limpio)
    text_limpio = text_limpio.lower()
    text_limpio = re.sub(r'\d', ' ', text_limpio)
    text_limpio = re.sub(r'\s+', ' ', text_limpio)
    text_limpio = re.sub(r'[^\w\s]', '', text_limpio)
    
    # Creación del diccionario (no tendremos en cuenta las stopwords)
    vectorizador = CountVectorizer(stop_words=stopwords.words('english'))
    frecuencias = vectorizador.fit_transform([text_limpio]) 
    vocabulario = list(vectorizador.vocabulary_.keys()) # Obtenemos el vocabulario
    diccionario_palabras = dict(zip(vocabulario, frecuencias.sum(axis=0).A1))

    """Construcción del diccionario con el valor de cada oración"""
    oraciones = sent_tokenize(text) # Segmentar el texto en oraciones
    diccionario_oraciones = {} # Creación del diccionario vacío
    for oracion in oraciones:
        for palabra, freq in diccionario_palabras.items():
            if palabra in oracion.lower():
                if oracion in diccionario_oraciones.keys():
                    diccionario_oraciones[oracion] += freq
                else:
                    diccionario_oraciones[oracion] = freq
    
    """Obtenemos un cierto número de esas oraciones una vez ordenadas según su valor.
       El número de oraciones será aleatorio entre 1 y el número de oraciones total."""
    oraciones_ordenadas = sorted(diccionario_oraciones, key=diccionario_oraciones.get, reverse=True)
    oraciones_resumen = oraciones_ordenadas[:random.randint(1, len(oraciones))]
    resumen = ' '.join(oraciones_resumen) # Devolvemos en forma de texto

    """Evaluación de la similitud entre el texto y el resumen generado"""
    tfidf = TfidfVectorizer()
    matriz = tfidf.fit_transform([text, resumen])
    similitud = cosine_similarity(matriz)
    return resumen, similitud

# MÓDULO 6
def texts_distance(text1:str, text2:str):
    # Conversión de los textos a vectores
    vectorizador = TfidfVectorizer() # Creamos una instancia del vectorizador
    vector1, vector2 = vectorizador.fit_transform([text1, text2])

    # Conversión de los vectores a matrices Numpy (necesario para la implementación de la función que calcula la distancia)
    array1 = vector1.toarray().ravel()
    array2 = vector2.toarray().ravel()

    # Cálculo de la distancia
    distancia_coseno = cosine(array1, array2)

    return 'La distancia entre los dos textos indicados es: ', distancia_coseno
