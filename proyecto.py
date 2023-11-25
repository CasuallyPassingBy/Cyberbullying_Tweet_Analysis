# El pequeño muro de imports
import streamlit as st
import pandas as pd
from nltk import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.util import ngrams
import numpy as np
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re

lemmatizer = WordNetLemmatizer()
stop_words_en = list(stopwords.words("English")) + ['rt']

def load_database():
    db = pd.read_csv('c:/Users/kbtmo/OneDrive/Documentos/programming_projects/Pelene/Proyecto_final/cyberbullying_tweets_procesados.csv')  # Replace with your database loading logic
    return db


# streamlit black magic ()
@st.cache_data()
def cached_database():
    return load_database()

df = cached_database()
# df['cyberbullying_type'].unique() = ['not_cyberbullying', 'gender', 'religion', 'other_cyberbullying','age', 'ethnicity']

def unstringifies_your_vector(x):
    np.fromstring(x.replace('\n','').replace('[','').replace(']','').replace('  ',' '), sep=' ')

def preprocesamiento(text:str):
    text = text.lower()
    text = re.sub(r'(@\w+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(b[\'\"]rt )', '', text)
    text = word_tokenize(text)
    text = [token for token in text if token not in stop_words_en]
    text = [lemmatizer.lemmatize(token) for token in text]
    text = ' '.join(text)
    return text

def get_ngrams(column, n):
    vocab = []
    for text in column:
        ngrams_1 = ngrams(word_tokenize(text), n)
        for ngram in ngrams_1:
            resultado = " ".join(ngram)
            vocab.append(resultado)
    return pd.Series(vocab).value_counts()

def wordcloud_generator(column, n):
    ngrams_out = get_ngrams(df[df["cyberbullying_type"] == column]['clean_tweet'], n)
    ngrams_dict = dict(ngrams_out)
    wordcloud = WordCloud(
        width = 600,
        height = 600,
        colormap = 'winter',
        background_color = 'black',
        min_font_size= 6, 
    ).generate_from_frequencies(ngrams_dict)
    return wordcloud

def bar_chart_generator(column, n, type, k = 15):
    ngrams_out = get_ngrams(df[df["cyberbullying_type"] == column]['clean_tweet'], n)

    # ChatGPT black magic
    fig, ax = plt.subplots()  # Create a new figure
    ngrams_out.head(k).plot.bar(ax=ax)
    ax.set_title(f'Top 15 {n}-grams for {type}')
    ax.set_xlabel('N-grams')
    ax.set_ylabel('Frequency')
    return fig # Return the Matplotlib object


# Este diccionario tiene las opciones que ve el usuario y como se relaciona con los diferentes tipos de cyberbullying
good_looking_options = {
    'Not Cyberbullying':'not_cyberbullying',
    'Gender': 'gender',
    'Religion': 'religion',
    'Other Cyberbullying':'other_cyberbullying',
    'Age':'age',
    'Ethnicity':'ethnicity' 
}

df['vector'] = df['vector'].apply(unstringifies_your_vector)

def main():
    st.write('# Analysis of Cyberbullying on Twitter')
    st.write('## Data Proprocessing')
    selected_types = st.multiselect(
        label = 'Cyberbullying categories', 
        options = good_looking_options.keys(), 
        default = 'Not Cyberbullying'
    )
    n = st.slider(label = 'n-grams', min_value = 1, max_value = 4, step = 1)
    
    if st.button('Generate Wordclouds'):
        wordclouds = [(wordcloud_generator(good_looking_options.get(type), n), type) for type in selected_types]
        for wordcloud, type in wordclouds:
            plt.figure(figsize=(5,5))
            plt.title(f'{type}')
            plt.imshow(wordcloud)
            plt.axis('off')
            st.pyplot(plt)

    if st.button('Generate Barcharts'):
        bar_charts = [bar_chart_generator(good_looking_options.get(type), n, type) for type in selected_types]
        for bar_chart in bar_charts:
            st.pyplot(bar_chart)
    
if __name__ == '__main__':
    main()