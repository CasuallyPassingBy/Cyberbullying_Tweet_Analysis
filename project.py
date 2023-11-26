# Importing necessary libraries
from tensorflow.keras.models import load_model
import streamlit as st
import pandas as pd
from nltk import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.util import ngrams
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re
import plotly_express as px
import joblib
import spacy

# Loading spaCy's English language model
nlp = spacy.load('en_core_web_lg')
lemmatizer = WordNetLemmatizer()  # Initializing lemmatizer
stop_words_en = list(stopwords.words("English")) + ['rt'] # Creating a list of stopwords

path = 'C:/Users/kbtmo/OneDrive/Documentos/programming_projects/Pelene/Proyecto_final'  # Defining the path

# Function to load the database
def load_database():
    db = pd.read_csv(f'{path}/cyberbullying_tweets_wordclouds.csv')  # Loading the database
    return db

# Function to cache the database using Streamlit
@st.cache_data()
def cached_database():
    return load_database()

# Function to load machine learning models
def load_models():
    random_forest_model = joblib.load(f'{path}/final_random_forest_model.joblib')
    SVM_model = joblib.load(f'{path}/SVC.joblib')
    logistic_regression_model = joblib.load(f'{path}/LogisticRegression.joblib')
    scaler = joblib.load(f'{path}/scaler.joblib')
    neural_network = load_model(f'{path}/best_nueral_network')  # Loading the neural network model
    return random_forest_model, SVM_model, logistic_regression_model, neural_network, scaler

# Function to cache the models using Streamlit
@st.cache_data()
def cached_models():
    return load_models()

# Function for text preprocessing
def preprocesamiento(text:str):
    # Preprocessing steps
    text = text.lower()
    text = re.sub(r'(@\w+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(b[\'\"]rt )', '', text)
    text = word_tokenize(text)
    text = [token for token in text if token not in stop_words_en]
    text = [lemmatizer.lemmatize(token) for token in text]
    text = ' '.join(text)
    return text

# Function to convert text to vectors
def vectorizer(text:str):
    text = preprocesamiento(text)
    text = nlp(text).vector
    text = np.reshape(text, (-1, 300))
    return text

# Function to generate n-grams
def get_ngrams(column, n):
    vocab = []
    for text in column:
        ngrams_1 = ngrams(word_tokenize(text), n)
        for ngram in ngrams_1:
            resultado = " ".join(ngram)
            vocab.append(resultado)
    return pd.Series(vocab).value_counts()

# Function to generate word clouds
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

# Function to generate bar charts
def bar_chart_generator(column, n, type, k = 15):
    ngrams_out = get_ngrams(df[df["cyberbullying_type"] == column]['clean_tweet'], n)
    fig, ax = plt.subplots()
    ngrams_out.head(k).plot.bar(ax=ax)
    ax.set_title(f'Top 15 {n}-grams for {type}')
    ax.set_xlabel('N-grams')
    ax.set_ylabel('Frequency')
    return fig

# Dictionary mapping user options to cyberbullying types
good_looking_options = {
    'Not Cyberbullying':'not_cyberbullying',
    'Gender': 'gender',
    'Religion': 'religion',
    'Other Cyberbullying':'other_cyberbullying',
    'Age':'age',
    'Ethnicity':'ethnicity'
}
bad_looking_options = {value: key for key, value in good_looking_options.items()}

# Loading the cached database and models
df = cached_database()
random_forest_model, SVM_model, logistic_regression_model, nueral_network, scaler = cached_models()

# Function to predict using traditional machine learning models
def predict(vector):
    output_forest = random_forest_model.predict(vector)
    output_SVM = SVM_model.predict(vector)
    output_log_reg = logistic_regression_model.predict(vector)
    outputs = [output_forest, output_SVM, output_log_reg]
    outputs = [bad_looking_options.get(output[0]) for output in outputs]
    return outputs

# Function to predict using the neural network
def predict_neural_network(vector):
    vector = vector.reshape(1, -1)
    vector = scaler.transform(vector)
    prediction = nueral_network.predict(vector)
    return prediction

# Function to generate a pie chart
def piechart_generator(probability_vector):
    fig = px.pie(values=probability_vector, names = good_looking_options.keys())
    return fig

# Main function
def main():
    st.write('# Analysis of Cyberbullying on Twitter')
    st.write('## Data Preprocessing')
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
    st.write('### T-SNE Plot ')
    fig = px.scatter(
        data_frame= df,
        x = df['tsne_1'],
        y = df['tsne_2'],
        template = 'plotly_dark',
        hover_data = ['tweet_text'],
        color = df['cyberbullying_type']
    )
    st.plotly_chart(fig)
    st.write('# Machine Learning')
    text = st.text_input('Input a possibly offensive tweet to classify: ')
    if text != '':
        with st.spinner('Working on it'):
            st.write('## Traditional Machine Learning')
            vector = vectorizer(text)
            output_random_forest, output_SVM, output_log_regression = predict(vector) 
            st.write(f'**Random Forest prediction:** {output_random_forest}')
            st.write(f'**SVM prediction:** {output_SVM}')
            st.write(f'**Logistic Regression prediction:** {output_log_regression}')
            st.write('## Neural Network')
            nueral_network_prediction = predict_neural_network(vector)
            probability_vector = nueral_network_prediction[0]
            fig = piechart_generator(probability_vector)
            st.plotly_chart(fig)
    
if __name__ == '__main__':
    main()
