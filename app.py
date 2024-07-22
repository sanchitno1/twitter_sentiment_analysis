import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords')
nltk.download('punkt')

# Load the model and vectorizer
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Preprocess function
def preprocess(text):
    text = text.lower()  # Make text lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuations and commas
    text = re.sub(r'\d+', '', text)  # Remove numbers
    tokens = word_tokenize(text)  # Tokenize the text
    words = set(stopwords.words('english'))
    text = [word for word in tokens if word not in words]  # Remove stopwords
    return ' '.join(text)

# Streamlit app
st.title("Twitter Sentiment Analysis")

tweet = st.text_area("Enter the tweet")

if st.button("Predict Sentiment"):
    if tweet:
        tweet_processed = preprocess(tweet)  # Preprocess the input text
        tweet_tfidf = vectorizer.transform([tweet_processed])  # Vectorize the input text
        prediction = model.predict(tweet_tfidf)  # Predict the sentiment
        if prediction[0] == 1:
            predicted = "Positive Sentiment 😁"
            st.write(f'The sentiment of the tweet is: {predicted}')
            st.image('positive.jpg', width=200) 
        elif prediction[0] == 0:
            predicted = "Neutral Sentiment 🙂"
            st.write(f'The sentiment of the tweet is: {predicted}')
            st.image('neutral.jpg', width=200) 
        else:
            predicted = "Negative Sentiment 😣"
            st.write(f'The sentiment of the tweet is: {predicted}')
            st.image('negative.png', width=200) 
    else:
        st.write("Please enter a tweet.")
