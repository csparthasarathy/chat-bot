import streamlit as st
import nltk
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import time
from PIL import Image

# Initialize NLTK
nltk.download('popular')

# Load pre-trained model and data
model = load_model('model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Function to clean up the sentence
def clean_up_sentence(sentence):
    # Tokenize the pattern - split words into an array
    sentence_words = nltk.word_tokenize(sentence)
    # Stem each word - create a short form for the word
    sentence_words = [word.lower() for word in sentence_words]
    return sentence_words

# Return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words):
    # Tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # Bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # Assign 1 if the current word is in the vocabulary position
                bag[i] = 1
    return np.array(bag)

# Predict the class of the sentence
def predict_class(sentence, model):
    # Filter out predictions below a threshold
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.6
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Get a response based on intents
def getResponse(intents_json, tag):
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    if ints:
        res = getResponse(intents, ints[0]['intent'])
        return res
    else:
        return "I'm sorry, but I couldn't understand your query."


st.set_page_config(page_title="Jarvis a mental health Chatbot", page_icon=":robot:")


# Define Streamlit UI
st.title("Mental Health Conversational AI")

# Streamlit sidebar
with st.sidebar:
    # Your additional Streamlit sidebar code
  
    
    
    page_bg = f"""
    <style>
    [data-testid="stSidebar"] {{
    background-color:#EFEFE8;
    }}
    
    [data-testid="stToolbar"] {{
    background-color:#FCFCFC;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

    # You can add images, headers, and other elements here
   
    image = Image.open('logo.jpg')
    st.image(image, width=150)  # Adjust the width as needed
    st.markdown("<h1 style='text-align: left;'> About </h1>", unsafe_allow_html=True)
    st.markdown("""
<p style='text-align: left; color: black;'> Meet Jarvis, your friendly mental health chatbot !! Whether you're feeling down, anxious, or stressed, 
Jarvis is here to help you navigate through your emotions and provide you with the guidance you need to feel better.
With Jarvis, you can talk about your mental health concerns in a comfortable way. So don't hesitate to chat with Jarvis anytime, anywhere! </p>
""", unsafe_allow_html=True)



 



# Streamlit main content
user_messages = []
chatbot_messages = []

userText = st.text_area("Type your query here and press 'Enter'", key="user_input")

# Main interaction loop
if st.button("Submit"):
    user_message = userText
    user_messages.append(user_message)

    # Get a response from the chatbot
    chatbot_message = chatbot_response(user_message)
    chatbot_messages.append(chatbot_message)

# Display chat history in paragraphs
for i in range(len(user_messages)):
    st.markdown(f"**You:** {user_messages[i]}", unsafe_allow_html=True)
    st.markdown(f"**Chatbot:** {chatbot_messages[i]}", unsafe_allow_html=True)
















