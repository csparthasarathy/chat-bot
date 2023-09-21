import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pickle
import numpy as np
from keras.models import load_model
import json
import random

app = FastAPI()

# Load the model and data
lemmatizer = WordNetLemmatizer()
model = load_model('model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Set up templates for rendering HTML
templates = Jinja2Templates(directory="templates")

# Serve static files (CSS) from the "static" directory
app.mount("/static", StaticFiles(directory="static"), name="static")

def clean_up_sentence(sentence
                      ):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.route("/get", methods=["GET", "POST"])

async def get_bot_response(msg: str = Form(...)):
    response = chatbot_response(msg)
    return {"response": response}


def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(intents, ints[0]['intent']) if ints else "I'm sorry, but I couldn't understand your query."
    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

