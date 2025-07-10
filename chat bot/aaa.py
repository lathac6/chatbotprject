import tkinter as tk
from tkinter import *
import random
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import json

# Load intents data from JSON file
with open('intents.json') as file:
    intents = json.load(file)

# Prepare data for classification
X = []
y = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        X.append(pattern.lower())
        y.append(intent['tag'])

# Vectorize text data
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Function to get response from classifier
def get_response(message):
    message_vectorized = vectorizer.transform([message.lower()])
    predicted_tag = clf.predict(message_vectorized)[0]
    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])
    return "I'm sorry, I don't understand."

# Function to send message and get bot's response
def send_message(event=None):
    message = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)
    if message != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + message + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))
        response = get_response(message)
        ChatLog.insert(END, "Bot: " + response + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

# Create GUI
base = tk.Tk()
base.title("Simple Chatbot")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial")
ChatLog.config(state=DISABLED)
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff', command=send_message)

EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")
EntryBox.bind("<Return>", send_message)

scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()
