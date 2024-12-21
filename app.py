# Import necessary libraries
import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Set up SSL context for NLTK downloads
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Extract terms and weights from the vectorizer (to be used later)
def save_tfidf_weights():
    # Ensure the vectorizer is fitted with training data
    x = vectorizer.fit_transform(patterns)
    terms = vectorizer.get_feature_names_out()  # Get all terms/ngrams
    weights = vectorizer.idf_  # Get the weights (importance of terms)

    # Combine terms and weights into a table format
    tfidf_table = pd.DataFrame(list(zip(terms, weights)), columns=["Term/Ngram", "TF-IDF Weight"])

    # Sort terms by their weights for better readability
    tfidf_table = tfidf_table.sort_values(by="TF-IDF Weight", ascending=False)

    # Save to a CSV file
    tfidf_table.to_csv("tfidf_weights.csv", index=False)  # Save the file in the same directory
    print("TF-IDF weights have been saved to 'tfidf_weights.csv'")

# Preprocess the data from intents.json
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Save TF-IDF weights after training
save_tfidf_weights()

# Define the chatbot response function
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

# Counter for unique text input keys
counter = 0

# Define the main Streamlit app
def main():
    global counter

    # Inject custom CSS for better styling
    st.markdown("""
    <style>
    body {
        background-color: #000000;  /* Background color */
        color: #333333;  /* Text color */
    }

    .stApp {
        background-color: #000000;  /* Background for the app */
    }

    .custom-input-container {
        position: relative;
        display: inline-block;
        width: 100%;
        margin-top: 30px;
    }

    .custom-input-container input {
        width: 100%;
        color: #333333;
        background-color: #f0f0f0; /* Light grey background */
        border: 2px solid #4f4f4f; /* Dark grey border */
        border-radius: 5px;
        padding: 8px;
        outline: none;
        font-size: 16px;
        caret-color: black;
    }

    .stTextArea textarea {
        background-color: #f0f0f0; /* Light grey background */
        color: #333333; /* Text color */
        border: 2px solid #4f4f4f; /* Dark grey border */
        border-radius: 5px;
        padding: 8px;
        font-size: 16px;
    }

    .sidebar .sidebar-content {
        background-color: #0066cc;
        color: white;
    }

    .stButton button {
        background-color: #007bff;
        color: white;
    }

    h1 {
        color: #0066cc;
        text-align: center;
        font-size: 28px;
    }

    h2, h3 {
        color: #00FF7F; /* Secondary color */
        font-size: 20px;
    }

    .stMarkdown {
        padding-top: 30px;
        font-size: 18px;
        color: #4f4f4f;
        text-align: center;
        margin-top: -15px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Set the title for the page
    st.title("Educational Tutor")

    # Sidebar menu
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Empowering learning with Technology: Your Guide to Educational Tools and Innovation.")

        # Check if the chat_log.csv file exists
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            user_input_str = str(user_input)
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            # Initialize the timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Write the conversation data to the CSV file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        st.header("Conversation History")

        # Check if the file exists and read it
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip the header row
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")
        else:
            st.write("No conversation history found.")

    # About Menu
    elif choice == "About":
        st.write("This chatbot is designed to assist students with their college-related queries.")
        st.subheader("Features:")
        st.write("""
        - Provides answers to frequently asked questions about Educational Tools and Innovation.
        - Saves conversation history for review.
        - Built using Natural Language Processing (NLP) techniques.
        """)

# Run the app
if __name__ == '__main__':
    main()
