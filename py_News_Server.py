import csv
import os

import requests
import socket
import threading
from time import sleep
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder


filepath = "C:\\Users\\micha\\Downloads\\pycharm\\pythonServer\\articles.csv"


def scrape_urls(news_site):
    # Initialise specific homepage, and html classes used by each site
    if news_site == "BBC":
        homepage_url = "https://www.bbc.co.uk/news"
        url_class = "gs-c-promo-heading"
        image_class = None
        paragraph_class = lambda x: x and x.startswith("ssrcss-") and x.endswith("-Paragraph")
    elif news_site == "Guardian":
        homepage_url = "https://www.theguardian.com/uk-news"
        url_class = "u-faux-block-link__overlay"
        image_class = "dcr-evn1e9"
        paragraph_class = None
    elif news_site == "SKY":
        homepage_url = "https://news.sky.com"
        url_class = "sdc-site-tile__headline-link"
        image_class = "sdc-article-image__item"
        paragraph_class = None

    # Load BeautifulSoup Parser
    homepage_info = requests.get(homepage_url)
    parser = BeautifulSoup(homepage_info.content, "html.parser")

    # Load URLs from CSV
    with open(filepath, "r") as file:
        reader = csv.reader(file)
        url_list = set(row[4] for row in reader)

    # Find URLs on website homepage
    links = parser.find_all("a", class_=url_class)
    for link in links:
        article_url = link["href"]

        # Account for relative links
        if not article_url.startswith("https://") and news_site == "BBC":
            article_url = f"https://www.bbc.co.uk{article_url}"
        elif not article_url.startswith("https://") and news_site == "Guardian":
            article_url = f"https://www.theguardian.com{article_url}"
        elif not article_url.startswith("http") and news_site == "SKY":
            article_url = f"https://news.sky.com{article_url}"

        # Only Scrape new URLs
        if article_url not in url_list:
            print(f"Scraping: {article_url}")
            # Removes non text based Articles and start scraping them
            if news_site == "BBC" and "live_coverage" not in article_url and "gettyimages" not in article_url:
                scrape_article(news_site, article_url, paragraph_class)
            elif news_site == "Guardian" and not article_url.endswith("-live") and '/audio' not in article_url:
                scrape_article(news_site, article_url, paragraph_class)
            elif news_site == "SKY" and "live-updates" not in article_url:
                scrape_article(news_site, article_url, paragraph_class)


def scrape_article(news_site, article_url, paragraph_class):
    # Load BeautifulSoup Parser
    webpage = requests.get(article_url)
    parser = BeautifulSoup(webpage.content, "html.parser")

    # Get Current Date
    date = datetime.now().strftime("%d/%m/%y %H:%M")

    # Scrape Title of Article
    try:
        title = parser.find("h1").text
        title = title.replace("\n", "").replace("\t", "")
    except AttributeError:
        title = None

    # Scrape First Image
    if news_site == "BBC":
        first_image = parser.find("img")["src"]
        if not first_image.endswith("jpg"):
            first_image = "https://ichef.bbci.co.uk/images/ic/1920xn/p09xtmrp.jpg"
    elif news_site == "Guardian":
        first_image = parser.find("img", class_="dcr-evn1e9")
        if first_image is not None:
            first_image = first_image["src"]
        else:
            first_image = "https://i.guim.co.uk/img/static/sys-images/Guardian/Pix/pictures/2015/1/28/1422439180039/f8deb38f-0283-45a0-87e9-7c2db48b8d9a-1020x612.png?width=1200&height=630&quality=85&auto=format&fit=crop&overlay-align=bottom%2Cleft&overlay-width=100p&overlay-base64=L2ltZy9zdGF0aWMvb3ZlcmxheXMvdGctZGVmYXVsdC5wbmc&s=894c877383167eed473f3867e2076ea1"
    elif news_site == "SKY":
        first_image = parser.find("img", class_="sdc-article-image__item")
        if first_image is not None:
            first_image = first_image["src"]
        else:
            first_image = "https://www.google.com/url?sa=i&url=https%3A%2F%2F1000logos.net%2Fsky-news-logo%2F&psig=AOvVaw2pDid-7NfprcEaBbGH64j1&ust=1681198182470000&source=images&cd=vfe&ved=0CBAQjRxqFwoTCPDU1LXlnv4CFQAAAAAdAAAAABAI"

    # Scrape all paragraphs
    paragraphs = parser.find_all("p", class_=paragraph_class)
    if news_site == "BBC":
        text_body = " ".join(p.text for p in paragraphs)
    else:
        text_body = ""
        for p in paragraphs:
            if p.text.startswith("\n"):
                break
            text_body += p.text + " "

    # Categories text with AI
    text_body = ai_check(text_body)

    # Store Information in a CSV file
    if text_body and title:
        with open(filepath, "a", newline="") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow([first_image, title, date, text_body, article_url])


def message_client(client):
    # Read in and remove white space around message
    request = client.recv(100).decode().strip()
    print(request)
    # Verify Client
    if request == "Send":
        # Read in and send Article Information
        with open(filepath, "r") as file:
            send_data = file.read()
            client.send(send_data.encode())
            print(send_data)
    # Close connection
    client.close()


def start_server():
    # Create new TCP Socket for Server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Start Server with localhost on port 9999 (Configure APK to use same port)
    server.bind(("localhost", 9999))
    server.listen()
    print("Server running on localhost port 9999 waiting for client connection")

    while True:
        # Accept incoming client
        client, addr = server.accept()
        # Spawn Thread to each new Client Connection
        client_connection = threading.Thread(target=message_client, args=(client,))
        client_connection.start()


def setup_scraping():

    while True:
        # Poll every 30 minutes for new articles
        scrape_urls("BBC")
        scrape_urls("Guardian")
        scrape_urls("SKY")

        # Get 1 day ago
        day_old = datetime.now() - timedelta(days=1)

        # Read file of saved Classified Articles
        with open(filepath, "r") as file:
            # Create file reader and List of Articles to keep and Set of Articles for unique urls
            duplicate_articles = set()
            articles = csv.reader(file)
            in_date = []
            # Populate list of Articles to keep with unique Articles less than a day old
            for article in articles:
                if datetime.strptime(article[2], "%d/%m/%y %H:%M") >= day_old and article[1] not in duplicate_articles:
                    in_date.append(article)
                    duplicate_articles.add(article[1])

        # Write file of saved Classified Articles
        with open(filepath, "w") as file:
            # Keep format same for APK CSV Reader
            writer = csv.writer(file, lineterminator="\n")
            # Write Articles
            for article in in_date:
                writer.writerow(article)

        sleep(1800)


def ai_check(text_body):
    # Pad text to have same length
    text_body_int = tokenizer.texts_to_sequences([text_body])
    text_body_input = tf.keras.preprocessing.sequence.pad_sequences(text_body_int, maxlen=1777)

    # Use model to predict category
    probability = Loaded_NEWS_AI.predict(text_body_input)

    # Find most likely prediction
    category = np.argmax(probability, axis=1)
    # Use label encoder to turn category int into label
    category_label = label_encoder.inverse_transform(category)
    return f"{category_label[0]}"


# Read in file
dataset = pd.read_csv('C:\\Users\\micha\\Downloads\\large_dataset.csv', header=None, names=['text', 'label'])

# Split data for training and testing randomly
train_dataset, test_dataset = train_test_split(dataset, test_size=0.15, random_state=42)

# Splitting training and testing data into data and classification
X_train = train_dataset['text'].values
y_train = train_dataset['label'].values
X_test = test_dataset['text'].values
y_test = test_dataset['label'].values

# Use Tokenizer to identify words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(np.concatenate((X_train, X_test)))

# Replace words with matched int
X_train_int = tokenizer.texts_to_sequences(X_train)
X_test_int = tokenizer.texts_to_sequences(X_test)

# Use label encoder to change text labels into ints [0, 1, 2, 3, 4, 5]
label_encoder = LabelEncoder()

y_train_int = label_encoder.fit_transform(y_train)
y_test_int = label_encoder.transform(y_test)

# Load pretrained model
Loaded_NEWS_AI = tf.keras.models.load_model('C:/Users/micha/Downloads/Saved_News_AI')

# Start Scraping and Classifying Articles
Scraping_Thread = threading.Thread(target=setup_scraping)
Scraping_Thread.start()

# Start Server
start_server()
