import requests
import ssl
import socket
from urllib.parse import urlparse
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from extract import extract_features
from feature import *
from bs4 import BeautifulSoup

global feedback_data
feedback_data = pd.DataFrame(columns=['ssl_score', 'url_score', 'content_score','image_score', 'overall_score', 'label'])


# Load the Malicious URL Prediction model
model_path = r"./model/Malicious_URL_Prediction.h5"

# Initialize DistilBERT model and tokenizer
nlp_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
nlp_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Initialize ResNet model for image analysis
image_model = models.resnet50(pretrained=True)
image_model.eval()

# Image preprocessing transformations
image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize feedback data storage and classification model
feedback_data = pd.DataFrame(columns=['ssl_score', 'url_score', 'content_score', 
                                      'image_score', 'overall_score', 'label'])
classification_model = RandomForestClassifier(n_estimators=100, random_state=42)

def check_ssl(url):
    try:
        hostname = urlparse(url).hostname
        context = ssl.create_default_context()
        with socket.create_connection((hostname, 443)) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                return 100  # SSL valid
    except Exception:
        return 0  # SSL invalid

def get_prediction(url, model_path):
    model = keras.models.load_model(model_path)
    url_features = extract_features(url)
    url_features_array = np.array([url_features])
    prediction = model.predict(url_features_array)
    return round(prediction[0][0] * 100, 3)

def analyze_content(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        text_content = ' '.join(soup.stripped_strings)

        inputs = nlp_tokenizer(text_content, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = nlp_model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        content_similarity = probabilities[0][1].item()

        form_presence = 1 if soup.find('form') else 0

        suspicious_keywords = ['login', 'password', 'credit card', 'social security', 'urgent', 'verify']
        keyword_count = sum(keyword in text_content.lower() for keyword in suspicious_keywords)
        keyword_score = max(0, 1 - keyword_count / len(suspicious_keywords))

        # Improved content score calculation
        content_length = len(text_content)
        length_score = min(1, content_length / 1000)  # Assume 1000 characters is a good length
        link_count = len(soup.find_all('a'))
        link_score = min(1, link_count / 20)  # Assume 20 links is a good number

        content_score = (content_similarity + (1 - form_presence) + keyword_score + length_score + link_score) / 5 * 100
        
        text_analysis = []
        if form_presence:
            text_analysis.append("The page contains a form, which could be used to collect sensitive information.")
        if keyword_count > 0:
            text_analysis.append(f"Found {keyword_count} suspicious keywords that are common in phishing attempts.")
        if 'https' in text_content.lower():
            text_analysis.append("The page mentions 'https', which could be an attempt to appear secure.")
        if 'update' in text_content.lower() or 'verify' in text_content.lower():
            text_analysis.append("The page asks users to update or verify information, a common phishing tactic.")
        if not text_analysis:
            text_analysis.append("No specific suspicious elements found in the text content.")

        return content_score, text_analysis
    except Exception as e:
        print(f"Content analysis error: {e}")
        return 0, ["Unable to analyze content"]

def analyze_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_t = image_transforms(img)
        batch_t = torch.unsqueeze(img_t, 0)

        with torch.no_grad():
            out = image_model(batch_t)

        probabilities = torch.nn.functional.softmax(out[0], dim=0)
        general_score = probabilities.max().item()

        img_cv = cv2.imread(image_path)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 100, 200)
        edge_ratio = np.sum(edges > 0) / edges.size

        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
        dilation = cv2.dilate(thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        text_regions = sum(cv2.contourArea(c) for c in contours)
        text_ratio = text_regions / (img_cv.shape[0] * img_cv.shape[1])

        orb = cv2.ORB_create()
        kp = orb.detect(img_cv, None)
        logo_score = len(kp) / 1000

        image_score = (general_score + (1 - edge_ratio) + (1 - text_ratio) + logo_score) / 4 * 100

        image_analysis = []
        if edge_ratio > 0.1:
            image_analysis.append("The image contains many edges, which could indicate a complex or poorly designed interface.")
        if text_ratio > 0.3:
            image_analysis.append("The image contains a large amount of text, which is common in phishing attempts.")
        if logo_score < 0.1:
            image_analysis.append("No clear logo detected, which is unusual for legitimate websites.")
        if general_score < 0.5:
            image_analysis.append("The image doesn't strongly resemble common website layouts.")
        if not image_analysis:
            image_analysis.append("No specific suspicious elements found in the image.")

        return image_score, image_analysis
    except Exception as e:
        print(f"Image analysis error: {e}")
        return 0, ["Unable to analyze image"]

def calculate_overall_score(scores):
    weights = {
        'ssl_score': 0.15,
        'url_score': 0.2,
        'content_score': 0.25,
        'image_score': 0.25
    }
    overall_score = sum(score * weights[key] for key, score in scores.items())
    return overall_score

def train_model():
    global classification_model, feedback_data
    if len(feedback_data) > 1:
        X = feedback_data.drop('label', axis=1)
        y = feedback_data['label']

        le = LabelEncoder()
        y = le.fit_transform(y)

        classification_model.fit(X, y)
        y_pred = classification_model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"Model updated. Current accuracy: {accuracy:.2f}")
        if len(feedback_data) > 10:
            print(classification_report(y, y_pred))
    else:
        print("Not enough data to train the model yet. Using simple heuristics for now.")

def analyze_website(url, image_path):
    ssl_score = check_ssl(url)
    url_score = get_prediction(url, model_path)
    content_score, text_analysis = analyze_content(url)
    image_score, image_analysis = analyze_image(image_path)

    scores = {
        'ssl_score': ssl_score,
        'url_score': 100-url_score,
        'content_score': content_score,
        'image_score': image_score
    }

    overall_score = calculate_overall_score(scores)

    return scores, overall_score, text_analysis, image_analysis

def get_verdict(overall_score):
    if overall_score >= 80:
        return "The website is highly likely legitimate."
    elif 55 <= overall_score < 80:
        return "The website is likely legitimate."
    elif 40 <= overall_score < 55:
        return "The website is likely fraudulent."
    else:
        return "The website is highly likely fraudulent."


def main():
    global feedback_data
    while True:
        url = input("Enter the URL to analyze (or 'quit' to exit): ")
        if url.lower() == 'quit':
            break
        image_path = input("Enter the path to the screenshot image: ")

        scores, overall_score, text_analysis, image_analysis = analyze_website(url, image_path)

        print("\nAnalysis Results:")
        print(f"SSL Score: {scores['ssl_score']:.2f}%")
        print(f"URL Score: {scores['url_score']:.2f}%")
        print(f"Content Score: {scores['content_score']:.2f}%")
        print(f"Image Score: {scores['image_score']:.2f}%")
        print(f"Overall Score: {overall_score:.2f}%\n")

        print("Text Analysis:")
        for analysis in text_analysis:
            print(f"- {analysis}")

        print("\nImage Analysis:")
        for analysis in image_analysis:
            print(f"- {analysis}")

        verdict = get_verdict(overall_score)
        print(f"\nVerdict: {verdict}")

        feedback = input("\nWas the prediction correct? (yes/no): ")
        new_data = pd.DataFrame([{**scores, 'overall_score': overall_score, 'label': 'correct' if feedback.lower() == 'yes' else 'incorrect'}])
        feedback_data = pd.concat([feedback_data, new_data], ignore_index=True)

        train_model()

if __name__ == "__main__":
    main()