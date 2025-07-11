import pandas as pd
import re
import string
import os
from textblob import TextBlob
import matplotlib.pyplot as plt

# --- 1. Load dataset ---
df = pd.read_csv("tweets.csv")  # Just keep tweets.csv in the same directory
print("Data loaded successfully.")

# --- 2. Clean text ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+|#\w+", "", text)  # Remove mentions and hashtags
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text.lower()

df['cleaned_text'] = df['text'].apply(clean_text)

# --- 3. Sentiment Analysis ---
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['cleaned_text'].apply(get_sentiment)

# --- 4. Visualize results ---
plt.figure(figsize=(8, 5))
df['sentiment'].value_counts().plot(kind='bar', color=['green', 'blue', 'red'])
plt.title('Sentiment Analysis of Tweets')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.xticks(rotation=0)

# Save plot
os.makedirs("output", exist_ok=True)
plt.savefig("output/sentiment_distribution.png")
plt.show()

# --- 5. Save results ---
df.to_csv("output/tweets_with_sentiment.csv", index=False)
print("Analysis complete. Output saved to 'output/' folder.")

