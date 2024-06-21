import pandas as pd
import re
import nltk
import string

from collections import Counter

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel


def clean_hebrew_tweet(text):
  tokens = word_tokenize(text)
  stop_words = set(stopwords.words('hebrew') + ['RT'])
  tokens = [token for token in tokens if (token not in stop_words) and (len(token) > 1)]
  tokens = [token for token in tokens if (token.isalnum()) and (token not in string.punctuation)]
  cleaned_text = " ".join(tokens).strip()
  return cleaned_text

if __name__ == "__main__":
    data = pd.read_csv("data-1716191272369.csv")
    data["length"] = data["text"].str.len()
    nltk.download('punkt')
    nltk.download('stopwords')
    
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )

    url_pattern = re.compile(r'http\S+|www\.\S+')

    data["clean_text"] = data["text"].str.replace(url_pattern, '', regex=True).str.replace(emoji_pattern, '', regex=True).apply(clean_hebrew_tweet)
    data["token_num"] = data["clean_text"].str.split().str.len()
    clean_data = data.query("(length >= 20) & (token_num > 2)").copy(deep=True)
    
    dictionary = Dictionary(clean_data["clean_text"].str.split())
    corpus = [dictionary.doc2bow(text) for text in clean_data["clean_text"].str.split()]
    model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=11, random_state=42)
    
    dominant_topics = []
    for i, row in enumerate(model[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        dominant_topic = row[0][0] + 1
        dominant_topics.append(dominant_topic)
    clean_data['dominant_topic'] = dominant_topics
    
    chosen_topics = clean_data["dominant_topic"].value_counts(normalize=True).sort_values(ascending=False).reset_index().query("proportion > 0.1")["dominant_topic"].tolist()
    clean_data["hashtags"] = clean_data["text"].apply(lambda x: re.findall(r'#\w+', x))
    
    top_topics_names = {}
    for topic in chosen_topics:
        topic_hashtags = [hashtag for hashtags in clean_data.query("dominant_topic == @topic")['hashtags'] for hashtag in hashtags]
        for i in range(1, 10):
            most_common_hashtag = Counter(topic_hashtags).most_common(i)[0][0]
            topic_name = most_common_hashtag.replace('#', '').replace('_', ' ')
            if topic_name not in top_topics_names.values():
                top_topics_names[topic] = topic_name
                break
    for topic, topic_name in top_topics_names.items():
        print(f"Topic {topic} Name: {topic_name}")

    

