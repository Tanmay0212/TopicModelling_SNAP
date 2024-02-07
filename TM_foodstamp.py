#!/usr/bin/env python
# coding: utf-8

# In[1]:


# cell to install packages


# In[2]:


# cell to import packages

import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation 
import numpy as np
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
from gensim.models import LdaModel
from gensim.matutils import Sparse2Corpus
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# In[3]:


# Load the CSV file
df = pd.read_csv('foodstamp_submissions_allyears.csv')  


# In[4]:


# Basic preprocessing steps

print('Filtering out posts that are "deleted" or "removed"...')
df = df[~df['selftext'].str.contains('deleted|removed', case=False, na=False)]

print('Dealing with NA values')
df['selftext'].fillna('', inplace=True)

print('Removing any links')
df['selftext'] = df['selftext'].str.replace(r'http\S+', '', regex=True)


# List of U.S. states
states = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", 
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", 
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", 
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", 
    "New Hampshire", "New Jersey", "New Mexico", "New York", 
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", 
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", 
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", 
    "West Virginia", "Wisconsin", "Wyoming"
]


# In[5]:


# Compile regex patterns for states
state_patterns = {state: re.compile(r'\b' + re.escape(state) + r'\b', re.IGNORECASE) 
                  for state in states}

# Define function to extract states
def extract_states(text):
    found_states = set()
    for state, pattern in state_patterns.items():
        if pattern.search(text):
            found_states.add(state)
    return list(found_states)

# Extracting states and exploding the DataFrame
df['extracted_states'] = df['selftext'].astype(str).apply(extract_states)
df_exploded = df.explode('extracted_states')
df_exploded.rename(columns={'extracted_states': 'state'}, inplace=True)


# In[6]:


# Printing the first 100 extracted state names

for i, state in enumerate(df_exploded['state'].head(100)):
    print(f"Entry {i+1}: {state}")


# In[7]:


# Preprocessing text for topic modeling

print("Preprocessing text for topic modeling...")
df_exploded['processed_text'] = df_exploded['selftext'].str.lower().str.replace(r'[^\w\s]', '', regex=True)

# Tokenize the documents
tokenized_docs = [doc.split() for doc in df_exploded['processed_text']]

# Create a Gensim Dictionary and Corpus
dictionary = corpora.Dictionary(tokenized_docs)
corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

# Function to compute coherence values
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        # Build LDA model
        model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)

        # Compute Coherence Score
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# In[8]:


# Compute coherence values
limit = 40; start = 2; step = 6
model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=tokenized_docs, start=start, limit=limit, step=step)

# Show graph
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


# In[9]:


n_topics = 8  # adjust the number of topics according to the optimal score from above cell
n_top_words = 10

def print_topics(model, count_vectorizer, n_top_words, state):
    words = count_vectorizer.get_feature_names_out()
    print(f"\nState: {state}")
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx, " ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

for state in states:
    # Filter data for the current state
    state_data = df_exploded[df_exploded['state'] == state]

    if state_data.empty:
        continue

    # Preprocess text data
    state_data['processed_text'] = state_data['selftext'].str.lower().str.replace(r'[^\w\s]', '', regex=True)

    # Vectorize the text data
    vectorizer = CountVectorizer(stop_words='english')
    data_vectorized = vectorizer.fit_transform(state_data['processed_text'])

    # Apply LDA for topic modeling
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    lda_model.fit(data_vectorized)

    # Displaying Topics for each state
    print_topics(lda_model, vectorizer, n_top_words, state)


# In[10]:


# Applying topic modeling and visualization (word cloud) simultaneously

n_topics = 8  # Number of topics
n_top_words = 10

# Modified function to print and return top words for topics
def get_topics_top_words(model, count_vectorizer, n_top_words, state):
    words = count_vectorizer.get_feature_names_out()
    topics_top_words = {}
    print(f"\nState: {state}")
    for topic_idx, topic in enumerate(model.components_):
        top_words = [words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        print("Topic #%d:" % topic_idx, " ".join(top_words))
        topics_top_words[topic_idx] = top_words
    return topics_top_words

# Main loop for each state
for state in states:
    state_data = df_exploded[df_exploded['state'] == state]

    if state_data.empty:
        continue

    state_data['processed_text'] = state_data['selftext'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
    vectorizer = CountVectorizer(stop_words='english')
    data_vectorized = vectorizer.fit_transform(state_data['processed_text'])

    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    lda_model.fit(data_vectorized)

    # Get top words for each topic
    topics_top_words = get_topics_top_words(lda_model, vectorizer, n_top_words, state)

    # Generate word clouds for each topic
    for topic, words in topics_top_words.items():
        wordcloud = WordCloud(width=800, height=800, background_color='white', 
                              min_font_size=10).generate(" ".join(words))

        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.title(f"State: {state}, Topic #{topic}")
        plt.show()


# In[ ]:




