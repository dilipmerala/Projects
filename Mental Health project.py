#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


file = "MentalHealthSurvey1.txt"


# In[3]:


excel_data_df = pd.read_excel('MentalHealthSurvey.xlsx')

# print whole sheet data
print(excel_data_df)


# In[4]:


import re
import string
frequency = {}
document_text = open(file, 'r')
text_string = document_text.read().lower()
match_pattern = re.findall(r'\b[a-z]{3,15}\b', text_string)
 
for word in match_pattern:
    count = frequency.get(word,0)
    frequency[word] = count + 1
     
frequency_list = frequency.keys()
 
for words in frequency_list:
    print (words, frequency[words])


# In[5]:


from nltk.probability import FreqDist
fd = FreqDist(frequency)
print(fd)


# In[7]:


import matplotlib.pyplot as plt
fd.plot(30, cumulative=False)
plt.savefig('MentalHealthWC.jpg')
plt.show()


# In[8]:


get_ipython().run_line_magic('pwd', '')


# In[9]:


fd_alpha = FreqDist(frequency)
print(fd_alpha)
fd_alpha.plot(30, cumulative=False)


# In[10]:


import sys
sys.path


# In[11]:


import numpy
numpy.__path__


# In[12]:


get_ipython().system('conda env list')


# In[13]:


get_ipython().system('jupyter kernelspec list')


# In[14]:


sys.executable


# In[15]:


from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import numpy as np
import urllib
import requests


# In[16]:


words = text_string
mask = np.array(Image.open(requests.get('http://www.clker.com/cliparts/O/i/x/Y/q/P/yellow-house-hi.png', stream=True).raw))

# This function takes in your text and your mask and generates a wordcloud. 
def generate_wordcloud(words, mask):
    word_cloud = WordCloud(width = 512, height = 512, background_color='white', stopwords=STOPWORDS, mask=mask).generate(words)
    plt.figure(figsize=(10,8),facecolor = 'white', edgecolor='blue')
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig('WC.jpg', dpi = 100)
    plt.show()
    
#Run the following to generate your wordcloud
generate_wordcloud(words, mask)


# In[17]:


# Generate a word cloud image
Newwordcloud = WordCloud().generate(words)

# Display the generated image:
# the matplotlib way:
plt.imshow(Newwordcloud, interpolation='bilinear')
plt.axis("off")


# In[18]:


# lower max_font_size
Newwordcloud1 = WordCloud(max_font_size=40).generate(words)
plt.figure()
plt.imshow(Newwordcloud1, interpolation="bilinear")
plt.axis("off")
plt.savefig('WC1.jpg', dpi=100)
plt.show()


# # Tweepy - Cleaning Twitter data, including removal of URLs as well as stop and collection words, and calculating and plotting word frequencies

# In[33]:


import seaborn as sns
import itertools
import collections

import tweepy as tw
import nltk
from nltk.corpus import stopwords
import networkx

import warnings

warnings.filterwarnings("ignore")

sns.set(font_scale=1.5)
sns.set_style("whitegrid")


# In[20]:


from tweepy import OAuthHandler


# In[21]:


consumer_key= 'your key here'
consumer_secret= 'your key here'
access_token= 'your key here'
access_token_secret= 'your key here'


# In[22]:


auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)


# In[23]:


search_term = "#mental+health -filter:retweets"

tweets = tw.Cursor(api.search,
                   q=search_term,
                   lang="en",
                   since='2018-11-01').items(1000)

all_tweets = [tweet.text for tweet in tweets]

all_tweets[:5]


# In[24]:


def remove_url(txt):
    """Replace URLs found in a text string with nothing 
    (i.e. it will remove the URL from the string).

    Parameters
    ----------
    txt : string
        A text string that you want to parse and remove urls.

    Returns
    -------
    The same txt string with url's removed.
    """

    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())


# In[25]:


all_tweets_no_urls = [remove_url(tweet) for tweet in all_tweets]
all_tweets_no_urls[:5]


# In[26]:


all_tweets_no_urls[0].split()


# In[27]:


all_tweets_no_urls[0].lower().split()


# In[28]:


words_in_tweet = [tweet.lower().split() for tweet in all_tweets_no_urls]
words_in_tweet[:2]


# In[29]:


#List of all words across tweets
all_words_no_urls = list(itertools.chain(*words_in_tweet))

#create counter
counts_no_urls = collections.Counter(all_words_no_urls)

counts_no_urls.most_common(15)


# In[30]:


clean_tweets_no_urls = pd.DataFrame(counts_no_urls.most_common(15),
                             columns=['words', 'count'])

clean_tweets_no_urls.head()


# In[31]:


fig, ax = plt.subplots(figsize=(8, 8))

# Plot horizontal bar graph
clean_tweets_no_urls.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="purple")

ax.set_title("Common Words Found in Tweets (Including All Words)")

plt.show()


# In[34]:


nltk.download('stopwords')


# In[35]:


stop_words = set(stopwords.words('english'))

# View a few words from the set
list(stop_words)[0:10]


# In[36]:


#words in first tweet
words_in_tweet[0]


# In[38]:


tweets_nsw = [[word for word in tweet_words if not word in stop_words]
              for tweet_words in words_in_tweet]

tweets_nsw[0]


# In[39]:


all_words_nsw = list(itertools.chain(*tweets_nsw))

counts_nsw = collections.Counter(all_words_nsw)

counts_nsw.most_common(15)


# In[77]:


clean_tweets_nsw = pd.DataFrame(counts_nsw.most_common(15),
                             columns=['words', 'count'])

fig, ax = plt.subplots(figsize=(8, 8))

# Plot horizontal bar graph
clean_tweets_nsw.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="purple")

ax.set_title("Common Words Found in Tweets (Without Stop Words)")
plt.savefig('MH1.jpg', dpi=100)
plt.show()


# In[41]:


collection_words = ['mentalhealth', 'mental', 'health']
tweets_nsw_nc = [[w for w in word if not w in collection_words]
                 for word in tweets_nsw]


# In[44]:


tweets_nsw[2]


# In[45]:


tweets_nsw_nc[2]


# In[46]:


# Flatten list of words in clean tweets
all_words_nsw_nc = list(itertools.chain(*tweets_nsw_nc))

# Create counter of words in clean tweets
counts_nsw_nc = collections.Counter(all_words_nsw_nc)

counts_nsw_nc.most_common(15)


# In[47]:


#number of unique words across all tweets
len(counts_nsw_nc)


# In[48]:


#top 15 most common words
clean_tweets_ncw = pd.DataFrame(counts_nsw_nc.most_common(15),
                             columns=['words', 'count'])
clean_tweets_ncw.head()


# In[78]:


fig, ax = plt.subplots(figsize=(8, 8))

# Plot horizontal bar graph
clean_tweets_ncw.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="purple")

ax.set_title("Common Words Found in Tweets (Without Stop or Collection Words)")
plt.savefig('MH2.jpg', dpi=100)
plt.show()


# # Exploring co-occuring words

# In[51]:


from nltk import bigrams


# In[52]:


# Create list of lists containing bigrams in tweets
terms_bigram = [list(bigrams(tweet)) for tweet in tweets_nsw_nc]

# View bigrams for the first tweet
terms_bigram[0]


# In[53]:


# Use a counter to capture the bigrams as dictionary keys and their counts are as dictionary values.
#Begin by flattening the list of bigrams. 
#You can then create the counter and query the top 20 most common bigrams across the tweets.

# Flatten list of bigrams in clean tweets
bigrams = list(itertools.chain(*terms_bigram))

# Create counter of words in clean bigrams
bigram_counts = collections.Counter(bigrams)

bigram_counts.most_common(20)


# In[54]:


bigram_df = pd.DataFrame(bigram_counts.most_common(20),
                             columns=['bigram', 'count'])

bigram_df


# In[55]:


# Create dictionary of bigrams and their counts
d = bigram_df.set_index('bigram').T.to_dict('records')


# In[57]:


import networkx as nx


# In[58]:


# Create network plot 
G = nx.Graph()

# Create connections between nodes
for k, v in d[0].items():
    G.add_edge(k[0], k[1], weight=(v * 10))

G.add_node("usa", weight=100)


# In[79]:


fig, ax = plt.subplots(figsize=(10, 8))

pos = nx.spring_layout(G, k=2)

# Plot networks
nx.draw_networkx(G, pos,
                 font_size=16,
                 width=3,
                 edge_color='grey',
                 node_color='purple',
                 with_labels = False,
                 ax=ax)

# Create offset labels
for key, value in pos.items():
    x, y = value[0]+.135, value[1]+.045
    ax.text(x, y,
            s=key,
            bbox=dict(facecolor='red', alpha=0.25),
            horizontalalignment='center', fontsize=13)
plt.savefig('MH3.jpg', dpi=100)    
plt.show()


# # Sentiment Analysis

# In[61]:


from textblob import TextBlob


# In[64]:


#Create texblob objects of the tweets
sentiment_objects = [TextBlob(tweet) for tweet in all_tweets_no_urls]

sentiment_objects[4].polarity, sentiment_objects[4]


# In[65]:


# Create list of polarity valuesx and tweet text
sentiment_values = [[tweet.sentiment.polarity, str(tweet)] for tweet in sentiment_objects]

sentiment_values[4]


# In[67]:


# Create dataframe containing the polarity value and tweet text
sentiment_df = pd.DataFrame(sentiment_values, columns=["polarity", "tweet"])

sentiment_df.head(10)


# In[69]:


fig, ax = plt.subplots(figsize=(8, 6))

# Plot histogram of the polarity values
sentiment_df.hist(bins=[-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1],
             ax=ax,
             color="purple")

plt.title("Sentiments from Tweets on Mental Health")
plt.show()


# In[70]:


# Remove polarity values equal to zero
sentiment_df = sentiment_df[sentiment_df.polarity != 0]


# In[80]:


fig, ax = plt.subplots(figsize=(8, 6))

# Plot histogram with break at zero
sentiment_df.hist(bins=[-1, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1],
             ax=ax,
             color="purple")

plt.title("Sentiments from Tweets on Mental Health")
plt.savefig('MH4.jpg', dpi=100)
plt.show()


# # Exploring Job Search tweets

# In[73]:


search_term = "#jobsearch -filter:retweets"

job_tweets = tw.Cursor(api.search,
                   q=search_term,
                   lang="en",
                   since='2018-09-23').items(1000)

# Remove URLs and create textblob object for each tweet
job_tweets_no_urls = [TextBlob(remove_url(tweet.text)) for tweet in job_tweets]

job_tweets_no_urls[:5]


# In[74]:


# Calculate polarity of tweets
job_sent_values = [[tweet.sentiment.polarity, str(tweet)] for tweet in job_tweets_no_urls]

# Create dataframe containing polarity values and tweet text
job_sent_df = pd.DataFrame(job_sent_values, columns=["polarity", "tweet"])
job_sent_df = job_sent_df[job_sent_df.polarity != 0]

job_sent_df.head()


# In[81]:


fig, ax = plt.subplots(figsize=(8, 6))

job_sent_df.hist(bins=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1],
        ax=ax, color="purple")

plt.title("Sentiments from Tweets on Job Search")
plt.savefig('JS1.jpg', dpi=100)
plt.show()


# In[82]:


get_ipython().run_line_magic('pwd', '')


# In[ ]:




