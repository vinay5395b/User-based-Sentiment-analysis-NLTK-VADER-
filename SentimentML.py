import pandas as pd
import numpy as np
import tweepy
#import json
import csv
import re
#from textblob import TextBlob
import nltk
from nltk.stem.porter import *
from wordcloud import WordCloud

from nltk import sentiment
nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
pd.set_option("display.max_colwidth",200)

# Variables that contains the user credentials to access Twitter API 
access_token = "314787063-qyTCCOwUr0GbVysjmKArwDT9zQPQTl4tl9AY0AW1"
access_token_secret = "uD4BWOS4LSM5rUvtgwkubZuzMXkW1ViKtcefzHbKhEOTm"
consumer_key = "Ivzl3U99MszyXWkVmGVGgzota"
consumer_secret = "vwnxIhKuZe3LwiDbDbQ0lQcHtD3QVjukahMZzNQI8bLyELBzjC"



################################################################################
def get_all_tweets(screen_name):
    #Twitter only allows access to a users most recent 3240 tweets with this method
    
    #authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    
    #initialize a list to hold all the tweepy Tweets
    alltweets = []    
    
    #make initial request for most recent tweets (100 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name = screen_name,count=100)
    
    #save most recent tweets
    alltweets.extend(new_tweets)
    
    #save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1
    
    #keep grabbing tweets until there are no tweets left to grab
    while len(alltweets) < 500:
        print("getting tweets before %s" % (oldest))
        
        #all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name = screen_name,count=100,max_id=oldest)
        
        #save most recent tweets
        alltweets.extend(new_tweets)
        
        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
        
        print("...%s tweets downloaded so far" % (len(alltweets)))
    
    #transform the tweepy tweets into a 2D array that will populate the csv    
    outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] for tweet in alltweets]
    #print(outtweets)     
    #write the csv    
    with open('%s_tweets.csv' % screen_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["id","created_at","text"])
        writer.writerows(outtweets)
    
    pass

#if __name__ == '__main__':
    #pass in the username of the account you want to download
get_all_tweets("chrissyteigen")
get_all_tweets("realDonaldTrump")
get_all_tweets("SpeakerRyan")



chrissy = pd.read_csv("chrissyteigen_tweets.csv")
#len(obama)
trump = pd.read_csv("realDonaldTrump_tweets.csv")
trump = trump[:500]

#len(trump)
ryan = pd.read_csv("SpeakerRyan_tweets.csv")

## Adding label column to each df
chrissy['label'] = 0
trump['label'] = 1
ryan['label'] = 2

# merging the dataframes into one df
df = pd.concat([chrissy, trump, ryan])

df.reset_index(inplace=True)

# We just want the 'text' and 'label' columns, drop all others
df.drop(['index'], axis=1, inplace=True)
df.drop(['id', 'created_at'], axis=1, inplace=True)

# shuffle all the rows
df = df.sample(frac=1).reset_index(drop=True)

backup_df = df

####### Text pre-processing/cleaning #########

df['tidy_tweet'] = df['text']


# user defined func to remove unwanted patterns

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt

# remove user handles
df['tidy_tweet'] = np.vectorize(remove_pattern)(df['tidy_tweet'], "@[\w]*")

# remove links
df['tidy_tweet'] = df['tidy_tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])

# remove ASCII coding
#df['tidy_tweet'] = re.sub(r"[^\x00-\x7F]+", "", df['tidy_tweet'])
#df['tidy_tweet'] = df['tidy_tweet'].str.replace(r'[^\\x00-\\x7F]+', '').astype('str')

# keep only text
df['tidy_tweet'] = df['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")

# remove extra white spaces
df['tidy_tweet'] = df['tidy_tweet'].replace({' +':' '},regex=True)

#remove words having length less than 3, which are meaningless 
df['tidy_tweet'] = df['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))



#df['tidy_tweet'].tail(10)
#df.iloc[1492]


#df['tidy_tweet'].head()



## tokenizing

tokenized_tweet = df['tidy_tweet'].apply(lambda x: x.split())

stemmer = PorterStemmer() 
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

df['tokenized'] = tokenized_tweet

all_words = ' '.join([text for text in df['tokenized']])

# wordcloud for all
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words) 
plt.figure(figsize=(10, 7)) 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis('off') 
plt.show()

# wordcloud for each user

chrissy_wordcl =' '.join([text for text in df['tokenized'][df['label'] == 0]])  # chrissy teigen
trump_wordcl = ' '.join([text for text in df['tokenized'][df['label'] == 1]])
ryan_wordcl = ' '.join([text for text in df['tokenized'][df['label'] == 2]])

# Chrissy Teigen

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(chrissy_wordcl)
plt.figure(figsize=(10, 7)) 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis('off') 
plt.show()


#trump
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(trump_wordcl)
plt.figure(figsize=(10, 7)) 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis('off') 
plt.show()

# Ryan
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(ryan_wordcl)
plt.figure(figsize=(10, 7)) 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis('off') 
plt.show()


####################




## trying VADER for sentiment analysis (to identify tweet intensity)

sid = SentimentIntensityAnalyzer()

#def get_vader_score(sent):
#    # Polarity score returns dictionary
#    ss = sid.polarity_scores(sent)
#    for k in sorted(ss):
#        print('{0}: {1}, '.format(k, ss[k]), end='')
#        #print()
#
#get_vader_score(df['tokenized'][2])
#sid.polarity_scores(df['tokenized'][2])


df['negative'] = 0.0
df['positive'] = 0.0
df['neutral'] = 0.0

for i in range(len(df)):
    ss = sid.polarity_scores(df['tidy_tweet'][i])
    df['negative'][i] = ss['neg']
    df['positive'][i] = ss['pos']
    df['neutral'][i] = ss['neu']



###### Visualizations for comparison

# getting average of sentiment values
avg_chrissy = list(df[df['label']==0].mean())
c_neg = avg_chrissy[1]
c_pos = avg_chrissy[2]
c_neu = avg_chrissy[3]

avg_trump = list(df[df['label']==1].mean())
t_neg = avg_trump[1]
t_pos = avg_trump[2]
t_neu = avg_trump[3]

avg_ryan= list(df[df['label']==2].mean())
r_neg = avg_ryan[1]
r_pos = avg_ryan[2]
r_neu = avg_ryan[3]

avg_val = pd.DataFrame(columns = ['Name','Avg Pos','Avg Neg', 'Avg Neu']) 

avg_val['Name'] = ['Chrissy Teigen', 'Donald Trump', 'Paul Ryan']

avg_val['Avg Pos'][0] = c_pos
avg_val['Avg Pos'][1] = t_pos
avg_val['Avg Pos'][2] = r_pos

avg_val['Avg Neg'][0] = c_neg
avg_val['Avg Neg'][1] = t_neg
avg_val['Avg Neg'][2] = r_neg

avg_val['Avg Neu'][0] = c_neu
avg_val['Avg Neu'][1] = t_neu
avg_val['Avg Neu'][2] = r_neu

avg_val

## Plotting values to chart
avg_val.plot(x="Name", y=["Avg Pos", "Avg Neg", "Avg Neu"], kind="bar", figsize=(10,7))

