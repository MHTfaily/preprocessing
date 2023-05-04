# libraries
import re
import contractions
import enchant
from collections import defaultdict
import nltk
from nltk.util import ngrams
from nltk.corpus import words,stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import emoji
from textblob import TextBlob
import pandas as pd


def get_word_frequency(dataframe):
    word_frequency = defaultdict(int)
    for text in dataframe['text']:
        # Tokenize the text into words
        words = re.findall(r'\b\w+\b', text)
        # Update the word frequency dictionary
        for word in words:
            word_frequency[word.lower()] += 1
    return dict(word_frequency)


all_words = set(nltk.corpus.words.words()) #Used as engish words

social_media_abbreviations = {
    "lol": "laughing out loud",
    "omg": "oh my god",
    "btw": "by the way",
    "bff": "best friends forever",
    "fb": "Facebook",
    "ig": "Instagram",
    "tbt": "throwback Thursday",
    "fomo": "fear of missing out",
    "imo": "in my opinion",
    "irl": "in real life",
    "jk": "just kidding",
    "omw": "on my way",
    "rofl": "rolling on the floor laughing",
    "smh": "shaking my head",
    "tbh": "to be honest",
    "ftw": "for the win",
    "gtg": "got to go",
    "hmu": "hit me up",
    "icymi": "in case you missed it",
    "idk": "I don't know",
    "np": "no problem",
    "thx": "thanks",
    "wtf": "what the f***",
    "yw": "you're welcome",
    "brb": "be right back",
    "afk": "away from keyboard",
    "smh": "shaking my head",
    "imo": "in my opinion",
    "rn": "right now",
    "oml": "oh my lord",
    "nvm": "never mind",
    "fbf": "flashback Friday",
    "smth": "something",
    "imho": "in my humble opinion",
    "ppl": "people",
    "tho": "though",
    "tl;dr": "too long; didn't read",
    "mcm": "Man Crush Monday",
    "wcw": "Woman Crush Wednesday",
    "g2g": "got to go",
    "irl": "in real life",
    "fwiw": "for what it's worth",
    "ftl": "for the loss",
    "jic": "just in case",
    "mt": "modified tweet",
    "pov": "point of view",
    "tfw": "that feeling when",
    "yolo": "you only live once",
    "omdb": "over my dead body",
    "w/e":"weekend"
}


#used for the apply in the dataframe in the class
def remove_url_mention_tag_tweet(tweet):
    # Remove URLs, mentions, and hashtags from tweet
    tweet = re.sub("(?P<url>https?://[^\s]+)", "", tweet)
    tweet = re.sub("(?<!\w)@\w+", "", tweet)
    tweet = re.sub("(?<!\w)#[\w]+", "", tweet)

    return tweet

#used in cleaning the tweets
def expand_social_media_abbreviations_1(tweet, abbreviations):
    words = tweet.split()  # Split the tweet into words
    expanded_tweet = []  # List to store expanded words
    for word in words:
        if word in abbreviations:
            # If the word is an abbreviation, replace it with its full form
            expanded_tweet.append(abbreviations[word])
        else:
            # If not an abbreviation, keep the original word
            expanded_tweet.append(word)
    # Join the words back to form the expanded tweet
    expanded_tweet = ' '.join(expanded_tweet)
    return expanded_tweet

def extract_url_mention_tag_tweet_2(tweet):
    # Extract URLs using regular expression
    urls = re.findall("(?P<url>https?://[^\s]+)", tweet)

    # Extract mentions using regular expression
    mentions = re.findall("(?<!\w)@\w+", tweet)

    # Extract hashtags using regular expression
    hashtags = re.findall("(?<!\w)#[\w]+", tweet)

    # Remove URLs, mentions, and hashtags from tweet
    tweet = re.sub("(?P<url>https?://[^\s]+)", "", tweet)
    tweet = re.sub("(?<!\w)@\w+", "", tweet)
    tweet = re.sub("(?<!\w)#[\w]+", "", tweet)

    return tweet, urls, mentions, hashtags

def remove_rt_tag_3(tweet):
    return re.sub(r'\bRT\b', '', tweet).strip()

def Replace_contractions_with_their_expanded_forms_4(tweet):  #"ain't" => "am not",...
    return contractions.fix(tweet) #used library contractions

def translate_emoji_5(tweet): # ex: it becomes cry...
    return emoji.demojize(tweet)

def remove_non_ascii_6(tweet): #Héllø Wørld! => Hll Wrld!
    encoded_text = tweet.encode("utf-8", "ignore")
    decoded_text = encoded_text.decode("utf-8")
    return decoded_text

def remove_digits_7(tweet): 
    clean_text = re.sub(r'\d+', '', tweet)
    return clean_text

def remove_non_letters_8(tweet):
    # Use regex to remove non-letter characters
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', tweet)
    return cleaned_text

def lowercase_9(tweet):
    return tweet.lower()

def correct_elongated_words_10(tweet): #This is soooo cooooool! GOOOD => This is so cool! GOOD
    #Execution time for 100 simulations: 0.379620 seconds
    tweet = tweet+"."
    is_known_word = enchant.Dict("en").check
    def is_known_word_fast(word):
        if not re.match(r'^[a-zA-Z]+$', word):
            return False
        return is_known_word(word)
    
    words = re.split(r'(\W+)', tweet)
    
    output = []
    for word, sep in zip(words[::2], words[1::2]):
        sep = re.sub(r'(?i)(.)\1+', r'\1', sep)
        if word and not is_known_word_fast(word):
            word_minimized = re.sub(r'(?i)(\w)\1+', r'\1\1', word)
            if is_known_word(word_minimized):
                word = word_minimized
            else:
                word = re.sub(r'(?i)(\w)\1+', r'\1', word)
        output.append(word + sep)
    corrected_tweet = ''.join(output)
    
    return corrected_tweet[:-1]

def remove_short_long_words_11(tweet, min_length=2, max_length=20):
    words = tweet.split()  
    filtered_words = []
    for word in words:
        if len(word) >= min_length and len(word) <= max_length:
            filtered_words.append(word)
    filtered_tweet = ' '.join(filtered_words)  # Join the filtered words back to form the tweet
    return filtered_tweet

def correct_misspelled_words_12(tweet, word_frequency, misspell_indicator=1):    
    
    def get_misspelled_words(tweet):
        # Tokenize the tweet into words
        words = nltk.word_tokenize(tweet)
        # Get the set of English words from nltk corpus
        english_words = all_words
        # Identify misspelled words
        misspelled_words = [word for word in words if word.lower() not in english_words]
        return misspelled_words
    
    new_words = []
    for word in get_misspelled_words(tweet):
        if word_frequency[word] <= misspell_indicator:
            new_words.append((word,str(TextBlob(word).correct())))
    for i in new_words:
        tweet = tweet.replace(i[0],i[1])
    return tweet.lower()

def remove_stopwords_13(tweet):
    stop_words = set(stopwords.words('english'))  # Get the set of English stop words
    words = tweet.split()  # Split the tweet into words
    filtered_words = [word for word in words if word.lower() not in stop_words]  # Filter out stop words
    filtered_tweet = ' '.join(filtered_words)  # Join the filtered words back to form the tweet
    return filtered_tweet

def Tweet_lemmatizer_14(tweet):
    lemmatizer = WordNetLemmatizer()
    new_tweet = []
    tags = nltk.pos_tag(tweet.split(" "), tagset = "universal") #Here we're doing pos taging to use it in the lemmatizing to me more accurate
    new_tags = []
    for i in range(len(tags)):
        l = []
        for j in range(2):
            l.append(tags[i][j])
        new_tags.append(l)
    tags = new_tags
    for i in range(len(tags)):
        if tags[i][1]=="VERB":
            tags[i][1]=tags[i][1].lower()[0]
        elif tags[i][1]=="ADJ":
            tags[i][1]=tags[i][1].lower()[0]
        elif tags[i][1]=="NOUN":
            tags[i][1]=tags[i][1].lower()[0]
        elif tags[i][1]=="ADV":
            tags[i][1]="r"
        else:
            tags[i][1]="n"
    for i in tags:
        new_tweet.append(lemmatizer.lemmatize(i[0], i[1]))
    return new_tweet


class PreProcessingTweets:
    def __init__(self,data):
        self.data = data
        self.url = []
        self.mention = []
        self.tag = []        
        self.data['text'] = self.data['text'].apply(expand_social_media_abbreviations_1, abbreviations=social_media_abbreviations).apply(remove_url_mention_tag_tweet).apply(remove_rt_tag_3).apply(Replace_contractions_with_their_expanded_forms_4).apply(translate_emoji_5).apply(remove_non_ascii_6).apply(remove_digits_7).apply(remove_non_letters_8).apply(lowercase_9).apply(correct_elongated_words_10).apply(remove_short_long_words_11)
        self.word_frequency = get_word_frequency(data)
        
    def clean(self):
        self.data = self.data['text'].apply(correct_misspelled_words_12, word_frequency = self.word_frequency, misspell_indicator=1).apply(remove_stopwords_13).apply(Tweet_lemmatizer_14)

    def get_data(self):
        return self.data
