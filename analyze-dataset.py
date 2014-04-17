import json
import numpy as np 
import random
import nltk
import matplotlib.pyplot as plt 
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict


#bjson='yelp_academic_dataset_business.json'
rjson='yelp_academic_dataset_review.json'
# This func reads the json file and yields one line of information at a time

def jsonReader(jsonfile):
    with open(jsonfile) as fil:
        #businesses=fil.read().split('\n')
        for line in fil:
            try:
                info = json.loads(line)
                yield info
            except:
                print "Reached end of file"
                continue

def analyze_total_reviews(generator):
    x=[]
    y=[]
    count=0
    for info in generator:
        count+=1
        x.append(count)
        y.append(info['review_count'])
    plt.scatter(x,y)
    plt.show()

# This function splits the dataset into train and testsets
def split_dataset(data,testpercent):
    testsize=len(data)*testpercent

    test_idx=random.sample(xrange(len(data)),testsize)
    xtest=np.zeros([testsize,data.shape[1]-1])
    ytest=np.zeros([testsize,1])
    for i in xrange(len(test_idx)):
        xtest[i]=data[test_idx[i,:-1]]
        ytest[i]=data[test_idx[i,-1]]

    trainsize=len(data)-size

    xtrain=np.zeros([trainsize,data.shape[1]-1])
    ytrain=np.zeros([trainsize,1])
    train_idx=list(set(xrange(len(data)))-set(test_idx))
    for i in xrange(len(train_idx)):
        xtrain[i]=data[train_idx[i,:-1]]
        ytrain[i]=data[train_idx[i,1]]
    
    return xtrain,ytrain,xtest,ytest


def select_relavant_words(text):
    text=nltk.word_tokenize(text)
    tagged_text = nltk.pos_tag(text)
    imp_words=[]
    for word,tag in tagged_text:
        if tag in set(['JJ','JJR','JJS','RB','RBR','RBS','UH']):
            imp_words.append(word)
    return ' '.join(imp_words)

def get_linguistic_feature_vector(corpus):
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)
    return X,vectorizer

def calc_imp_word_frequencies(generator):
    word_freq=defaultdict(int)
    for i,review in enumerate(generator):
        print iy
        for word in select_relavant_words(review['text']).split(' '):
            word_freq[word]+=1
            #print word_freq
            #raw_input('')
    return word_freq

def main():
    pass

def generate_corpus_file(generator):
    with open('corpus.txt','w') as fil:
        for i,review in enumerate(generator):
            print i
            if i%100==0:
                print review
            if review['votes']['useful']>0:
                fil.write(select_relavant_words(review['text'].encode('utf-8')))
                fil.write('\n')


generator=jsonReader(rjson)
#generate_corpus_file(generator)
#word_freq=calc_imp_word_frequencies(generator)
#tagged=create_linguistic_features(generator.next()['text'])
#analyze_total_reviews(generator)

"""
50749
'Bagels','Bakeries','Barbeque','Bars','Beer', 'Wine & Spirits','Breakfast & Brunch','Buffets','Bubble Tea','Burgers','Cafes','Cafeteria','Chicken Wings','Cheesesteaks','Coffee & Tea','Comfort Food','Creperies','Cupcakes','Delis','Desserts','Dim Sum','Diners','Donuts','Ethnic Food','Fast Food','Fish & Chips','Food','Food Court','Food Trucks','Food Stands','Gelato','Gluten-Free','Halal','Ice Cream & Frozen Yogurt','Hot Dogs','Juice Bars & Smoothies','Live/Raw Food','Pizza','Pita','Pretzels','Restaurants','Salad','Sandwiches','Seafood','Soul Food','Soup','Steakhouses','Sushi Bars','Szechuan','Tapas Bars','Tapas/Small Plates','Tex-Mex','Vegan','Vegetarian','Wine Bars'
'JJ','JJR','JJS','RB','RBR','RBS','UH'
"""
