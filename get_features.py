import json
import numpy as np 
import random
import nltk
import matplotlib.pyplot as plt 
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from collections import defaultdict
import pickle


bjson='yelp_academic_dataset_business.json'
rjson='yelp_academic_dataset_review.json'
Cfilename='corpus2.txt'
pklfile='features.pkl'
sentipkl='sentiWordNet.p'

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
def split_dataset(data,test_idx,testsize):
    xtest=np.zeros([testsize,data.shape[1]-1])
    ytest=np.zeros([testsize,1])
    for i in xrange(len(test_idx)):
        xtest[i]=data[test_idx[i]][:-1]
        ytest[i]=data[test_idx[i]][-1]

    trainsize=len(data)-testsize

    xtrain=np.zeros([trainsize,data.shape[1]-1])
    ytrain=np.zeros([trainsize,1])
    train_idx=list(set(xrange(len(data)))-set(test_idx))
    for i in xrange(len(train_idx)):
        xtrain[i]=data[train_idx[i]][:-1]
        ytrain[i]=data[train_idx[i]][-1]
    
    return xtrain,ytrain,xtest,ytest


def select_relavant_words(text):
    text=nltk.word_tokenize(text)
    tagged_text = nltk.pos_tag(text)
    imp_words=[]
    for word,tag in tagged_text:
        if tag in set(['JJ','JJR','JJS','RB','RBR','RBS','UH']):
            imp_words.append(word)
    return ' '.join(imp_words)

def get_linguistic_feature_vector(corpus,vocab,vect):
    if vect=='tfidf':
        vectorizer = TfidfVectorizer(min_df=1,vocabulary=vocab)
    else:
        vectorizer = CountVectorizer(min_df=1,vocabulary=vocab)
    X = vectorizer.fit_transform(corpus)
    return X,vectorizer

def calc_imp_word_frequencies(generator):
    word_freq=defaultdict(int)
    for i,review in enumerate(generator):
        print i
        for word in select_relavant_words(review['text']).split(' '):
            word_freq[word]+=1
            #print word_freq
            #raw_input('')
    return word_freq

# This function filters out businesses related to food and dining and returns the imp business ids
def filter_businesses():
    imp_ids=[]
    Bfeatures=[]
    check_feat=defaultdict(int)
    fv=[]
    y=[]
    Bgenerator=jsonReader(bjson)
    for business in Bgenerator:
        if len(set(business['categories']) & set(['Restaurants','Food']))>0:
            imp_ids.append(business['business_id'])
            fv.append([])
            for feat in ['Good For Groups','Good for Kids','Has TV','Outdoor Seating','Price Range']:
                if feat in business['attributes'].keys():
                    fv[-1].append(int(business['attributes'][feat]))
                else:
                    fv[-1].append(-1)
            """
            try:
                Bfeatures.append([business['Price Range'],business['Noise Level']])
            except:
                print business
                raw_input('')
            """
            y.append(business['stars'])
    return imp_ids,Bfeatures,y,np.array(fv)

def generate_review_corpus(imp_ids,Rgenerator):
    with open('corpus2.txt','w') as fil:
        for i,review in enumerate(Rgenerator):
            print i
            if i==100:
                print review
            if review['votes']['useful']>0 and review['business_id'] in imp_ids:
                fil.write(select_relavant_words(review['text'].encode('utf-8')))
                fil.write('\n')

def getTop_n_words(n,imp_ids):
    vocab=pickle.load(open('top_words.pkl'))
    corpus=open('business-texts.txt').read().split('\n')[:-1]

    return vocab[:n],corpus
    """
    word_freq=defaultdict(int)
    for i,text in enumerate(corpus):
        print i
        for word in text.rstrip().split(' '):
            word_freq[word.lower()]+=1

    corpus=['']*len(imp_ids)
    i=0
    Rgenerator=jsonReader(rjson)
    for review in Rgenerator:
        if review['votes']['useful']>0 and review['business_id'] in imp_ids:
            idx=imp_ids.index(review['business_id'])
            if i%1000==0:
                print i
            text=review['text'].encode('utf-8')
            corpus[idx]+=text
            #for word in text.rstrip().split(' '):
            #    word_freq[word.lower()]+=1
            i+=1

    with open('business-texts.txt','w') as fil:
        for i,text in enumerate(corpus):
            print i
            text=text.replace('\n',' ')
            fil.write(text)
            fil.write('\n')
    print len(corpus)
    print len(imp_ids)
    raw_input('')

    with open('top_words.pkl','w') as pklf:
        pickle.dump([a for a,b in sorted(word_freq.items(),key=lambda x: x[1],reverse=True)],pklf)
    return [a for a,b in sorted(word_freq.items(),key=lambda x: x[1],reverse=True)[:n]],corpus
    """
def get_business_features(imp_ids,fv,x):
    i=0
    Rgenerator=jsonReader(rjson)
    try:
        for review in Rgenerator:
            if review['votes']['useful']>0 and review['business_id'] in imp_ids:
                idx=imp_ids.index(review['business_id'])
                print i
                fv[idx]+=x[i]
                i+=1
    except:
        return fv
    return fv
def get_feature_pickle(ctype,y,n,x,imp_business_ids,addFeatures,addFeat):
    #vocab,corpus=getTop_n_words(100)
    #x,vectorizer=get_linguistic_feature_vector(corpus,vocab,'normal')
    #imp_business_ids,Bfeatures,y,addFeat=filter_businesses()
    rowsize=len(y)
    #fv=np.zeros([rowsize,n])
    features=x.toarray()
    #features=get_business_features(imp_business_ids,fv,x)
    if addFeatures==True:
        print "Additional features added"
        features=np.hstack((features,addFeat))
    with open('pklfiles/'+ctype+'-'+str(n)+'-'+str(int(addFeatures))+'.pkl','w') as fil:
        pickle.dump(features,fil)

def return_train_test(y,fv):
    data=np.hstack((fv,np.array(y)[:,np.newaxis]))
    #dataS=np.hstack((fvS,np.array(y)[:,np.newaxis]))
    #dataA=np.hstack((fvA,np.array(y)[:,np.newaxis]))
    testsize=int(len(data)*0.25)
    test_idx=random.sample(xrange(len(data)),testsize)

    #fourTupleNormalFeatures=split_dataset(data,test_idx,testsize)
    #fourTupleSentiFeatures=split_dataset(dataS,test_idx,testsize)
    #fourTupleAddFeatures=split_dataset(dataA,test_idx,testsize)
    #return fourTupleNormalFeatures,fourTupleSentiFeatures,fourTupleAddFeatures
    return split_dataset(data,test_idx,testsize)

def process_sentiments(pklfile,sentipkl,n,vectorizer):
    senti=pickle.load(open(sentipkl))
    fv=pickle.load(open(pklfile))

    names=vectorizer.get_feature_names()

    wts=np.ones([n])
    for i,name in enumerate(names):
        if name+'#a' in senti.keys():
            wts[i]=senti[name+'#a']['posScore']
        else:
            wts[i]=0.1
    print "wtss"

    fwts=np.tile(wts,(fv.shape[0],1))
    senti_fv=np.multiply(fv,fwts)
    return senti_fv


def main(ctype,ftype,n,addFeat):
    pklfile='pklfiles/'+ctype+'-'+str(n)+'-'+str(int(addFeat))+'.pkl'
    try:
        print "Getting imprtant businesses"
        imp_business_ids,Bfeatures,y,add_fv=filter_businesses()
        fv=pickle.load(open(pklfile))
    except:
        print "Getting imprtant businesses"
        imp_business_ids,Bfeatures,y,add_fv=filter_businesses()
        print "Getting top n words..."
        vocab,corpus=getTop_n_words(n,imp_business_ids)

        print "Getting vectorizer and X..."
        if ctype=='tfidf':
            print 'Ctype:',ctype
            x,vectorizer=get_linguistic_feature_vector(corpus,vocab,'tfidf')
        else:
            print 'Ctype:',ctype
            x,vectorizer=get_linguistic_feature_vector(corpus,vocab,'normal')
        
        get_feature_pickle(ctype,y,n,x,imp_business_ids,addFeat,add_fv)
        fv=pickle.load(open(pklfile))


    if ftype=='normal':
        print 'Ftype: ',ftype
        return return_train_test(y,fv)
    elif ftype=='senti':
        print 'Ftype: ',ftype
        senti_fv=process_sentiments(pklfile,sentipkl,n,vectorizer)
        return return_train_test(y,senti_fv)


    
#Bgenerator=jsonReader(bjson)
#Rgenerator=jsonReader(rjson)


"""
Ctype: normal,tfidf  Ftype:normal,senti  n: no of words  addFeat=additional features
"""
xtrain,ytrain,xtest,ytest=main(ctype='normal',ftype='normal',n=50,addFeat=True)



#a,b,c=main(ctype='normal',ftype='normal',n=100,addFeat=False)
#vocab,corpus=getTop_n_words(100)
#x,vectorizer=get_linguistic_feature_vector(corpus,vocab)
##fv=pickle.load(open(pklfile))
#fourTupleNormalFeatures,fourTupleSentiFeatures =main()

#generate_review_corpus(imp_business_ids,Rgenerator)
#word_freq=calc_imp_word_frequencies(generator)
#tagged=create_linguistic_features(generator.next()['text'])
#analyze_total_reviews(generator)
#main()
"""
'Good For Groups','Good for Kids','Has TV','Noise Level','Outdoor Seating','Price Range','Noise Level','stars'
50749
'Bagels','Bakeries','Barbeque','Bars','Beer', 'Wine & Spirits','Breakfast & Brunch','Buffets','Bubble Tea','Burgers','Cafes','Cafeteria','Chicken Wings','Cheesesteaks','Coffee & Tea','Comfort Food','Creperies','Cupcakes','Delis','Desserts','Dim Sum','Diners','Donuts','Ethnic Food','Fast Food','Fish & Chips','Food','Food Court','Food Trucks','Food Stands','Gelato','Gluten-Free','Halal','Ice Cream & Frozen Yogurt','Hot Dogs','Juice Bars & Smoothies','Live/Raw Food','Pizza','Pita','Pretzels','Restaurants','Salad','Sandwiches','Seafood','Soul Food','Soup','Steakhouses','Sushi Bars','Szechuan','Tapas Bars','Tapas/Small Plates','Tex-Mex','Vegan','Vegetarian','Wine Bars'
'JJ','JJR','JJS','RB','RBR','RBS','UH'
"""
