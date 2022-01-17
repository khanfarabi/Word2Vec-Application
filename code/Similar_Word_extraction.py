#Reading Data
import time
start = time.time()
import time
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from scipy import spatial
from numpy import dot
from numpy.linalg import norm
import numpy as np
import pandas as pd
import random
import sys
import random
import re
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
sto = set(stopwords.words("english"))
wordnet_lemmatizer = WordNetLemmatizer()
stopwords=[]
for k in sto:
    stopwords.append(k)

flags = (re.UNICODE if sys.version < '3' and type(text) is unicode else 0)

class data_preprocessing:
    @classmethod
    def hotel_data(cls):
                    hotel=pd.read_csv("Hotel_Reviews.csv")
                    revid=0
                    Words={}

                    cn=0
                    cn1=0
                    for k in hotel['Negative_Review']:
                        keep=[]
                        if cn<5000:
                                #print(k)
                                for word in re.findall(r"\w[\w']*", k, flags=flags):
                                    if word.isdigit() or len(word)==1:
                                        continue
                                    word_lower = word.lower()
                                    if word_lower in stopwords:
                                            continue
                                    if not any(c.isdigit() for c in word_lower) and "'" not in word_lower:
                                        if word_lower.isdigit():
                                            continue
                                        else:
                                             if len(word_lower)>=4:
                                                    keep.append(word_lower)
                                if len(keep)>=25:
                                        Words[revid]=keep
                                        revid=revid+1

                                cn=cn+1
                    for k in hotel['Positive_Review']:
                        keep=[]
                        if cn1<5000:
                                #print(k)
                                for word in re.findall(r"\w[\w']*", k, flags=flags):
                                    if word.isdigit() or len(word)==1:
                                        continue
                                    word_lower = word.lower()
                                    if word_lower in stopwords:
                                            continue
                                    word1 = wordnet_lemmatizer.lemmatize(word_lower, pos = "n")
                                    word2 = wordnet_lemmatizer.lemmatize(word1, pos = "v")
                                    word3 = wordnet_lemmatizer.lemmatize(word2, pos = ("a"))
                                    if not any(c.isdigit() for c in  word3 ) and "'" not in word3:
                                        if  word3 .isdigit():
                                            continue
                                        else:
                                             if len(word3)>3:
                                                    keep.append(word3)
                                if len(keep)>=25:
                                        Words[revid]=keep
                                        revid=revid+1

                                cn1=cn1+1
                                
                    return Words
    @classmethod
    def movie_data(cls):
                    f22=pd.read_csv("IMDB Dataset.csv")
                    Words1={}
                    rid=0
                    cn2=0
                    for k in f22['review']:
                        keep=[]
                        if cn2<1000:
                                #print(k)
                                for word in re.findall(r"\w[\w']*", k, flags=flags):
                                    if word.isdigit() or len(word)==1:
                                        continue
                                    word_lower = word.lower()
                                    if word_lower in stopwords:
                                            continue
                                    word1 = wordnet_lemmatizer.lemmatize(word_lower, pos = "n")
                                    word2 = wordnet_lemmatizer.lemmatize(word1, pos = "v")
                                    word3 = wordnet_lemmatizer.lemmatize(word2, pos = ("a"))
                                    if not any(c.isdigit() for c in  word3 ) and "'" not in word3:
                                        if  word3 .isdigit():
                                            continue
                                        else:
                                             if len(word3)>3:
                                                    keep.append(word3)
                                if len(keep)>=10 and len(keep)<=40:
                                        Words1[rid]=keep
                                        rid=rid+1

                                cn2=cn2+1  
                
                    return Words1
                
                
    @classmethod
    def sent_generation(cls,Words):
            sent=[]                         
            for k in Words:
                gh=[]
                jj=str(k)
                gh.append(jj)
                for v in Words[k]:
                    gh.append(v)
                sent.append(gh)
            return sent
    @classmethod
    #K-Means 
    def cluster(cls,Words):
                    #cluster generation with k-means
                    import sys
                    from nltk.cluster import KMeansClusterer
                    import nltk
                    from sklearn import cluster
                    from sklearn import metrics
                    import gensim 
                    import operator
                    from gensim.models import Word2Vec

                    #Words=hotel_d

                    model = Word2Vec(Words, min_count=1)

                    X = model[model.wv.vocab]



                    NUM_CLUSTERS=10
                    kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance,repeats=25)
                    assigned_clusters1 = kclusterer.cluster(X,assign_clusters=True)
                    #print (assigned_clusters)
                    cluster={}
                    words = list(model.wv.vocab)
                    for i, word in enumerate(words):
                      gh=[] 
                      gh1=[] 
                      gh2=[] 
                      if word.isdigit(): 
                        cluster[word]=assigned_clusters1[i]
                        #print (word + ":" + str(assigned_clusters[i]))
                    cluster_final={}
                    for j in range(NUM_CLUSTERS):
                        gg=[]
                        for tt in cluster:
                            if int(cluster[tt])==int(j):
                                if tt not in gg:
                                    gg.append(tt)
                        if len(gg)>0:
                                    cluster_final[j]=gg
                    cc=0
                    final_clu={}
                    for t in cluster_final:
                        ghh=[]
                        for k in cluster_final[t]:
                            if int(k) in Words:
                                   ghh.append(int(k))
                        if len(ghh)>=2:
                                final_clu[cc]=ghh
                                cc=cc+1
                    return final_clu





#Find the similar words from the input data using user input
class similar_phrase:
    @classmethod
    def same_words(cls,sent,m,Words):
                        print("Enter the input phrase for "+m)
                        input_text1=input()
                        vc=[input_text1]
                        sent.append(vc)
                        model = Word2Vec(sent, min_count=1,iter=100)
                        #result = model.most_similar(positive=[input_text1],  topn=20)
                        #for t in result:
                            #if t[0].isdigit():
                                #continue
                          #  else:
                                 # print(t)
                        
                        unique_words=[]
                        ss=[]
                        for t in Words:
                            for kk in Words[t]:
                                if kk not in ss:
                                    ss.append(kk)
                        s=set(ss)
                        for v in s:
                            unique_words.append(v)
                        wrdvec={}
                        for vb in unique_words:
                            wrdvec[vb]=model[vb]
                        phrase_in={}
                        phrase_in[input_text1]=model[input_text1]

                        wrd_sim=[]
                        dc={}
                        for bb in unique_words:
                            vc1=wrdvec[bb]
                            vc2=phrase_in[input_text1]
                            #sm=1 - spatial.distance.cosine(vc1, vc2)
                            sm1=dot(vc1,vc2)/(norm(vc1)*norm(vc2))
                            dc[bb]=sm1
                        import operator
                        sorted_sim_map1 = sorted(dc.items(), key=operator.itemgetter(1),reverse=True)
                        cv=0
                        for vc in  sorted_sim_map1:
                            if cv<20:
                                wrd_sim.append(vc[0])
                                cv=cv+1
                        #print(wrd_sim)
                        return wrd_sim
    @classmethod
    def same_words_cl(cls,sent,m,Words1,cl):
                        print("Enter the input phrase for "+m)
                        input_text1=input()
                        vc=[input_text1]
                        Words={}
                        for t in Words1:
                            if str(t) in cl or int(t) in cl:
                                Words[t]=Words1[t]
                        sent=[]                         
                        for k in Words:
                            gh=[]
                            jj=str(k)
                            gh.append(jj)
                            for v in Words[k]:
                                gh.append(v)
                            sent.append(gh)
                        sent.append(vc)
                        model = Word2Vec(sent, min_count=1,iter=100)
                        #result = model.most_similar(positive=[input_text1],  topn=20)
                        #for t in result:
                            #if t[0].isdigit():
                                #continue
                          #  else:
                                 # print(t)
                        unique_words=[]
                        ss=[]
                        for t in Words:
                            for kk in Words[t]:
                                if kk not in ss:
                                    ss.append(kk)
                        s=set(ss)
                        for v in s:
                            unique_words.append(v)
                        wrdvec={}
                        for vb in unique_words:
                            wrdvec[vb]=model[vb]
                        phrase_in={}
                        phrase_in[input_text1]=model[input_text1]

                        wrd_sim=[]
                        dc={}
                        for bb in unique_words:
                            vc1=wrdvec[bb]
                            vc2=phrase_in[input_text1]
                            #sm=1 - spatial.distance.cosine(vc1, vc2)
                            sm1=dot(vc1,vc2)/(norm(vc1)*norm(vc2))
                            dc[bb]=sm1
                        import operator
                        sorted_sim_map1 = sorted(dc.items(), key=operator.itemgetter(1),reverse=True)
                        cv=0
                        for vc in  sorted_sim_map1:
                            if cv<20:
                                wrd_sim.append(vc[0])
                                cv=cv+1
                        #print(wrd_sim)
                        return wrd_sim
    



                        #1 - spatial.distance.cosine(vector1, vector2)
                        
#mwords=similar_phrase.same_words(movie_sent,"movie",movie_d)
#print("Movie Review")
#print(mwords)
print("Enter the option: For the hotel review data : hotel : For the movie review data : movie")
import sys
LL=list(sys.argv[1:])

if LL[0]=='hotel':
        hotel_d=data_preprocessing.hotel_data()
        hotel_sent=data_preprocessing.sent_generation(hotel_d)

        #hotel_cluster=data_preprocessing.cluster(hotel_d)
        #movie_cluster=data_preprocessing.cluster(movie_d)
        hwords=similar_phrase.same_words(hotel_sent,"hotel",hotel_d)
        print("Hotel Review")
        print(hwords)
elif LL[0]=='movie':
        movie_d=data_preprocessing.movie_data()
        movie_sent=data_preprocessing.sent_generation(movie_d)
        mwords=similar_phrase.same_words(movie_sent,"movie",movie_d)
        print("Movie Review")
        print(mwords)
#for k in hotel_cluster:
    #pass#hwords=similar_phrase.same_words_cl(hotel_sent,"hotel",hotel_d,hotel_cluster[k])
   # pass#print("Hotel Review"+"cluster_"+str(k))
    #pass#print(hwords)

end = time.time()
print(end - start)