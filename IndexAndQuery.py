import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from scipy import spatial
nltk.download('stopwords')
nltk.download('punkt')

with open('data.json', encoding='utf-8') as inputfile:
    df = pd.read_json(inputfile)

df.to_csv('csvfile.csv', encoding='utf-8', index=False)

data = pd.read_csv('csvfile.csv')

title = data['publication__name']
title.head()

punctuations = string.punctuation
stop_words = set(stopwords.words('english'))
filt_sentence = []
ful_sentence = []
filt_sentence1 = []
ful_sentence1 = []
invertedindex = {}


def vectorize_text(words_list, doc):
    vec=[]
    for item in words_list:
        if item in doc:
            vec.append(1)
        else:
            vec.append(0)
    return vec
        
        

def rank_docs(query, docs, inverted_index):
    scores = []
    ind_list = list(inverted_index.keys())
    query_vec = vectorize_text(ind_list, query)
    
    for doc in docs:
        doc_vec = vectorize_text(ind_list, doc)
        scores.append(1-spatial.distance.cosine(query_vec, doc_vec))
    
    arg_sor = np.argsort(scores)[::-1]
    sorted_scores = np.array(scores)[arg_sor]
    sorted_docs = np.array(docs)[arg_sor]
        
    return sorted_docs,sorted_scores

def retrieve_documents(inverted_index, query_tokens):
    retrieved_docs = []
    for token in query_tokens:
        match_docs = inverted_index.get(token)
        if match_docs != None:
            retrieved_docs.extend(list(match_docs))
            
    return list(set(retrieved_docs))

def pre_process_text(data):
      data = data.lower()
      data = "".join(character for character in data if character not in punctuations)
      wordtokens = word_tokenize(data)
      filt_sentence = ([w for w in wordtokens if not w.lower() in stop_words])
      
      return filt_sentence


for i in title:
    filt_tokens =  pre_process_text(i)
    ful_sentence.append((" ").join(filt_tokens))
    #print(ful_sentence)
    
    

    for i, doc in enumerate(ful_sentence):
        for term in doc.split():
            if term in invertedindex:
                invertedindex[term].add(i)
            else: invertedindex[term] = {i}
print(invertedindex)


query = "improve finance"
preprocessed_query = pre_process_text(query)
match_docs = retrieve_documents(invertedindex, preprocessed_query)

print(query)
docs = []
for doc in match_docs:
    docs.append(title[doc])
    
sorted_docs,sorted_scores = rank_docs(query, docs, invertedindex)
for doc, score in zip(sorted_docs, sorted_scores):
    print(doc, score)


print("---------------------------")
query = "finance"
preprocessed_query = pre_process_text(query)
match_docs = retrieve_documents(invertedindex, preprocessed_query)
print(query)
docs = []
for doc in match_docs:
    docs.append(title[doc])
    
sorted_docs,sorted_scores = rank_docs(query, docs, invertedindex)
for doc, score in zip(sorted_docs, sorted_scores):
    print(doc, score)
    
print("---------------------------")
query = "improve and finance"
preprocessed_query = pre_process_text(query)
match_docs = retrieve_documents(invertedindex, preprocessed_query)
print(query)
docs = []
for doc in match_docs:
    docs.append(title[doc])
    
sorted_docs,sorted_scores = rank_docs(query, docs, invertedindex)
for doc, score in zip(sorted_docs, sorted_scores):
    print(doc, score)
    
print("---------------------------")
query = "abroad or four"
preprocessed_query = pre_process_text(query)
match_docs = retrieve_documents(invertedindex, preprocessed_query)
print(query)
docs = []
for doc in match_docs:
    docs.append(title[doc])
    
sorted_docs,sorted_scores = rank_docs(query, docs, invertedindex)
for doc, score in zip(sorted_docs, sorted_scores):
    print(doc, score)
    
print("---------------------------")
query = "write and learn"
preprocessed_query = pre_process_text(query)
match_docs = retrieve_documents(invertedindex, preprocessed_query)
print(query)
docs = []
for doc in match_docs:
    docs.append(title[doc])
    
sorted_docs,sorted_scores = rank_docs(query, docs, invertedindex)
for doc, score in zip(sorted_docs, sorted_scores):
    print(doc, score)


        