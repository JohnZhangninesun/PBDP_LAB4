# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 23:27:41 2018

@author: 张旭
"""
import numpy as np
import jieba

if __name__=='__main__':
    #得到特征向量
    f=open("chi_words.txt",encoding="utf-8")
    word1=f.read()
    words=np.array(word1.split('\n'))
    f.close()
    labels=['negative','neutral','positive']
    num=[471,512,517]
    label_num=0
    f1=open('trainData.txt','w')
    for label in labels:
        docs_list=[]
        for i in range(0,num[label_num]):
            f=open('training_data/'+label+'/'+str(i)+'.txt')
            str1=f.read()
            str1=str1.replace(' ','')
            doc_list=list(jieba.cut(str1,cut_all=True))
            doc1=''
            for doc in doc_list:
                doc1=doc1+' '+doc
            docs_list.append(doc1)
            f.close()
        docs=np.array(docs_list)
        #read file 
        print('read file\n')
        
        cfs = []
        for e in docs:
           cf = [e.count(word) for word in words]
           cfs.append(cf)
        cfs = []
        cfs.extend([e.count(word) for word in words] for e in docs)
        cfs = np.array(cfs)
        #cfs
        print("cfs\n")
        tfs = []
        for e in cfs:
            tf = e/(np.sum(e))
            tfs.append(tf)
        tfs = []
        tfs.extend(e/(np.sum(e)) for e in cfs)#不能使用append()
        #tfs
        print('tfs\n')
        dfs = list(np.zeros(words.size, dtype=int))
        for i in range(words.size):
            for doc in docs:
                if doc.find(words[i]) != -1:
                    dfs[i] += 1
        dfs = []
        for i in range(words.size):
            oneHot = [(doc.find(words[i]) != -1 and 1 or 0) for doc in docs]        
            dfs.append(oneHot.count(1))
        
        
        dfs, oneHots = [],[]
        for word in words:
            oneHots.append([(e.find(word) != -1 and 1 or 0) for e in docs])
        dfs.extend(e.count(1) for e in oneHots)
        
        
        dfs = []
        oneHots = [[doc.find(word) != -1 and 1 or 0 for doc in docs] for word in words]
        dfs.extend(e.count(1) for e in oneHots)
        #dfs
        
        N = np.shape(docs)[0]
        idfs = [(np.log10(N*1.0/(1+e))) for e in dfs]#f(e) = np.log10(N*1.0/(1+e))
        #idfs
        print("idfs\n")
        tfidfs = []
        for i in range(np.shape(docs)[0]):
            word_tfidf = np.multiply(tfs[i], idfs)
            tfidfs.append(word_tfidf)
        #tfidf
        print("tfidf")
        for a in tfidfs:
            for b in a:
                f1.write(str(b)+' ')
            f1.write(str(label_num)+' \n')
        label_num+=1
        #write
        print("finish")
    print("==========end=========")
    f1.close()