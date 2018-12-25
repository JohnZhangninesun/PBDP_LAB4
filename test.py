# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 22:40:25 2018

@author: 张旭
"""

import numpy as np
import jieba

if __name__=='__main__':
    
    f=open("fulldata.txt",encoding='utf-8')
    #f=open("fulldata2.txt")
    lines=f.read()
    line_list=lines.split('\n')
    docs_list=[]
    for line in line_list:
        content=line.split('\t')
        
        if len(content)==7:
            
            title=content[-3]
        else:
            line_list.remove(line)
            continue
        title_seg=list(jieba.cut(title,cut_all=True))
        title1=''
        for titles in title_seg:
            title1=title1+' '+titles
        docs_list.append(title1)
    
    docs=np.array(docs_list)
    f.close()
    print("chi_words.txt")
    #chi_words
    f=open("chi_words.txt",encoding="utf-8")
    word1=f.read()
    words=np.array(word1.split('\n'))
    f.close()

    cfs = []
    for e in docs:
       cf = [e.count(word) for word in words]
       cfs.append(cf)
    cfs = []
    cfs.extend([e.count(word) for word in words] for e in docs)
    cfs = np.array(cfs)
    #cfs
    print('cfs')

    tfs = []
    for e in cfs:
        tf = e/(np.sum(e))
        tfs.append(tf)
    tfs = []
    tfs.extend(e/(np.sum(e)) for e in cfs)
    #tfs
    print('tfs')
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
    N = np.shape(docs)[0]
    idfs = [(np.log10(N*1.0/(1+e))) for e in dfs]
    #idfs
    print('idfs')
    tfidfs = []
    for i in range(np.shape(docs)[0]):
        word_tfidf = np.multiply(tfs[i], idfs)
        tfidfs.append(word_tfidf)
    #tfidf
    print('tfidf')
    f=open('testData.txt','w',encoding="utf-8")
    num=0
    for a in tfidfs:
        if num<len(line_list):
            content=line_list[num].split('\t')
            num+=1
            if len(content)==7:
                title=content[-3].replace(' ','')
            f.write(title+' ')
            num+=1
            for b in a:
                f.write(str(b)+' ')
            f.write('-1 \n')
    f.close()
    #write file
    print('finished!\n')