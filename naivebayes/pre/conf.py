# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 21:56:29 2018

@author: 张旭
"""

if __name__=='__main__':
    f=open("chi_words.txt",encoding="utf-8")
    word1=f.read()
    words=word1.split('\n')
    f=open('NBayes.conf','w',encoding="utf-8")
    f.write('3 negative neutral positive\n')
    num = len(words)
    f.write(str(num))
    for word in words:
        f.write(' '+word+' 100')