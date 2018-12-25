# PBDP_LAB4
### 实验详情见实验报告，此处为文件说明
- knn:
   - test.py:用于测试数据集预处理，读取特征词文件chi_word.txt和测试数据集fulldata.txt，计算tfidf值向量化并输出至
   testData.txt文件
   - train.py:用于训练集数据预处理，读取特征词文件chi_words.txt和训练集数据training_data.txt，计算tfidf值向量化并输出至
   trainData.txt
   - KNN0.java:利用knn算法，读取trainData.txt和testData.txt数据，并行化进行数据训练和测试机数据分类预测，输出分类结果
   - knn.jar:KNN0.java程序jar包，用于hadoop运行
   - part-r-00000:词文件为分类部分结果文件，全结果文件过大
- naivebayes
   - pre
      - conf.py:生成所需配置文件，读取特征词文件chi_word.txt生成包含类别和特征属性的配置文件NBayes.conf
      - test.py:用于测试数据集预处理，读取特征词文件chi_word.txt和测试数据集fulldata.txt，计算tfidf值向量化并输出至
   NBayes.test文件
      - train.py:用于训练集数据预处理，读取特征词文件chi_words.txt和训练集数据training_data.txt，计算tfidf值向量化并输出至
   NBayes.train文件
   - src:朴素贝叶斯算法源码文件夹，利用朴素贝叶斯算法，读取测试集和训练集向量化数据及配置文件，并行化进行数据训练和测试集数据分类预测，
   输出分类结果
   - nbm.jar:用于hadoop运行的jar包
   - part-r-00000:词文件为分类部分结果文件，全结果文件过大
- lab4实验报告161278050.pdf:实验报告
