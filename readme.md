---
title: Implement logistic regression with SGD in Spark (Python)
date: 2020-11-26 00:00:00
tags: 
- Spark
categories: [ ]
keywords:  
- 
description: ""
---

<!-- more -->



# 測試 Spark 環境

使用授課教授給的 image 會有各種神奇問題。([hadoop-cluster by sdwangntu](https://github.com/sdwangntu/hadoop-cluster))
連最基本的 ``spark-shell`` 指令，都能跑出錯誤或是時常與 master 斷線 ...。
也需另外設定 ``export PYTHONPATH=$SPARK_HOME/python/lib/py4j-0.10.7-src.zip:$PYTHONPAT``

尤其該 Image 檔案偏大，若在 Windows Docker 又有 bug 發生，會直接把環境搞壞 
(ex: vdisk 直接暴漲到 300G、Ram 占用 50G 以上)

另外，也有嘗試使用 [big-data-europe/**docker-spark**](https://github.com/big-data-europe/docker-spark)
該 Image 的特性是 Non-root 的系統 (可以更改 docker run 參數去強制 root 身分登入)，
如果想要裝第三方的 python 套件得另外根據另一個 dockerfile 去 build image (build 時準備 requirements.txt)

最大的痛點是即使用了 root 去登入，也沒有內建 apt 之類的套件管理系統，就直接轉戰其他 repo 了。



最後是在 [bitnami/**bitnami-docker-spark**](https://github.com/bitnami/bitnami-docker-spark) 這個 image 去做嘗試，非常親民，該有的都有
而且預設登入位置即是 spark 目錄，第一次來也能直接確認 sprak 的絕對位置在哪邊。

測試的部分，執行 ``spark-shell``

![](https://i.imgur.com/t2UFlWl.png)

乾乾淨淨沒有 error

另外也可以跑測試程式 (來源 : [Apache Spark Examples](http://spark.apache.org/examples.html))
這邊拿最簡單的 WordCount 做舉例

```python Wordcount.py
text_file = sc.textFile("hdfs://...")
counts = text_file.flatMap(lambda line: line.split(" ")) \
             .map(lambda word: (word, 1)) \
             .reduceByKey(lambda a, b: a + b)
counts.saveAsTextFile("hdfs://...")
```

跑完可以好好產生結果就代表沒問題。(用途 : 統計文章內各字詞出現次數。)

上述的 Code 可能需要搭配一些環境設定的程式碼，舉例我這邊的 : 

```python Wordcount.py
# Add Spark Python Files to Python Path
import sys
import os
SPARK_HOME = "/opt/bitnami/spark" # Set this to wherever you have compiled Spark
os.environ["SPARK_HOME"] = SPARK_HOME # Add Spark path
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1" # Set Local IP
sys.path.append( SPARK_HOME + "/python") # Add python files to Python Path

import pyspark
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark import SparkConf, SparkContext

def getSparkContext():
    """
    Gets the Spark Context
    """
    conf = (SparkConf()
         .setMaster("local") # run on local
         .setAppName("Logistic Regression") # Name of App
         .set("spark.executor.memory", "1g")) # Set 1 gig of memory
    sc = SparkContext(conf = conf) 
    return sc

sc = pyspark.SparkContext()
url= './tester.txt'
opt_file = 'output'
text_file = sc.textFile(url)
counts = text_file.flatMap(lambda line: line.split(" ")) \
             .map(lambda word: (word, 1)) \
             .reduceByKey(lambda a, b: a + b)
counts.saveAsTextFile(opt_file)
```

這中間有一句 ``import pyspark``，這句要看環境，如果不用加這個他就能找到，那就不用加。



# 程式撰寫

目標 :  Write SGD function of logistic regression. 

相關閱讀 : Google : ``logistic regression with sgd implement python``
相關參考 : [Reference](#Ref)

門外漢長話短說 : 透過梯度下降法去慢慢逼近理想的值(步伐隨機)。搭配著公式去調整和使用。

只寫一般版本的 Logistic regression implement 參考那些網頁就好，但是要搭配 Spark 就得另外理解資料結構。

- Spark 內的 LabeledPoint Class:
  -  一個 Pair，前面是 Label, 後面是 Features。
- Spark 內的 RDD Type:  
  - 一種資料結構，彈性分散式資料集，有專屬的 API (Action API、Transformation API)來操作他。
  - 透過專屬 API 可以達成 Map / Reduce 的操作。甚至更多不一樣的做法。
  - .map() 內通常會搭配 lambda 去做使用。

![](https://i.imgur.com/DRo2Fy6.png)


## 用內建的 LogisticRegressionWithSGD 之測試

根據題目要求之教學文章的程式，經由修改部分內容，詳細如下
From [Logistic Regression in Apache Spark](https://www.hackerearth.com/practice/notes/samarthbhargav/logistic-regression-in-apache-spark/)

- 只擷取部分
- 主要更動內容為，將 numpy 更改為 LabeledPoint 來讓 LogisticRegressionWithSGD train。
- lambda 吃參數時，不知道是不是 python3 的特性，不能用 ``lambda a,b: a+b`` 這樣的寫法。
  要改成 ``lambda a: a[0]+a[1]``，這是個大坑。

```python
def mapper(line):
    feats = line.strip().split(",") 
    label = feats[len(feats) - 1] 
    feats = feats[: len(feats) - 1]
    features = [ float(feature) for feature in feats ] # need floats
    return LabeledPoint(label, features) # FIXED

sc = pyspark.SparkContext()

data = sc.textFile("./data")
parsedData = data.map(mapper)

# Train model
model = LogisticRegressionWithSGD.train(parsedData)

labelsAndPreds = parsedData.map(lambda point: (int(point.label), 
        model.predict(point.features)))

# Evaluating the model on training data
trainErr = labelsAndPreds.filter(lambda v: v[0] != v[1]).count()/ float(parsedData.count())
print("Training Error = " + str(trainErr))
```





## 關於決定哪些部分需做 Map / Reduce

可以平行處理的就可以做 map。

雖然說在計算 SGD 的 coefficients (Weight) 時，是有機會在以下部分做平行 : 

```python
def some_part_of_epoch(data, label, l_rate, n_epoch):
    yhat = predict(data, coef)
    error = label - yhat
    return l_rate * error * yhat * (1.0 - yhat)
```

但是，coef (coefficients) 理論上是要經過**每一筆** data 一筆一筆去調整(重新計算找到更好的)
如果將這個 part 做平行處理，在做 predict() 時，有可能會發生 coef 讀取時讀到舊的 coef
或是亦可能在 coef 這個 variable 發生 race condition (這還是得看 OS 如何去維護)

(coef 讀取時讀到舊的 coef: 假設兩個 thread  A, B，A 讀取 coef -> B 讀取 coef -> A 寫入 coef -> B 寫入 coef 
 等同 A 這筆 data 沒有作用)

故，此場景下較適合交給 RDD 同時處理的部分為 Testing 的部分
因為 Training 是在幫助演化 coefficients，而 Testing 是使用同一組 coefficients 在做處理。

```python
# Linear Regression Algorithm With Stochastic Gradient Descent
def logistic_regression(train, test, l_rate, n_epoch):
    predictions = list()
    coef = coefficients_sgd(train, l_rate, n_epoch)

    # rdd_list of pair
    pred_rdd = test.map(lambda point: (point.label, round( predict(point.features, coef) ))) 
    return pred_rdd
```



## 其餘處理部分

### 最初需做標準化

```python
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
```

### Predict Function

```python
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i + 1] * row[i]
    return 1.0 / (1.0 + exp(-yhat))
```

### 演化 SGD coefficients  的 Function

```python
def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train.first().features))]

    for epoch in range(n_epoch):
        for row in train.collect():
            yhat = predict(row.features, coef)
            error = row.label - yhat
            coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
            for i in range(len(row.features)-1):
                coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row.features[i]
    return coef 
```

### 處理整個估計流程的 Function

包括切割訓練集、分組、轉換成 RDD。

```python
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds) # copy [A, B, C, D, E]
        train_set.remove(fold)  # [B, C, D, E]
        
        train_set = sum(train_set, []) # [[1, 2] ,[3, 4]] combine to one array [1,2,3,4]
        train_set = [ LabeledPoint(t[-1], t[:-1]) for t in train_set]
        train_set_rdd = sc.parallelize(train_set)

        test_set = list()
        
        # fold is a rdd
        for row in fold:
            test_set.append(LabeledPoint(row[-1], list(row)[:-1]))
        # Convert list to RDD
        test_set_rdd = sc.parallelize(test_set)

        predicted = algorithm(train_set_rdd, test_set_rdd, *args)

        accuracy = predicted.filter(lambda v: v[0] == v[1]).count() / float(predicted.count()) * 100.0

        scores.append(accuracy)
    return scores
```

## Result

1. 根據題目要求，教學文章的程式執行結果
    From [Logistic Regression in Apache Spark](https://www.hackerearth.com/practice/notes/samarthbhargav/logistic-regression-in-apache-spark/)
    - 其執行結果，Error 約為 4.4%，正確率約為 95.5%。

![](https://i.imgur.com/QXPza8r.png)



2. 自製 implement 之程式執行結果
   - 此執行結果之準度約為 97.8%，Error 約為 2.2%  (0.021 ... )。

![](https://i.imgur.com/1Yd7rXq.png)




# Ref

[yarn logs -applicationId 无法导出logs日志 Log aggregation has not completed or is not enabled.](https://blog.csdn.net/qq_43688472/article/details/85251008)

- 這個在 debug Spark 環境簡直是必須。Log 預設是很雜亂的，要另外設定 conf 才能到目標 hdfs 上。

[Logistic Regression in Apache Spark ](https://www.hackerearth.com/practice/notes/samarthbhargav/logistic-regression-in-apache-spark/)

[機器學習自學筆記06: Logistic regression](https://medium.com/%E4%B8%89%E5%8D%81%E4%B8%8D%E5%93%AD/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E8%87%AA%E5%AD%B8%E7%AD%86%E8%A8%9806-logistic-regression-3c0dbc10400e)

[機器/深度學習-基礎數學(三):梯度最佳解相關算法(gradient descent optimization algorithms)](https://chih-sheng-huang821.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E5%9F%BA%E7%A4%8E%E6%95%B8%E5%AD%B8-%E4%B8%89-%E6%A2%AF%E5%BA%A6%E6%9C%80%E4%BD%B3%E8%A7%A3%E7%9B%B8%E9%97%9C%E7%AE%97%E6%B3%95-gradient-descent-optimization-algorithms-b61ed1478bd7)

[How To Implement Logistic Regression From Scratch in Python](https://machinelearningmastery.com/implement-logistic-regression-stochastic-gradient-descent-scratch-python/)

[samarthbhargav/spark-example](https://github.com/samarthbhargav/spark-example/blob/master/run-spark-ex.py)

[machine learning 下的 Logistic Regression 實作(使用python)](https://medium.com/jackys-blog/machine-learning-%E4%B8%8B%E7%9A%84-logistic-regression-%E5%AF%A6%E4%BD%9C-%E4%BD%BF%E7%94%A8python-d19b971ff9dc)

[pyspark.mllib package : LogisticRegressionWithSGD](https://spark.apache.org/docs/2.0.0/api/python/pyspark.mllib.html#pyspark.mllib.classification.LogisticRegressionWithSGD)

[《巨量資料技術與應用》實務操作講義- Spark簡易操作](http://debussy.im.nuu.edu.tw/sjchen/BigData/%E5%B7%A8%E9%87%8F%E8%B3%87%E6%96%99%E6%8A%80%E8%A1%93%E8%88%87%E6%87%89%E7%94%A8%E6%93%8D%E4%BD%9C%E8%AC%9B%E7%BE%A9-Spark%E6%93%8D%E4%BD%9C.html)





<div style="text-align: center">End</div>
-----------------------------------

![](https://i.imgur.com/888jFLr.gif)

<div style="text-align: right">2020.11.26</div>