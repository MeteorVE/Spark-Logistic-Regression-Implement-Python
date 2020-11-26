# -*- coding: utf-8 -*-
# Add Spark Python Files to Python Path
import sys
import os
SPARK_HOME = "/opt/bitnami/spark" # Set this to wherever you have compiled Spark
os.environ["SPARK_HOME"] = SPARK_HOME # Add Spark path
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1" # Set Local IP
sys.path.append( SPARK_HOME + "/python") # Add python files to Python Path

import pyspark
from pyspark.mllib.classification import LogisticRegressionWithSGD
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint # added


sc = pyspark.SparkContext()

# ===================================

# Logistic Regression on Diabetes Dataset
from random import seed
from random import randrange
from csv import reader
from math import exp
 
# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())
 
# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
 
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)

    # example: dataset[ 10000 data ], fold= 5, fold_size= 2000, output= [[ [a,b,c,d,e]*2000 ], [ ... ] ... ]
    return dataset_split
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
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

        predicted = algorithm(train_set_rdd, test_set_rdd, *args) # 回傳新的東西

        accuracy = predicted.filter(lambda v: v[0] == v[1]).count() / float(predicted.count()) * 100.0

        scores.append(accuracy)
    return scores
 
# Make a prediction with coefficients
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i + 1] * row[i]
    return 1.0 / (1.0 + exp(-yhat))

def para_epoch(row, label, l_rate, n_epoch):
    yhat = predict(row, coef)
    error = label - yhat
    return l_rate * error * yhat * (1.0 - yhat)
    

# Estimate logistic regression coefficients using stochastic gradient descent
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


# Linear Regression Algorithm With Stochastic Gradient Descent
def logistic_regression(train, test, l_rate, n_epoch):
    predictions = list()
    coef = coefficients_sgd(train, l_rate, n_epoch) # get coef, [ 4個係數 ]

    # rdd_list of pair
    pred_rdd = test.map(lambda point: (point.label, round( predict(point.features, coef) )  )) 
    return pred_rdd
 
# Test the logistic regression algorithm on the diabetes dataset
seed(1)
# load and prepare data
filename = 'pima-indians-diabetes.csv'
filename = 'data_banknote_authentication.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)
# normalize
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)


# evaluate algorithm
n_folds = 5
l_rate = 0.1
n_epoch = 100
scores = evaluate_algorithm(dataset, logistic_regression, n_folds, l_rate, n_epoch) #(dataset, algorithm, n_folds, *args)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))