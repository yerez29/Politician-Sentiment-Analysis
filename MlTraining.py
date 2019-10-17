"""
Made by: Yair Erez.
This file contain code for Machine Learning classifiers training on twitter politics related comments, later by used
to analyze nyt readers comments (in "Ml_Sentiment.py" file).
NOTE: TO RUN THIS CODE "trainData.csv" file NEED TO BE LOCATED IN WORKING DIRECTORY.
This program execution requires 64-bit python interpreter version and at least 16GB of RAM memory to run.
To make the classifiers available in hardware weaker computers, they can be created in a one time run by a strong one
and will be saved in the ".pickle" files for further use in the future.
Once classifiers has been created, no need to run this program again unless want to train them all over again.
All ".pickle" files have already created by us and will be handed over in "code_group20.zip" in moodle and will be
located in CODE directory within the project directory in google drive.
Just make sure that ".pickle" files of all classifiers and word features will be located in "Ml_Sentiment.py" working
directory before you run it.
For running this file make sure "trainData.csv" file is in your working directory.
"""

# IMPORTS

import nltk
import pandas as pd
from random import shuffle
from nltk.classify import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from nltk.tokenize import word_tokenize
from collections import Counter

# CONSTANTS

POSITIVE_STARTING_IDX = 799999
NUMBER_OF_POS_AND_NEG_COMMENTS = 15000
TRAIN_TEST_RATIO = 0.8
POS = "pos"
NEG = "neg"
TAG_SIGN = "@"

# parsing relevant data from trainData.csv file contains 1,600,000 politics related tweets with positive/negative tags
df = pd.read_csv("trainData.csv", encoding="latin")
col_name = df.columns[0]
df = df.rename(columns={col_name: 'target'})
col_name = df.columns[5]
df = df.rename(columns={col_name: 'text'})
target_column = df.target
text_column = df.text
allWordsList = []
commentsAndTags = []
negative_comments = []
positive_comments = []
# prepare 2 lists with positive and negative comments
for i in range(NUMBER_OF_POS_AND_NEG_COMMENTS):
    negative_comments.append(text_column[i])
    positive_comments.append(text_column[POSITIVE_STARTING_IDX + i])
# define part of speech we are interested in
allowed_word_types = ["JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
# pre-processing steps
for comment in positive_comments:  # prepare positive comments words that belong to defined POS
    commentsAndTags.append((comment, POS))
    words = word_tokenize(comment)
    partOfSpeech = nltk.pos_tag(words)
    for w in partOfSpeech:
        if w[1] in allowed_word_types:
            allWordsList.append((w[0]).lower())  # transfer all words to lowercase

for comment in negative_comments:
    commentsAndTags.append((comment, NEG))  # prepare negative comments words that belong to defined POS
    words = word_tokenize(comment)
    partOfSpeech = nltk.pos_tag(words)
    for w in partOfSpeech:
        if w[1] in allowed_word_types:
            allWordsList.append((w[0]).lower())  # transfer all words to lowercase
allWordsList = [x for x in allWordsList if x != TAG_SIGN]
# save comments and their tags
savedComments = open("commentsTags.pickle", "wb")
pickle.dump(commentsAndTags, savedComments)
savedComments.close()
# take most common 10,000 words
wordsCounting = Counter(allWordsList)
topWordsList = wordsCounting.most_common(10000)
# prepare words features
featuresWords = [i[0] for i in topWordsList]
# save word features
savedFeatures = open("featuresWords.pickle", "wb")
pickle.dump(featuresWords, savedFeatures)
savedFeatures.close()


def featuresSearch(comment):
    """
    This function returns a dictionary as features of current comment.
    :param comment: current comment to get features from.
    :return: a dictionary with keys and values as features.
    """
    commentWords = word_tokenize(comment)
    featuresDic = {}
    for word in featuresWords:
        featuresDic[word] = word in commentWords
    return featuresDic


# prepare list of tupples with features to train the classifiers
featuresContainer = [(featuresSearch(comm), tag) for (comm, tag) in commentsAndTags]
# shuffle list of tupples to get positive and negative independent random comments
shuffle(featuresContainer)
# take 80% as training data and 20% as testing data
testingData = featuresContainer[int(TRAIN_TEST_RATIO * (NUMBER_OF_POS_AND_NEG_COMMENTS * 2)):]
trainingData = featuresContainer[:int(TRAIN_TEST_RATIO * (NUMBER_OF_POS_AND_NEG_COMMENTS * 2))]
# train naive bayes classifier
naiveBayesClassifier = nltk.classify.NaiveBayesClassifier.train(trainingData)
# print Naive Bayes accuracy
print("Naive Bayes accuracy in percent:",
      (nltk.classify.util.accuracy(naiveBayesClassifier, testingData)) * 100)
# save trained naive bayes classifier
classifier_to_save = open("naiveBayes.pickle", "wb")
pickle.dump(naiveBayesClassifier, classifier_to_save)
classifier_to_save.close()
# train multinomial naive bayes classifier
multinomial_naive_bayes_classifier = SklearnClassifier(MultinomialNB())
multinomial_naive_bayes_classifier.train(trainingData)
# print multinomial Naive Bayes classifier accuracy
print("multinomial naive bayes accuracy in percent:",
      (nltk.classify.util.accuracy(multinomial_naive_bayes_classifier, testingData)) * 100)
# save trained multinomial naive bayes classifier
classifier_to_save = open("multiNaiveBayes.pickle", "wb")
pickle.dump(multinomial_naive_bayes_classifier, classifier_to_save)
classifier_to_save.close()
# train Bernoulli naive bayes classifier
bernoulli_naive_bayes_classifier = SklearnClassifier(BernoulliNB())
bernoulli_naive_bayes_classifier.train(trainingData)
# print Bernoulli Naive Bayes classifier accuracy
print("Bernoulli naive bayes accuracy in percent",
      (nltk.classify.util.accuracy(bernoulli_naive_bayes_classifier, testingData)) * 100)
# save trained Bernoulli naive bayes classifier
classifier_to_save = open("bernoulliNaiveBayes.pickle", "wb")
pickle.dump(bernoulli_naive_bayes_classifier, classifier_to_save)
classifier_to_save.close()
# train Logistic Regression classifier
logistic_regression_classifier = SklearnClassifier(LogisticRegression())
logistic_regression_classifier.train(trainingData)
# print Logistic Regression classifier accuracy
print("Logistic Regression accuracy in percent",
      (nltk.classify.util.accuracy(logistic_regression_classifier, testingData)) * 100)
# save trained Logistic regression classifier
classifier_to_save = open("logisticRegression.pickle",
                       "wb")
pickle.dump(logistic_regression_classifier, classifier_to_save)
classifier_to_save.close()
# train Linear Support Vector classifier
linear_support_vector_classifier = SklearnClassifier(LinearSVC())
linear_support_vector_classifier.train(trainingData)
# print Linear Support Vector classifier accuracy
print("Linear Support Vector accuracy in percent:",
      (nltk.classify.util.accuracy(linear_support_vector_classifier, testingData)) * 100)
# save trained Linear Support Vector classifier
classifier_to_save = open("linearSv.pickle", "wb")
pickle.dump(linear_support_vector_classifier, classifier_to_save)
classifier_to_save.close()
