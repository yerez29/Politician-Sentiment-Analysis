# This program requires 64-bit python version and at least 16GB of ram to run.
# To make the classifiers available in weaker computers, they will be created by a strong one and will be saved in
# the ".pickle" files.
# No need to run this program as it takes large amount of time.
# Just make sure that ".pickle" files of all classifiers and word features will be located in "MlRunning.py" directory
# before you run it.


# For running this file make sure trainData.csv tagged dataset is in your working directory.


import nltk
import pandas as pd
from random import shuffle
from nltk.classify import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from collections import Counter
POSITIVE_STARTING_IDX = 799999
NUMBER_OF_POS_AND_NEG_COMMENTS = 15000
TRAIN_TEST_RATIO = 0.8
POS = "pos"
NEG = "neg"
TAG_SIGN = "@"


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


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
for i in range(NUMBER_OF_POS_AND_NEG_COMMENTS):
    negative_comments.append(text_column[i])
    positive_comments.append(text_column[POSITIVE_STARTING_IDX + i])
allowed_word_types = ["JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
for comment in positive_comments:
    commentsAndTags.append((comment, POS))
    words = word_tokenize(comment)
    partOfSpeech = nltk.pos_tag(words)
    for w in partOfSpeech:
        if w[1] in allowed_word_types:
            allWordsList.append((w[0]).lower())

for comment in negative_comments:
    commentsAndTags.append((comment, NEG))
    words = word_tokenize(comment)
    partOfSpeech = nltk.pos_tag(words)
    for w in partOfSpeech:
        if w[1] in allowed_word_types:
            allWordsList.append((w[0]).lower())
allWordsList = [x for x in allWordsList if x != TAG_SIGN]

save_documents = open("documents.pickle", "wb")
pickle.dump(commentsAndTags, save_documents)
save_documents.close()
c = Counter(allWordsList)
topWordsList = c.most_common(10000)
word_features = [i[0] for i in topWordsList]

save_word_features = open("word_features5k.pickle", "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


def find_features(comment):
    commentWords = word_tokenize(comment)
    featuresDic = {}
    for w in word_features:
        featuresDic[w] = w in commentWords
    return featuresDic


featureSets = [(find_features(comm), tag) for (comm, tag) in commentsAndTags]
shuffle(featureSets)
testing_set = featureSets[int(TRAIN_TEST_RATIO*(NUMBER_OF_POS_AND_NEG_COMMENTS * 2)):]
training_set = featureSets[:int(TRAIN_TEST_RATIO*(NUMBER_OF_POS_AND_NEG_COMMENTS * 2))]
classifier = nltk.classify.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:",
      (nltk.classify.util.accuracy(classifier, testing_set)) * 100)


save_classifier = open("originalnaivebayes5k.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:",
      (nltk.classify.util.accuracy(MNB_classifier, testing_set)) * 100)

save_classifier = open("MNB_classifier5k.pickle", "wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:",
      (nltk.classify.util.accuracy(BernoulliNB_classifier, testing_set)) * 100)

save_classifier = open("BernoulliNB_classifier5k.pickle", "wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:",
      (nltk.classify.util.accuracy(LogisticRegression_classifier, testing_set)) * 100)

save_classifier = open("LogisticRegression_classifier5k.pickle",
                       "wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:",
      (nltk.classify.util.accuracy(LinearSVC_classifier, testing_set)) * 100)

save_classifier = open("LinearSVC_classifier5k.pickle", "wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()
