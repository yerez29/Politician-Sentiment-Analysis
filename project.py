import csv
import sys
import nltk
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from math import ceil
from textblob.classifiers import NaiveBayesClassifier


def getArticles(keywords_list, articles_file_name):
    """
    :param wordLst:
    :param csv_file:
    :return:
    """

    relevant_articles_ids = []
    df = pd.read_csv(articles_file_name)
    keyword_column = df.keywords
    articles_ids_column = df.articleID
    for i in range(len(keyword_column)):
        for word in keywords_list:
            if word in keyword_column[i]:
                relevant_articles_ids.append(articles_ids_column[i])
                break
    return relevant_articles_ids


def getComments(keywords_articles_ids, comments_file_name):
    """
    :param articlsId:
    :return:
    """
    relevant_comments = []
    comments_indices = []
    df = pd.read_csv(comments_file_name)
    ids_column = df.articleID
    comments_column = df.commentBody
    num_of_ids = len(ids_column)
    for i in range(num_of_ids):
        if ids_column[i] in keywords_articles_ids:
            relevant_comments.append(comments_column[i])
            comments_indices.append(i)
    return relevant_comments, comments_indices

def percent(part, whole):

    return (part/whole) * 100

def getGeneralOpinions(relevant_comments, comments_file_name):
    """
    :param comments:
    :return:
    """
    train = []
    train_counter = 0
    general_mark = 0
    very_positive_mark = 0
    slightly_positive_mark = 0
    slightly_negative_mark = 0
    very_negative_mark = 0
    natural_mark = 0
    geo_location_marks = {}
    num_of_comments = len(relevant_comments)
    num_of_trains = ceil(num_of_comments / 100)
    date = comments_file_name[8:len(comments_file_name) - 4]
    for comment in relevant_comments:
        analysis = TextBlob(comment)
        cur_average_mark = analysis.sentences[0].sentiment.polarity
        general_mark += cur_average_mark
        if cur_average_mark > 0.00:
            if cur_average_mark > 0.3:
                very_positive_mark += 1
                if train_counter <= num_of_trains:
                    train.append((str(analysis.sentences[0]), "very pos"))
            else:
                slightly_positive_mark += 1
                if train_counter <= num_of_trains:
                    train.append((str(analysis.sentences[0]), "slightly pos"))
        elif cur_average_mark < 0.00:
            if cur_average_mark < -0.3:
                very_negative_mark += 1
                if train_counter <= num_of_trains:
                    train.append((str(analysis.sentences[0]), "very neg"))
            else:
                slightly_negative_mark += 1
                if train_counter <= num_of_trains:
                    train.append((str(analysis.sentences[0]), "slightly neg"))
        else:
            natural_mark += 1
            if train_counter <= num_of_trains:
                train.append((str(analysis.sentences[0]), "natural"))
        train_counter += 1
    general_mark /= num_of_comments
    print("total normalized mark in", date, "is", general_mark)
    very_positive_mark = percent(very_positive_mark, num_of_comments)
    slightly_positive_mark = percent(slightly_positive_mark, num_of_comments)
    slightly_negative_mark = percent(slightly_negative_mark, num_of_comments)
    very_negative_mark = percent(very_negative_mark, num_of_comments)
    natural_mark = percent(natural_mark, num_of_comments)
    labels = 'Very positive', 'Slightly positive', 'Slightly negative', 'Very negative', 'Natural'
    sizes = [very_positive_mark, slightly_positive_mark, slightly_negative_mark, very_negative_mark, natural_mark]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.title("Opinion distribution in " + date)
    plt.axis('equal')
    # plt.tight_layout()
    plt.show()
    plt.clf()
    print(train)
    return train

def getLocationBasedOpinions(relevantComments, comments_indices):



def getClassificationsByMl(relevant_comments, comments_file_name, train):

    cl = NaiveBayesClassifier(train)
    for comment in relevant_comments:
        blob = TextBlob(comment)
        analysis = TextBlob(str(blob.sentences[0]), classifier=cl)
        print(str(blob.sentences[0]))
        print("\n")
        print(analysis.classify())
        print("\n")


keywords_list = ["Trump", 'Donald', "trump", "donald", "TRUMP", "DONALD", "Trump, Donald J", "Donald Trump", "Donald John Trump", "Donald J. Trump"]
# JAN 2017
articles_file_name = "ArticlesJan2017.csv"
keywords_articles_ids = getArticles(keywords_list, articles_file_name)
comments_file_name = "CommentsJan2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
train = getGeneralOpinions(relevant_comments, comments_file_name)
getLocationBasedOpinions(relevant_comments, comments_indices)
getClassificationsByMl(relevant_comments, comments_file_name, train)
# FEB 2017
articles_file_name = "ArticlesFeb2017.csv"
keywords_articles_ids = getArticles(keywords_list, articles_file_name)
comments_file_name = "CommentsFeb2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name)
getLocationBasedOpinions(relevant_comments, comments_indices)
# MARCH 2017
articles_file_name = "ArticlesMarch2017.csv"
keywords_articles_ids = getArticles(keywords_list, articles_file_name)
comments_file_name = "CommentsMarch2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name)
getLocationBasedOpinions(relevant_comments, comments_indices)
# APRIL 2017
articles_file_name = "ArticlesApril2017.csv"
keywords_articles_ids = getArticles(keywords_list, articles_file_name)
comments_file_name = "CommentsApril2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name)
getLocationBasedOpinions(relevant_comments, comments_indices)
# MAY 2017
articles_file_name = "ArticlesMay2017.csv"
keywords_articles_ids = getArticles(keywords_list, articles_file_name)
comments_file_name = "CommentsMay2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name)
getLocationBasedOpinions(relevant_comments, comments_indices)
# JAN 2018
articles_file_name = "ArticlesJan2018.csv"
keywords_articles_ids = getArticles(keywords_list, articles_file_name)
comments_file_name = "CommentsJan2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name)
getLocationBasedOpinions(relevant_comments, comments_indices)
# FEB 2018
articles_file_name = "ArticlesFeb2018.csv"
keywords_articles_ids = getArticles(keywords_list, articles_file_name)
comments_file_name = "CommentsFeb2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name)
getLocationBasedOpinions(relevant_comments, comments_indices)
# MARCH 2018
articles_file_name = "ArticlesMarch2018.csv"
keywords_articles_ids = getArticles(keywords_list, articles_file_name)
comments_file_name = "CommentsMarch2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name)
getLocationBasedOpinions(relevant_comments, comments_indices)
# APRIL 2018
articles_file_name = "ArticlesApril2018.csv"
keywords_articles_ids = getArticles(keywords_list, articles_file_name)
comments_file_name = "CommentsApril2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name)
getLocationBasedOpinions(relevant_comments, comments_indices)
