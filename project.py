import csv
import sys
import nltk
import pandas as pd
from textblob import TextBlob
import matplotlib as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


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
    df = pd.read_csv(comments_file_name)
    ids_column = df.articleID
    comments_column = df.commentBody
    num_of_ids = len(ids_column)
    for i in range(num_of_ids):
        if ids_column[i] in keywords_articles_ids:
            relevant_comments.append(comments_column[i])
    return relevant_comments


def getMarks(relevant_comments, comments_file_name):
    """
    :param comments:
    :return:
    """
    general_mark = 0
    very_positive_mark = 0
    slightly_positive_mark = 0
    slightly_negative_mark = 0
    very_negative_mark = 0
    natural_mark = 0
    geo_location_marks = {}
    num_of_comments = len(relevant_comments)
    date = comments_file_name[8:len(comments_file_name) - 4]
    for comment in relevant_comments:
        blob = TextBlob(comment)
        comment_sentence_num = 0
        cur_mark = 0
        for sentence in blob.sentences:
            cur_mark += sentence.sentiment.polarity
            comment_sentence_num += 1
            break
        cur_average_mark = cur_mark / comment_sentence_num
        general_mark += cur_average_mark
        if cur_average_mark > 0.1:
            if cur_average_mark > 0.4:
                very_positive_mark += 1
            else:
                slightly_positive_mark += 1
        elif cur_average_mark < -0.1:
            if cur_average_mark < -0.4:
                very_negative_mark += 1
            else:
                slightly_negative_mark += 1
        else:
            natural_mark += 1
    general_mark /= num_of_comments
    print("very positive rate in", date, "is", (very_positive_mark / num_of_comments) * 100, "percent")
    print("slightly positive rate in", date, "is", (slightly_positive_mark / num_of_comments) * 100, "percent")
    print("slightly negative rate in", date, "is", (slightly_negative_mark / num_of_comments) * 100, "percent")
    print("very negative rate in", date, "is", (very_negative_mark / num_of_comments) * 100, "percent")
    print("natural rate in", date, "is", (natural_mark / num_of_comments) * 100, "percent")
    print("total normalized mark in", date, "is", general_mark)


keywords_list = ["Trump", 'Donald', "trump", "donald", "TRUMP", "DONALD", "Trump, Donald J", "Donald Trump", "Donald John Trump", "Donald J. Trump"]
# JAN 2017
articles_file_name = "ArticlesJan2017.csv"
keywords_articles_ids = getArticles(keywords_list, articles_file_name)
comments_file_name = "CommentsJan2017.csv"
relevant_comments = getComments(keywords_articles_ids, comments_file_name)
getMarks(relevant_comments, comments_file_name)
# FEB 2017
articles_file_name = "ArticlesFeb2017.csv"
keywords_articles_ids = getArticles(keywords_list, articles_file_name)
comments_file_name = "CommentsFeb2017.csv"
relevant_comments = getComments(keywords_articles_ids, comments_file_name)
getMarks(relevant_comments, comments_file_name)
# MARCH 2017
articles_file_name = "ArticlesMarch2017.csv"
keywords_articles_ids = getArticles(keywords_list, articles_file_name)
comments_file_name = "CommentsMarch2017.csv"
relevant_comments = getComments(keywords_articles_ids, comments_file_name)
getMarks(relevant_comments, comments_file_name)
# APRIL 2017
articles_file_name = "ArticlesApril2017.csv"
keywords_articles_ids = getArticles(keywords_list, articles_file_name)
comments_file_name = "CommentsApril2017.csv"
relevant_comments = getComments(keywords_articles_ids, comments_file_name)
getMarks(relevant_comments, comments_file_name)
# MAY 2017
articles_file_name = "ArticlesMay2017.csv"
keywords_articles_ids = getArticles(keywords_list, articles_file_name)
comments_file_name = "CommentsMay2017.csv"
relevant_comments = getComments(keywords_articles_ids, comments_file_name)
getMarks(relevant_comments, comments_file_name)
# JAN 2018
articles_file_name = "ArticlesJan2018.csv"
keywords_articles_ids = getArticles(keywords_list, articles_file_name)
comments_file_name = "CommentsJan2018.csv"
relevant_comments = getComments(keywords_articles_ids, comments_file_name)
getMarks(relevant_comments, comments_file_name)
# FEB 2018
articles_file_name = "ArticlesFeb2018.csv"
keywords_articles_ids = getArticles(keywords_list, articles_file_name)
comments_file_name = "CommentsFeb2018.csv"
relevant_comments = getComments(keywords_articles_ids, comments_file_name)
getMarks(relevant_comments, comments_file_name)
# MARCH 2018
articles_file_name = "ArticlesMarch2018.csv"
keywords_articles_ids = getArticles(keywords_list, articles_file_name)
comments_file_name = "CommentsMarch2018.csv"
relevant_comments = getComments(keywords_articles_ids, comments_file_name)
getMarks(relevant_comments, comments_file_name)
# APRIL 2018
articles_file_name = "ArticlesApril2018.csv"
keywords_articles_ids = getArticles(keywords_list, articles_file_name)
comments_file_name = "CommentsApril2018.csv"
relevant_comments = getComments(keywords_articles_ids, comments_file_name)
getMarks(relevant_comments, comments_file_name)
