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


def getLocationBasedOpinions(relevantComments, comments_indices, comments_file_name):

    Alabama = ["AL", "Alabama", "alabama"]
    Alaska = ["AK", "Alaska", "alaska"]
    Arizona = ["AZ", "Arizona", "arizona"]
    Arkansas = ["AR", "Arkansas", "arkansas"]
    California = ["CA", "California", "california"]
    Colorado = ["CO", "Colorado", "colorado"]
    Connecticut = ["CT", "Connecticut", "connecticut"]
    Delaware = ["DE", "Delaware", "delaware"]
    Florida = ["FL", "Florida", "florida"]
    Georgia = ["GA", "Georgia", "georgia"]
    Hawaii = ["HI", "Hawaii", "hawaii"]
    Idaho = ["ID", "Idaho", "idaho"]
    Illinois = ["IL", "Illinois", "illinois"]
    Indiana = ["IN", "Indiana", "indiana"]
    Iowa = ["IA", "Iowa", "iowa"]
    Kansas = ["KS", "Kansas", "kansas"]
    Kentucky = ["KY", "Kentucky", "kentucky"]
    Louisiana = ["LA", "Louisiana", "louisiana"]
    Maine = ["ME", "Maine", "maine"]
    Maryland = ["MD", "Maryland", "maryland"]
    Massachusetts = ["MA", "Massachusetts", "massachusetts"]
    Michigan = ["MI", "Michigan", "michigan"]
    Minnesota = ["MN", "Minnesota", "minnesota"]
    Mississippi = ["MS", "Mississippi", "mississippi"]
    Missouri = ["MO", "Missouri", "missouri"]
    Montana = ["MT", "Montana", "montana"]
    Nebraska = ["NE", "Nebraska", "nebraska"]
    Nevada = ["NV", "Nevada", "nevada"]
    New_Hampshire = ["NH", "New Hampshire", "new hampshire", "New hampshire", "new Hampshire"]
    New_Jersey = ["NJ", "New Jersey", "new jersey", "new Jersey", "New jersey"]
    New_Mexico = ["NM", "New Mexico", "new mexico", "new Mexico", "New mexico"]
    New_York = ["NY", "New York", "new york", "new York", "New york"]
    North_Carolina = ["NC", "North Carolina", "north carolina", "North carolina", "north Carolina"]
    North_Dakota = ["ND", "North Dakota", "north dakota", "North dakota", "north Dakota"]
    Ohio = ["OH", "Ohio", "ohio"]
    Oklahoma = ["OK", "Oklahoma", "oklahoma"]
    Oregon = ["OR", "Oregon", "oregon"]
    Pennsylvania = ["PA", "Pennsylvania", "pennsylvania"]
    Rhode_Island = ["RI", "Rhode Island", "rhode island", "Rhode island", "rhode Island"]
    South_Carolina = ["SC", "South Carolina", "south carolina", "South carolina", "south Carolina"]
    South_Dakota = ["SD", "South Dakota", "south dakota", "South dakota", "south Dakota"]
    Tennessee = ["TN", "Tennessee", "tennessee"]
    Texas = ["TX", "Texas", "texas"]
    Utah = ["UT", "Utah", "utah"]
    Vermont = ["VT", "Vermont", "vermont"]
    Virginia = ["VA", "Virginia", "virginia"]
    Washington = ["WA", "Washington", "washington"]
    West_Virginia = ["WV", "West Virginia", "West virginia", "west Virginia"]
    Wisconsin = ["WI", "Wisconsin", "wisconsin"]
    Wyoming = ["WY", "Wyoming", "wyoming"]
    list_of_countries = [Alabama, Alaska, Arizona, Arkansas, California, Colorado, Connecticut, Delaware, Florida, Georgia,
                         Hawaii, Idaho, Illinois, Indiana, Iowa, Kansas, Kentucky, Louisiana, Maine, Maryland, Massachusetts,
                         Michigan, Minnesota, Mississippi, Missouri, Montana, Nebraska, Nevada, New_Hampshire, New_Jersey,
                         New_Mexico, New_York, North_Carolina, North_Dakota, Ohio, Oklahoma, Oregon, Pennsylvania, Rhode_Island,
                         South_Carolina, South_Dakota, Tennessee, Texas, Utah, Vermont, Virginia, Washington, West_Virginia,
                         Wisconsin, Wyoming]
    df = pd.read_csv(comments_file_name)
    locations_column = df.userLocation
    countries_dict = {}
    rates_dict = {"very pos":0, "slightly pos":0, "natural":0, "slightly neg":0, "very neg":0}
    for country in list_of_countries:
        countries_dict[country] = rates_dict
    i = 0
    for comment in relevantComments:
        analysis = TextBlob(comment)
        comment_mark = analysis.sentences[0].sentiment.polarity
        if comment_mark > 0.00:
            if comment_mark > 0.3:
                address = locations_column[relevantComments[i]]
                
            else:

        elif comment_mark < 0.00:
            if comment_mark < -0.3:

            else:

        else:

        i += 1




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
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name)
getClassificationsByMl(relevant_comments, comments_file_name, train)
# FEB 2017
articles_file_name = "ArticlesFeb2017.csv"
keywords_articles_ids = getArticles(keywords_list, articles_file_name)
comments_file_name = "CommentsFeb2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name)
# MARCH 2017
articles_file_name = "ArticlesMarch2017.csv"
keywords_articles_ids = getArticles(keywords_list, articles_file_name)
comments_file_name = "CommentsMarch2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name)
# APRIL 2017
articles_file_name = "ArticlesApril2017.csv"
keywords_articles_ids = getArticles(keywords_list, articles_file_name)
comments_file_name = "CommentsApril2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name)
# MAY 2017
articles_file_name = "ArticlesMay2017.csv"
keywords_articles_ids = getArticles(keywords_list, articles_file_name)
comments_file_name = "CommentsMay2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name)
# JAN 2018
articles_file_name = "ArticlesJan2018.csv"
keywords_articles_ids = getArticles(keywords_list, articles_file_name)
comments_file_name = "CommentsJan2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name)
# FEB 2018
articles_file_name = "ArticlesFeb2018.csv"
keywords_articles_ids = getArticles(keywords_list, articles_file_name)
comments_file_name = "CommentsFeb2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name)
# MARCH 2018
articles_file_name = "ArticlesMarch2018.csv"
keywords_articles_ids = getArticles(keywords_list, articles_file_name)
comments_file_name = "CommentsMarch2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name)
# APRIL 2018
articles_file_name = "ArticlesApril2018.csv"
keywords_articles_ids = getArticles(keywords_list, articles_file_name)
comments_file_name = "CommentsApril2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name)
