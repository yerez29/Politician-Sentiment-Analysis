"""
This file is part of A Needle in a Data Haystack (course 67978) Project, group 20.
Made by: Yair Erez, Orel Keliahu and Batel Luzon.
This file contain code for Machine Learning analysis of nyt readers comments and results presentations in order to build
politicians popularity measurements.
NOTE: TO RUN THIS CODE ALL NEW YORK TIMES DATA FILES AND ALL ".PICKLE" FILES (CREATED BY "MlTraining.py" FILE) NEED TO
BE LOCATED IN WORKING DIRECTORY.
"""

# IMPORTS

import pickle
import pandas as pd
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from copy import deepcopy
from math import floor
from random import shuffle
from tabulate import tabulate
POS = "pos"
NEG = "neg"
NUM_OF_COMMENTS = 6000
CONFIDENCE_RATIO = 0.8

# CONSTS

ENTIRE_MONTH = "Entire Month"

# POLITICIANS

DONALD_TRUMP = "Donald Trump"
HILLARY_CLINTON = "Hillary Clinton"
BARACK_OBAMA = "Barack Obama"
BENJAMIN_NETANYAHU = "Benjamin Netanyahu"


class votingClassifier(ClassifierI):
    def __init__(self, *classifiers):
        """
        Class constructor. Input is a list of classifiers of different kind.
        :param classifiers:
        """
        self._classifiers = classifiers

    def classify(self, current_features):
        """
        Classification method.
        :param current_features: features to classify with.
        :return: a positive or negative text classification.
        """
        classifiers_votes = []
        for current_classifier in self._classifiers:
            current_vote = current_classifier.classify(current_features)
            classifiers_votes.append(current_vote)
        return mode(classifiers_votes)

    def confidence(self, current_features):
        """
        return the confidence of the classification, which is the rate of classifiers that agreed on current classification.
        :param current_features: features to classify with.
        :return: the confidence of the current classification.
        """
        current_votes = []
        for current_classifier in self._classifiers:
            current_vote = current_classifier.classify(current_features)
            current_votes.append(current_vote)
        current_chosen_votes = current_votes.count(mode(current_votes))
        current_confidence = current_chosen_votes / len(current_votes)
        return current_confidence


# loading current comments and tags
comments_and_tags_file = open("commentsTags.pickle", "rb")
current_comment_and_tags = pickle.load(comments_and_tags_file)
comments_and_tags_file.close()

# loading word features
features_words_file = open("featuresWords.pickle", "rb")
features_words = pickle.load(features_words_file)
features_words_file.close()


def featuresSearch(comment):
    """
    This function returns a dictionary as features of current comment.
    :param comment: current comment to get features from.
    :return: a dictionary with keys and values as features.
    """
    commentWords = word_tokenize(comment)
    featuresDic = {}
    for word in features_words:
        featuresDic[word] = word in commentWords
    return featuresDic


# load Naive Bayes classifier
current_file = open("naiveBayes.pickle", "rb")
classifier = pickle.load(current_file)
current_file.close()

# load Multinomial Naive Bayes classifier
current_file = open("multiNaiveBayes.pickle", "rb")
MNB_classifier = pickle.load(current_file)
current_file.close()

# load Bernoulli Naive Bayes classifier
current_file = open("bernoulliNaiveBayes.pickle", "rb")
BernoulliNB_classifier = pickle.load(current_file)
current_file.close()

# load Logistic Regression classifier
current_file = open("logisticRegression.pickle", "rb")
LogisticRegression_classifier = pickle.load(current_file)
current_file.close()

# load Linear Support Vector classifier
current_file = open("linearSv.pickle", "rb")
LinearSVC_classifier = pickle.load(current_file)
current_file.close()

voting_module = votingClassifier(classifier, LinearSVC_classifier, MNB_classifier, BernoulliNB_classifier, LogisticRegression_classifier)


def getCurrentSentiment(text):

    calculated_features = featuresSearch(text)
    return voting_module.classify(calculated_features), voting_module.confidence(calculated_features)


def getArticles(keywords_list, articles_file_name):
    """
    This function retrieves the ids of articles related to chosen politician.
    :param keywords_list: list of tags related to chosen politician.
    :param articles_file_name: articles file name.
    :return: a list with relevant article ids.
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
    This function returns 2 lists with nyt's readers comments and comments indices related to chosen politician.
    :param keywords_articles_ids: ids of relevant articles.
    :param comments_file_name: comment file name.
    :return: relevant comments and their indices within the file.
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
    """
        A simple function for percent calculations.
        :param part: partial number.
        :param whole: whole number.
        :return: calculated percentage.
    """
    if whole == 0:
        return 0
    return (part/whole) * 100


def getGeneralOpinions(relevant_comments, comments_file_name, segment_num, politician):
    """
    This function calculates and presents the distribution of comments positive, natural and negative sentiments
    within an entire month.
    :param relevant_comments: a list of nyt readers comments.
    :param comments_file_name: a comments file name.
    :param segment_num: an indicator for entire/partial month portion.
    :param politician: the name of politician to analyze.
    :return: no return for this function.
    """
    positive_mark = 0
    negative_mark = 0
    natural_mark = 0
    shuffle(relevant_comments)  # shuffle the list to get random independent comments
    date = comments_file_name[8:len(comments_file_name) - 4]
    stop = 0
    numOfComments = len(relevant_comments)
    for comment in relevant_comments:  # iterating relevant comments
        analysis = getCurrentSentiment(comment) # extract comment sentiment
        confidence_mark = analysis[1]
        if confidence_mark >= CONFIDENCE_RATIO:  # classify the comment according to sentiment and confidence
            if analysis[0] == POS:
                positive_mark += 1
            elif analysis[0] == NEG:
                negative_mark += 1
        else:
            natural_mark += 1
        stop += 1
        if stop > NUM_OF_COMMENTS - 1:  # break the loop if reached maximum number of comments
            break
    # calculate sentiment distribution
    if numOfComments > NUM_OF_COMMENTS:
        positive_mark = percent(positive_mark, NUM_OF_COMMENTS)
        negative_mark = percent(negative_mark, NUM_OF_COMMENTS)
        natural_mark = percent(natural_mark, NUM_OF_COMMENTS)
    else:
        positive_mark = percent(positive_mark, numOfComments)
        negative_mark = percent(negative_mark, numOfComments)
        natural_mark = percent(natural_mark, numOfComments)
    # present the results in pie chart form
    labels = 'Positive', 'Negative', 'Natural'
    sizes = [positive_mark, negative_mark, natural_mark]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    if segment_num == ENTIRE_MONTH:
        plt.title("Opinion distribution in " + date + " of " + politician)
    else:
        plt.title("Opinion distribution in " + segment_num + " part of " + date + " of " + politician)
    plt.axis('equal')
    plt.show()
    plt.clf()


def getLocationBasedOpinions(relevantComments, comments_indices, comments_file_name, politician):
    """
        This function calculates and presents the distribution of comments positive, natural and negative sentiments
        within an entire month based on geographical location of comments writer.
        :param relevantComments: a list of nyt readers relevant comments to chosen politician.
        :param comments_indices: a list of indices within the file of nyt readers relevant comments to chosen politician.
        :param comments_file_name: comments file name.
        :param politician: the name of politician to analyze.
        :return: no return for this function.
    """
    # create list of us countries and their abbreviations as appear in nyt readers comment file.
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
    # build and assign counting dictionary to each state
    rates_dict = {"pos": 0, "natural": 0, "neg": 0}
    for country in list_of_countries:
        countries_dict[country[1]] = deepcopy(rates_dict)
    i = 0
    shuffle(relevantComments)  # shuffle the list to get random independent comments
    for comment in relevantComments:  # iterating comments
        analysis = getCurrentSentiment(comment)  # extract sentiment and confidence
        if analysis[1] >= CONFIDENCE_RATIO:  # check confidence ratio for atleast 80 percent classification agreement
            if analysis[0] == POS:  # check sentiment value for positive
                address = locations_column[comments_indices[i]]
                if type(address) is str:
                    flag = False
                    for country in list_of_countries:  # find the relevant state and add 1 to appropriate value
                        for abb in country:
                            if abb in address:
                                countries_dict[country[1]]["pos"] += 1
                                flag = True
                                break
                        if flag:
                            break
            elif analysis[0] == NEG:  # check sentiment value for negative
                address = locations_column[comments_indices[i]]
                if type(address) is str:
                    flag = False
                    for country in list_of_countries:  # find the relevant state and add 1 to appropriate value
                        for abb in country:
                            if abb in address:
                                countries_dict[country[1]]["neg"] += 1
                                flag = True
                                break
                        if flag:
                            break
        else:  # less than 80 percent classification agreement - classify as natural
            address = locations_column[comments_indices[i]]
            if type(address) is str:
                flag = False
                for country in list_of_countries:  # find the relevant state and add 1 to appropriate value
                    for abb in country:
                        if abb in address:
                            countries_dict[country[1]]["natural"] += 1
                            flag = True
                            break
                    if flag:
                        break
        i += 1
        if i >= NUM_OF_COMMENTS:  # break the loop if reached maximum number of comments
            break
    date = comments_file_name[8:len(comments_file_name) - 4]
    # calculate sentiment distribuation for each state
    headers = ["State", "Positive(%)", "Negative(%)", "Natural(%)"]
    statesRates = []
    for key in countries_dict:
        total = sum(countries_dict[key].values())
        positive_mark = percent(countries_dict[key]["pos"], total)
        negative_mark = percent(countries_dict[key]["neg"], total)
        natural_mark = percent(countries_dict[key]["natural"], total)
        statesRates.append((key, positive_mark, negative_mark, natural_mark))
    # print the results in tabular form
    print("Opinion distribution by state at " + date + " of " + politician + ":")
    print(tabulate(statesRates, headers=headers, tablefmt="grid"))


def getTimeBasedOpinions(relevantComments, comments_indices, comments_file_name, politician):
    """
        This function calculates and present the results of sentiment distribution within each third of a month,
        each third is usually composed from 10 days.
        :param relevantComments: a list of nyt readers relevant comments to chosen politician.
        :param comments_indices: a list of indices within the file of nyt readers relevant comments to chosen politician.
        :param comments_file_name: comment file name.
        :param politician: the name of politician to analyze.
        :return: no return for this function.
    """
    # create a list of comments sorted by their chronological writing dates
    month_division = 4
    df = pd.read_csv(comments_file_name)
    comments_dates_column = df.createDate
    relevant_dates = []
    num_of_dates = len(relevantComments)
    date_comment_dic = {}
    for i in range(num_of_dates):
        relevant_dates.append(comments_dates_column[comments_indices[i]])
        date_comment_dic[comments_dates_column[comments_indices[i]]] = relevantComments[i]
    relevant_dates.sort()
    sorted_comment_dates = []
    for date in relevant_dates:
        sorted_comment_dates.append(date_comment_dic[date])
    segment_size = floor(num_of_dates / month_division)
    current_segment_counter = 0
    segment_num = 1
    current_comments = []
    # iterate sorted comments
    for comment in sorted_comment_dates:
        if current_segment_counter <= segment_size:  # append comment to current third
            current_comments.append(comment)
            current_segment_counter += 1
        else:  # send current comments third to general comments function with current month third indicator
            getGeneralOpinions(current_comments, comments_file_name, str(segment_num), politician)
            current_comments = []
            current_segment_counter = 0
            segment_num += 1


# Chosen politicians nyt tags lists

Trump_keywords_list = ["Trump", 'Donald', "trump", "donald", "TRUMP", "DONALD", "Trump, Donald J", "Donald Trump", "Donald John Trump", "Donald J. Trump"]
Clinton_keyword_list = ["Clinton", "Hillary", "clinton", "hillary", "HILLARY", "CLINTON", "Hillary Clinton", "hillary clinton", "HILLARY CLINTON", "Hillary clinton", "hillary Clinton", "Hillary Rodham Clinton", "Hillary Diane Rodham Clinton", "Clinton, Hillary Rodham"]
Obama_keyword_list = ["Obama", "Barack", "obama", "barack", "OBAMA", "BARACK", "Barack Obama", "barack obama", "BARACK OBAMA", "Barack Hussein Obama", "Barack H. Obama", "Barack h. Obama", "Obama, Barack"]
Netanyahu_keyword_list = ["Netanyahu", "Benjamin", "netanyahu", "benjamin", "NETANYAHU", "BENJAMIN", "Benjamin Netanyahu", "Benjamin netanyahu", "benjamin Netanyahu", "benjamin netanyahu", "BENJAMIN NETANYAHU", "Netanyahu, Benjamin", "Bibi netanyahu", "bibi netanyahu", "Bibi Netanyahu", "bibi Netanyahu"]

# Donald Trump machine learning based analysis for January 2017

articles_file_name = "ArticlesJan2017.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsJan2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)

# Donald Trump machine learning based analysis for February 2017

articles_file_name = "ArticlesFeb2017.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsFeb2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)

# Donald Trump machine learning based analysis for March 2017

articles_file_name = "ArticlesMarch2017.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsMarch2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)

# Donald Trump machine learning based analysis for April 2017

articles_file_name = "ArticlesApril2017.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsApril2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)

# Donald Trump machine learning based analysis for May 2017

articles_file_name = "ArticlesMay2017.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsMay2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)

# Donald Trump machine learning based analysis for January 2018

articles_file_name = "ArticlesJan2018.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsJan2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)

# Donald Trump machine learning based analysis for February 2018

articles_file_name = "ArticlesFeb2018.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsFeb2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)

# Donald Trump machine learning based analysis for March 2018

articles_file_name = "ArticlesMarch2018.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsMarch2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)

# Donald Trump machine learning based analysis for April 2018

articles_file_name = "ArticlesApril2018.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsApril2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)

# Hillary Clinton machine learning based analysis for January 2017

articles_file_name = "ArticlesJan2017.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsJan2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)

# Hillary Clinton machine learning based analysis for February 2017

articles_file_name = "ArticlesFeb2017.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsFeb2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)

# Hillary Clinton machine learning based analysis for March 2017

articles_file_name = "ArticlesMarch2017.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsMarch2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)

# Hillary Clinton machine learning based analysis for April 2017

articles_file_name = "ArticlesApril2017.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsApril2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)

# Hillary Clinton machine learning based analysis for May2017

articles_file_name = "ArticlesMay2017.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsMay2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)

# Hillary Clinton machine learning based analysis for January 2018

articles_file_name = "ArticlesJan2018.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsJan2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)

# Hillary Clinton machine learning based analysis for February 2018

articles_file_name = "ArticlesFeb2018.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsFeb2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)

# Hillary Clinton machine learning based analysis for March 2018

articles_file_name = "ArticlesMarch2018.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsMarch2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)

# Hillary Clinton machine learning based analysis for April 2018

articles_file_name = "ArticlesApril2018.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsApril2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)

# Barack Obama machine learning based analysis for January 2017

articles_file_name = "ArticlesJan2017.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsJan2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)

# Barack Obama machine learning based analysis for February 2017

articles_file_name = "ArticlesFeb2017.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsFeb2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)

# Barack Obama machine learning based analysis for March 2017

articles_file_name = "ArticlesMarch2017.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsMarch2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)

# Barack Obama machine learning based analysis for April 2017

articles_file_name = "ArticlesApril2017.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsApril2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)

# Barack Obama machine learning based analysis for May 2017

articles_file_name = "ArticlesMay2017.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsMay2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)

# Barack Obama machine learning based analysis for January 2018

articles_file_name = "ArticlesJan2018.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsJan2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)

# Barack Obama machine learning based analysis for February 2018

articles_file_name = "ArticlesFeb2018.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsFeb2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)

# Barack Obama machine learning based analysis for March 2018

articles_file_name = "ArticlesMarch2018.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsMarch2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)

# Barack Obama machine learning based analysis for April 2018

articles_file_name = "ArticlesApril2018.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsApril2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)

# Benjamin Netanyahu machine learning based analysis for January 2017

articles_file_name = "ArticlesJan2017.csv"
keywords_articles_ids = getArticles(Netanyahu_keyword_list, articles_file_name)
comments_file_name = "CommentsJan2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BENJAMIN_NETANYAHU)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)

# Benjamin Netanyahu machine learning based analysis for February 2017

articles_file_name = "ArticlesFeb2017.csv"
keywords_articles_ids = getArticles(Netanyahu_keyword_list, articles_file_name)
comments_file_name = "CommentsFeb2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BENJAMIN_NETANYAHU)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)

# Benjamin Netanyahu machine learning based analysis for March 2017

articles_file_name = "ArticlesMarch2017.csv"
keywords_articles_ids = getArticles(Netanyahu_keyword_list, articles_file_name)
comments_file_name = "CommentsMarch2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BENJAMIN_NETANYAHU)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)

# Benjamin Netanyahu machine learning based analysis for April 2017

articles_file_name = "ArticlesApril2017.csv"
keywords_articles_ids = getArticles(Netanyahu_keyword_list, articles_file_name)
comments_file_name = "CommentsApril2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BENJAMIN_NETANYAHU)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)

# Benjamin Netanyahu machine learning based analysis for May 2017

articles_file_name = "ArticlesMay2017.csv"
keywords_articles_ids = getArticles(Netanyahu_keyword_list, articles_file_name)
comments_file_name = "CommentsMay2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BENJAMIN_NETANYAHU)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)

# Benjamin Netanyahu machine learning based analysis for January 2018

articles_file_name = "ArticlesJan2018.csv"
keywords_articles_ids = getArticles(Netanyahu_keyword_list, articles_file_name)
comments_file_name = "CommentsJan2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BENJAMIN_NETANYAHU)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)

# Benjamin Netanyahu machine learning based analysis for February 2018

articles_file_name = "ArticlesFeb2018.csv"
keywords_articles_ids = getArticles(Netanyahu_keyword_list, articles_file_name)
comments_file_name = "CommentsFeb2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BENJAMIN_NETANYAHU)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)

# Benjamin Netanyahu machine learning based analysis for March 2018

articles_file_name = "ArticlesMarch2018.csv"
keywords_articles_ids = getArticles(Netanyahu_keyword_list, articles_file_name)
comments_file_name = "CommentsMarch2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BENJAMIN_NETANYAHU)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)

# Benjamin Netanyahu machine learning based analysis for April 2018

articles_file_name = "ArticlesApril2018.csv"
keywords_articles_ids = getArticles(Netanyahu_keyword_list, articles_file_name)
comments_file_name = "CommentsApril2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BENJAMIN_NETANYAHU)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
