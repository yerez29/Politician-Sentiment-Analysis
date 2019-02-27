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
NUM_OF_COMMENTS = 8000
CONFIDENCE_RATIO = 0.8

# CONSTS

ENTIRE_MONTH = "Entire Month"

# POLITICIANS

DONALD_TRUMP = "Donald Trump"
HILLARY_CLINTON = "Hillary Clinton"
BARACK_OBAMA = "Barack Obama"
BENJAMIN_NETANYAHU = "Benjamin Netanyahu"


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


documents_f = open("documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()


word_features5k_f = open("word_features5k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()


def find_features(comment):

    commentWords = word_tokenize(comment)
    featuresDic = {}
    for w in word_features:
        featuresDic[w] = w in commentWords
    return featuresDic


open_file = open("originalnaivebayes5k.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()


open_file = open("MNB_classifier5k.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()


open_file = open("BernoulliNB_classifier5k.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()


open_file = open("LogisticRegression_classifier5k.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()


open_file = open("LinearSVC_classifier5k.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

voted_classifier = VoteClassifier(classifier, LinearSVC_classifier, MNB_classifier, BernoulliNB_classifier, LogisticRegression_classifier)


def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)


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

    if whole == 0:
        return 0
    return (part/whole) * 100


def getGeneralOpinions(relevant_comments, comments_file_name, segment_num, politician):
    """
    :param comments:
    :return:
    """
    positive_mark = 0
    negative_mark = 0
    natural_mark = 0
    shuffle(relevant_comments)
    date = comments_file_name[8:len(comments_file_name) - 4]
    stop = 0
    numOfComments = len(relevant_comments)
    for comment in relevant_comments:
        analysis = sentiment(comment)
        confidence_mark = analysis[1]
        if confidence_mark >= CONFIDENCE_RATIO:
            if analysis[0] == POS:
                positive_mark += 1
            elif analysis[0] == NEG:
                negative_mark += 1
        else:
            natural_mark += 1
        stop += 1
        if stop > NUM_OF_COMMENTS - 1:
            break
    if numOfComments > NUM_OF_COMMENTS:
        positive_mark = percent(positive_mark, NUM_OF_COMMENTS)
        negative_mark = percent(negative_mark, NUM_OF_COMMENTS)
        natural_mark = percent(natural_mark, NUM_OF_COMMENTS)
    else:
        positive_mark = percent(positive_mark, numOfComments)
        negative_mark = percent(negative_mark, numOfComments)
        natural_mark = percent(natural_mark, numOfComments)
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
    rates_dict = {"pos": 0, "natural": 0, "neg": 0}
    for country in list_of_countries:
        countries_dict[country[1]] = deepcopy(rates_dict)
    i = 0
    shuffle(relevantComments)
    for comment in relevantComments:
        analysis = sentiment(comment)
        if analysis[1] >= CONFIDENCE_RATIO:
            if analysis[0] == POS:
                address = locations_column[comments_indices[i]]
                if type(address) is str:
                    flag = False
                    for country in list_of_countries:
                        for abb in country:
                            if abb in address:
                                countries_dict[country[1]]["pos"] += 1
                                flag = True
                                break
                        if flag:
                            break
            elif analysis[0] == NEG:
                address = locations_column[comments_indices[i]]
                if type(address) is str:
                    flag = False
                    for country in list_of_countries:
                        for abb in country:
                            if abb in address:
                                countries_dict[country[1]]["neg"] += 1
                                flag = True
                                break
                        if flag:
                            break
        else:
            address = locations_column[comments_indices[i]]
            if type(address) is str:
                flag = False
                for country in list_of_countries:
                    for abb in country:
                        if abb in address:
                            countries_dict[country[1]]["natural"] += 1
                            flag = True
                            break
                    if flag:
                        break
        i += 1
        if i >= NUM_OF_COMMENTS:
            break
    date = comments_file_name[8:len(comments_file_name) - 4]
    headers = ["State", "Positive(%)", "Negative(%)", "Natural(%)"]
    statesRates = []
    for key in countries_dict:
        total = sum(countries_dict[key].values())
        positive_mark = percent(countries_dict[key]["pos"], total)
        negative_mark = percent(countries_dict[key]["neg"], total)
        natural_mark = percent(countries_dict[key]["natural"], total)
        statesRates.append((key, positive_mark, negative_mark, natural_mark))
    print("Opinion distribution by state at " + date + " of " + politician + ":")
    print(tabulate(statesRates, headers=headers, tablefmt="grid"))


def getTimeBasedOpinions(relevantComments, comments_indices, comments_file_name, politician):

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
    for comment in sorted_comment_dates:
        if current_segment_counter <= segment_size:
            current_comments.append(comment)
            current_segment_counter += 1
        else:
            getGeneralOpinions(current_comments, comments_file_name, str(segment_num), politician)
            current_comments = []
            current_segment_counter = 0
            segment_num += 1


# Chosen politicians lists

Trump_keywords_list = ["Trump", 'Donald', "trump", "donald", "TRUMP", "DONALD", "Trump, Donald J", "Donald Trump", "Donald John Trump", "Donald J. Trump"]
Clinton_keyword_list = ["Clinton", "Hillary", "clinton", "hillary", "HILLARY", "CLINTON", "Hillary Clinton", "hillary clinton", "HILLARY CLINTON", "Hillary clinton", "hillary Clinton", "Hillary Rodham Clinton", "Hillary Diane Rodham Clinton", "Clinton, Hillary Rodham"]
Obama_keyword_list = ["Obama", "Barack", "obama", "barack", "OBAMA", "BARACK", "Barack Obama", "barack obama", "BARACK OBAMA", "Barack Hussein Obama", "Barack H. Obama", "Barack h. Obama", "Obama, Barack"]
Netanyahu_keyword_list = ["Netanyahu", "Benjamin", "netanyahu", "benjamin", "NETANYAHU", "BENJAMIN", "Benjamin Netanyahu", "Benjamin netanyahu", "benjamin Netanyahu", "benjamin netanyahu", "BENJAMIN NETANYAHU", "Netanyahu, Benjamin", "Bibi netanyahu", "bibi netanyahu", "Bibi Netanyahu", "bibi Netanyahu"]

# DONALD_TRUMP

# JAN 2017
articles_file_name = "ArticlesJan2017.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsJan2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)
# FEB 2017
articles_file_name = "ArticlesFeb2017.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsFeb2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)
# MARCH 2017
articles_file_name = "ArticlesMarch2017.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsMarch2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)
# APRIL 2017
articles_file_name = "ArticlesApril2017.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsApril2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)
# MAY 2017
articles_file_name = "ArticlesMay2017.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsMay2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)
# JAN 2018
articles_file_name = "ArticlesJan2018.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsJan2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)
# FEB 2018
articles_file_name = "ArticlesFeb2018.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsFeb2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)
# MARCH 2018
articles_file_name = "ArticlesMarch2018.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsMarch2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)
# APRIL 2018
articles_file_name = "ArticlesApril2018.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsApril2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, DONALD_TRUMP)


# Hillary Clinton

# JAN 2017
articles_file_name = "ArticlesJan2017.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsJan2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)
# FEB 2017
articles_file_name = "ArticlesFeb2017.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsFeb2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)
# MARCH 2017
articles_file_name = "ArticlesMarch2017.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsMarch2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)
# APRIL 2017
articles_file_name = "ArticlesApril2017.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsApril2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)
# MAY 2017
articles_file_name = "ArticlesMay2017.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsMay2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)
# JAN 2018
articles_file_name = "ArticlesJan2018.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsJan2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)
# FEB 2018
articles_file_name = "ArticlesFeb2018.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsFeb2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)
# MARCH 2018
articles_file_name = "ArticlesMarch2018.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsMarch2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)
# APRIL 2018
articles_file_name = "ArticlesApril2018.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsApril2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, HILLARY_CLINTON)


# BARACK OBAMA

# JAN 2017
articles_file_name = "ArticlesJan2017.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsJan2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)
# FEB 2017
articles_file_name = "ArticlesFeb2017.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsFeb2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)
# MARCH 2017
articles_file_name = "ArticlesMarch2017.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsMarch2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)
# APRIL 2017
articles_file_name = "ArticlesApril2017.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsApril2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)
# MAY 2017
articles_file_name = "ArticlesMay2017.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsMay2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)
# JAN 2018
articles_file_name = "ArticlesJan2018.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsJan2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)
# FEB 2018
articles_file_name = "ArticlesFeb2018.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsFeb2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)
# MARCH 2018
articles_file_name = "ArticlesMarch2018.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsMarch2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)
# APRIL 2018
articles_file_name = "ArticlesApril2018.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsApril2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BARACK_OBAMA)


# BENJAMIN NETANYAHU

# JAN 2017
articles_file_name = "ArticlesJan2017.csv"
keywords_articles_ids = getArticles(Netanyahu_keyword_list, articles_file_name)
comments_file_name = "CommentsJan2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BENJAMIN_NETANYAHU)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
# FEB 2017
articles_file_name = "ArticlesFeb2017.csv"
keywords_articles_ids = getArticles(Netanyahu_keyword_list, articles_file_name)
comments_file_name = "CommentsFeb2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BENJAMIN_NETANYAHU)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
# MARCH 2017
articles_file_name = "ArticlesMarch2017.csv"
keywords_articles_ids = getArticles(Netanyahu_keyword_list, articles_file_name)
comments_file_name = "CommentsMarch2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BENJAMIN_NETANYAHU)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
# APRIL 2017
articles_file_name = "ArticlesApril2017.csv"
keywords_articles_ids = getArticles(Netanyahu_keyword_list, articles_file_name)
comments_file_name = "CommentsApril2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BENJAMIN_NETANYAHU)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
# MAY 2017
articles_file_name = "ArticlesMay2017.csv"
keywords_articles_ids = getArticles(Netanyahu_keyword_list, articles_file_name)
comments_file_name = "CommentsMay2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BENJAMIN_NETANYAHU)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
# JAN 2018
articles_file_name = "ArticlesJan2018.csv"
keywords_articles_ids = getArticles(Netanyahu_keyword_list, articles_file_name)
comments_file_name = "CommentsJan2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BENJAMIN_NETANYAHU)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
# FEB 2018
articles_file_name = "ArticlesFeb2018.csv"
keywords_articles_ids = getArticles(Netanyahu_keyword_list, articles_file_name)
comments_file_name = "CommentsFeb2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BENJAMIN_NETANYAHU)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
# MARCH 2018
articles_file_name = "ArticlesMarch2018.csv"
keywords_articles_ids = getArticles(Netanyahu_keyword_list, articles_file_name)
comments_file_name = "CommentsMarch2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BENJAMIN_NETANYAHU)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
# APRIL 2018
articles_file_name = "ArticlesApril2018.csv"
keywords_articles_ids = getArticles(Netanyahu_keyword_list, articles_file_name)
comments_file_name = "CommentsApril2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(deepcopy(relevant_comments), comments_file_name, ENTIRE_MONTH, BENJAMIN_NETANYAHU)
getLocationBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
getTimeBasedOpinions(deepcopy(relevant_comments), comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
