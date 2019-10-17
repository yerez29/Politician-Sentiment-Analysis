"""
Made by: Yair Erez.
This file contain code for lexicon-based analysis of nyt readers comments and results presentations in order to build
politicians popularity measurements.
NOTE: TO RUN THIS CODE ALL NEW YORK TIMES DATA FILES NEED TO BE LOCATED IN WORKING DIRECTORY.
"""

# IMPORTS

import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from copy import deepcopy
from math import floor
from tabulate import tabulate

# CONSTS

ENTIRE_MONTH = "Entire Month"
POSITIVE_THRESHOLD = 0.3
NEGATIVE_THRESHOLD = -0.3

# POLITICIANS

DONALD_TRUMP = "Donald Trump"
HILLARY_CLINTON = "Hillary Clinton"
BARACK_OBAMA = "Barack Obama"
BENJAMIN_NETANYAHU = "Benjamin Netanyahu"

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
    very_positive_mark = 0
    slightly_positive_mark = 0
    slightly_negative_mark = 0
    very_negative_mark = 0
    natural_mark = 0
    num_of_comments = len(relevant_comments)
    date = comments_file_name[8:len(comments_file_name) - 4]
    for comment in relevant_comments:  # iterating comments
        analysis = TextBlob(comment)
        cur_average_mark = analysis.sentences[0].sentiment.polarity  # extract polarity
        if cur_average_mark > 0.00:  # sentiment classification
            if cur_average_mark > POSITIVE_THRESHOLD:
                very_positive_mark += 1
            else:
                slightly_positive_mark += 1
        elif cur_average_mark < 0.00:
            if cur_average_mark < NEGATIVE_THRESHOLD:
                very_negative_mark += 1
            else:
                slightly_negative_mark += 1
        else:
            natural_mark += 1
    # comments sentiment distribution calculation
    very_positive_mark = percent(very_positive_mark, num_of_comments)
    slightly_positive_mark = percent(slightly_positive_mark, num_of_comments)
    slightly_negative_mark = percent(slightly_negative_mark, num_of_comments)
    very_negative_mark = percent(very_negative_mark, num_of_comments)
    natural_mark = percent(natural_mark, num_of_comments)
    # results presentation in pie charts form
    labels = 'Very positive', 'Slightly positive', 'Slightly negative', 'Very negative', 'Natural'
    sizes = [very_positive_mark, slightly_positive_mark, slightly_negative_mark, very_negative_mark, natural_mark]
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
    rates_dict = {"very pos": 0, "slightly pos": 0, "natural": 0, "slightly neg": 0, "very neg": 0}
    for country in list_of_countries:
        countries_dict[country[1]] = deepcopy(rates_dict)
    i = 0
    for comment in relevantComments:  # iterating comments
        analysis = TextBlob(comment)
        comment_mark = analysis.sentences[0].sentiment.polarity  # extract the polarity
        if comment_mark > 0.00:  # classify according to polarity value - very positive comment
            if comment_mark > POSITIVE_THRESHOLD:
                address = locations_column[comments_indices[i]]
                if type(address) is str:  # check for the state and add 1 to relevant value in the states dictionary
                    flag = False
                    for country in list_of_countries:
                        for abb in country:
                            if abb in address:
                                countries_dict[country[1]]["very pos"] += 1
                                flag = True
                                break
                        if flag:
                            break
            else:  # classify according to polarity value - slightly positive comment
                address = locations_column[comments_indices[i]]
                if type(address) is str:  # check for the state and add 1 to relevant value in the states dictionary
                    flag = False
                    for country in list_of_countries:
                        for abb in country:
                            if abb in address:
                                countries_dict[country[1]]["slightly pos"] += 1
                                flag = True
                                break
                        if flag:
                            break
        elif comment_mark < 0.00:
            if comment_mark < -0.3:  # classify according to polarity value - very negative comment
                address = locations_column[comments_indices[i]]
                if type(address) is str:  # check for the state and add 1 to relevant value in the states dictionary
                    flag = False
                    for country in list_of_countries:
                        for abb in country:
                            if abb in address:
                                countries_dict[country[1]]["very neg"] += 1
                                flag = True
                                break
                        if flag:
                            break
            else:  # classify according to polarity value - slightly negative comment
                address = locations_column[comments_indices[i]]
                if type(address) is str:  # check for the state and add 1 to relevant value in the states dictionary
                    flag = False
                    for country in list_of_countries:
                        for abb in country:
                            if abb in address:
                                countries_dict[country[1]]["slightly neg"] += 1
                                flag = True
                                break
                        if flag:
                            break
        else:  # classify according to polarity value - natural comment
            address = locations_column[comments_indices[i]]
            if type(address) is str:  # check for the state and add 1 to relevant value in the states dictionary
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
    # calculate sentiments distribution in each state
    date = comments_file_name[8:len(comments_file_name) - 4]
    headers = ["State", "Very Positive(%)", "Slightly Positive(%)", "Slightly Negative(%)", "Very Negative(%)", "Natural(%)"]
    statesRates = []
    for key in countries_dict:
        total = sum(countries_dict[key].values())
        very_positive_mark = percent(countries_dict[key]["very pos"], total)
        slightly_positive_mark = percent(countries_dict[key]["slightly pos"], total)
        slightly_negative_mark = percent(countries_dict[key]["slightly neg"], total)
        very_negative_mark = percent(countries_dict[key]["very neg"], total)
        natural_mark = percent(countries_dict[key]["natural"], total)
        statesRates.append((key, very_positive_mark, slightly_positive_mark, slightly_negative_mark, very_negative_mark, natural_mark))
    # present the results in printed table
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

# Donald Trump lexicon based analysis for January 2017

articles_file_name = "ArticlesJan2017.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsJan2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)

# Donald Trump lexicon based analysis for February 2017

articles_file_name = "ArticlesFeb2017.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsFeb2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)

# Donald Trump lexicon based analysis for March 2017

articles_file_name = "ArticlesMarch2017.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsMarch2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)

# Donald Trump lexicon based analysis for April 2017

articles_file_name = "ArticlesApril2017.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsApril2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)

# Donald Trump lexicon based analysis for May 2017

articles_file_name = "ArticlesMay2017.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsMay2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)

# Donald Trump lexicon based analysis for January 2018

articles_file_name = "ArticlesJan2018.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsJan2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)

# Donald Trump lexicon based analysis for February 2018

articles_file_name = "ArticlesFeb2018.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsFeb2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)

# Donald Trump lexicon based analysis for March 2018

articles_file_name = "ArticlesMarch2018.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsMarch2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)

# Donald Trump lexicon based analysis for April 2018

articles_file_name = "ArticlesApril2018.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsApril2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)

# Hillary Clinton lexicon based analysis for January 2017

articles_file_name = "ArticlesJan2017.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsJan2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)

# Hillary Clinton lexicon based analysis for February 2017

articles_file_name = "ArticlesFeb2017.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsFeb2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)

# Hillary Clinton lexicon based analysis for March 2017

articles_file_name = "ArticlesMarch2017.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsMarch2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)

# Hillary Clinton lexicon based analysis for April 2017

articles_file_name = "ArticlesApril2017.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsApril2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)

# Hillary Clinton lexicon based analysis for May2017

articles_file_name = "ArticlesMay2017.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsMay2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)

# Hillary Clinton lexicon based analysis for January 2018

articles_file_name = "ArticlesJan2018.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsJan2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)

# Hillary Clinton lexicon based analysis for February 2018

articles_file_name = "ArticlesFeb2018.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsFeb2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)

# Hillary Clinton lexicon based analysis for March 2018

articles_file_name = "ArticlesMarch2018.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsMarch2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)

# Hillary Clinton lexicon based analysis for April 2018

articles_file_name = "ArticlesApril2018.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsApril2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)

# Barack Obama lexicon based analysis for January 2017

articles_file_name = "ArticlesJan2017.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsJan2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)

# Barack Obama lexicon based analysis for February 2017

articles_file_name = "ArticlesFeb2017.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsFeb2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)

# Barack Obama lexicon based analysis for March 2017

articles_file_name = "ArticlesMarch2017.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsMarch2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)

# Barack Obama lexicon based analysis for April 2017

articles_file_name = "ArticlesApril2017.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsApril2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)

# Barack Obama lexicon based analysis for May 2017

articles_file_name = "ArticlesMay2017.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsMay2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)

# Barack Obama lexicon based analysis for January 2018

articles_file_name = "ArticlesJan2018.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsJan2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)

# Barack Obama lexicon based analysis for February 2018

articles_file_name = "ArticlesFeb2018.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsFeb2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)

# Barack Obama lexicon based analysis for March 2018

articles_file_name = "ArticlesMarch2018.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsMarch2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)

# Barack Obama lexicon based analysis for April 2018

articles_file_name = "ArticlesApril2018.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsApril2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)

# Benjamin Netanyahu lexicon based analysis for January 2017

articles_file_name = "ArticlesJan2017.csv"
keywords_articles_ids = getArticles(Netanyahu_keyword_list, articles_file_name)
comments_file_name = "CommentsJan2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BENJAMIN_NETANYAHU)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BENJAMIN_NETANYAHU)

# Benjamin Netanyahu lexicon based analysis for February 2017

articles_file_name = "ArticlesFeb2017.csv"
keywords_articles_ids = getArticles(Netanyahu_keyword_list, articles_file_name)
comments_file_name = "CommentsFeb2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BENJAMIN_NETANYAHU)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BENJAMIN_NETANYAHU)

# Benjamin Netanyahu lexicon based analysis for March 2017

articles_file_name = "ArticlesMarch2017.csv"
keywords_articles_ids = getArticles(Netanyahu_keyword_list, articles_file_name)
comments_file_name = "CommentsMarch2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BENJAMIN_NETANYAHU)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BENJAMIN_NETANYAHU)

# Benjamin Netanyahu lexicon based analysis for April 2017

articles_file_name = "ArticlesApril2017.csv"
keywords_articles_ids = getArticles(Netanyahu_keyword_list, articles_file_name)
comments_file_name = "CommentsApril2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BENJAMIN_NETANYAHU)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BENJAMIN_NETANYAHU)

# Benjamin Netanyahu lexicon based analysis for May 2017

articles_file_name = "ArticlesMay2017.csv"
keywords_articles_ids = getArticles(Netanyahu_keyword_list, articles_file_name)
comments_file_name = "CommentsMay2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BENJAMIN_NETANYAHU)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BENJAMIN_NETANYAHU)

# Benjamin Netanyahu lexicon based analysis for January 2018

articles_file_name = "ArticlesJan2018.csv"
keywords_articles_ids = getArticles(Netanyahu_keyword_list, articles_file_name)
comments_file_name = "CommentsJan2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BENJAMIN_NETANYAHU)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BENJAMIN_NETANYAHU)

# Benjamin Netanyahu lexicon based analysis for February 2018

articles_file_name = "ArticlesFeb2018.csv"
keywords_articles_ids = getArticles(Netanyahu_keyword_list, articles_file_name)
comments_file_name = "CommentsFeb2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BENJAMIN_NETANYAHU)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BENJAMIN_NETANYAHU)

# Benjamin Netanyahu lexicon based analysis for March 2018

articles_file_name = "ArticlesMarch2018.csv"
keywords_articles_ids = getArticles(Netanyahu_keyword_list, articles_file_name)
comments_file_name = "CommentsMarch2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BENJAMIN_NETANYAHU)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BENJAMIN_NETANYAHU)

# Benjamin Netanyahu lexicon based analysis for April 2018

articles_file_name = "ArticlesApril2018.csv"
keywords_articles_ids = getArticles(Netanyahu_keyword_list, articles_file_name)
comments_file_name = "CommentsApril2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BENJAMIN_NETANYAHU)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BENJAMIN_NETANYAHU)
