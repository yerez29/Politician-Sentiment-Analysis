# IMPORTS

import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from copy import deepcopy
from math import floor

# CONSTS

ENTIRE_MONTH = "Entire Month"

# POLITICIANS

DONALD_TRUMP = "Donald Trump"
HILLARY_CLINTON = "Hillary Clinton"
BARACK_OBAMA = "Barack Obama"
BERNIE_SANDERS = "Bernie Sanders"


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


def getGeneralOpinions(relevant_comments, comments_file_name, segment_num, politician):
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
    num_of_comments = len(relevant_comments)
    date = comments_file_name[8:len(comments_file_name) - 4]
    for comment in relevant_comments:
        analysis = TextBlob(comment)
        cur_average_mark = analysis.sentences[0].sentiment.polarity
        general_mark += cur_average_mark
        if cur_average_mark > 0.00:
            if cur_average_mark > 0.3:
                very_positive_mark += 1
            else:
                slightly_positive_mark += 1
        elif cur_average_mark < 0.00:
            if cur_average_mark < -0.3:
                very_negative_mark += 1
            else:
                slightly_negative_mark += 1
        else:
            natural_mark += 1
    general_mark /= num_of_comments
    very_positive_mark = percent(very_positive_mark, num_of_comments)
    slightly_positive_mark = percent(slightly_positive_mark, num_of_comments)
    slightly_negative_mark = percent(slightly_negative_mark, num_of_comments)
    very_negative_mark = percent(very_negative_mark, num_of_comments)
    natural_mark = percent(natural_mark, num_of_comments)
    labels = 'Very positive', 'Slightly positive', 'Slightly negative', 'Very negative', 'Natural'
    sizes = [very_positive_mark, slightly_positive_mark, slightly_negative_mark, very_negative_mark, natural_mark]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    if segment_num == ENTIRE_MONTH:
        print("total normalized mark in " + date + " of " + politician + " is " + str(general_mark))
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
    rates_dict = {"very pos" : 0, "slightly pos" : 0, "natural" : 0, "slightly neg" : 0, "very neg" : 0}
    for country in list_of_countries:
        countries_dict[country[1]] = deepcopy(rates_dict)
    i = 0
    for comment in relevantComments:
        analysis = TextBlob(comment)
        comment_mark = analysis.sentences[0].sentiment.polarity
        if comment_mark > 0.00:
            if comment_mark > 0.3:
                address = locations_column[comments_indices[i]]
                if type(address) is str:
                    flag = False
                    for country in list_of_countries:
                        for abb in country:
                            if abb in address:
                                countries_dict[country[1]]["very pos"] += 1
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
                                countries_dict[country[1]]["slightly pos"] += 1
                                flag = True
                                break
                        if flag:
                            break
        elif comment_mark < 0.00:
            if comment_mark < -0.3:
                address = locations_column[comments_indices[i]]
                if type(address) is str:
                    flag = False
                    for country in list_of_countries:
                        for abb in country:
                            if abb in address:
                                countries_dict[country[1]]["very neg"] += 1
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
                                countries_dict[country[1]]["slightly neg"] += 1
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
    date = comments_file_name[8:len(comments_file_name) - 4]
    for key in countries_dict:
        total = sum(countries_dict[key].values())
        very_positive_mark = percent(countries_dict[key]["very pos"], total)
        slightly_positive_mark = percent(countries_dict[key]["slightly pos"], total)
        slightly_negative_mark = percent(countries_dict[key]["slightly neg"], total)
        very_negative_mark = percent(countries_dict[key]["very neg"], total)
        natural_mark = percent(countries_dict[key]["natural"], total)
        labels = 'Very positive', 'Slightly positive', 'Slightly negative', 'Very negative', 'Natural'
        sizes = [very_positive_mark, slightly_positive_mark, slightly_negative_mark, very_negative_mark, natural_mark]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True,
                startangle=90)
        plt.title("Opinion distribution in " + key + " at " + date + " of " + politician)
        plt.axis('equal')
        plt.show()
        plt.clf()


def getTimeBasedOpinions(relevantComments, comments_indices, comments_file_name, politician):

    month_division = 6
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
Sanders_keyword_list = ["Sanders", "Bernie", "sanders", "bernie", "SANDERS", "BERNIE", "Bernie Sanders", "Bernie sanders", "bernie Sanders", "bernie sanders", "BERNIE SANDERS", "Bernard Sanders", "Sanders, Bernie", "Sanders, Bernard"]

# DONALD_TRUMP

# JAN 2017
articles_file_name = "ArticlesJan2017.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsJan2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)
# FEB 2017
articles_file_name = "ArticlesFeb2017.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsFeb2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)
# MARCH 2017
articles_file_name = "ArticlesMarch2017.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsMarch2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)
# APRIL 2017
articles_file_name = "ArticlesApril2017.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsApril2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)
# MAY 2017
articles_file_name = "ArticlesMay2017.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsMay2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)
# JAN 2018
articles_file_name = "ArticlesJan2018.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsJan2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)
# FEB 2018
articles_file_name = "ArticlesFeb2018.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsFeb2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)
# MARCH 2018
articles_file_name = "ArticlesMarch2018.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsMarch2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)
# APRIL 2018
articles_file_name = "ArticlesApril2018.csv"
keywords_articles_ids = getArticles(Trump_keywords_list, articles_file_name)
comments_file_name = "CommentsApril2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, DONALD_TRUMP)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, DONALD_TRUMP)


# Hillary Clinton

# JAN 2017
articles_file_name = "ArticlesJan2017.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsJan2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)
# FEB 2017
articles_file_name = "ArticlesFeb2017.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsFeb2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)
# MARCH 2017
articles_file_name = "ArticlesMarch2017.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsMarch2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)
# APRIL 2017
articles_file_name = "ArticlesApril2017.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsApril2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)
# MAY 2017
articles_file_name = "ArticlesMay2017.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsMay2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)
# JAN 2018
articles_file_name = "ArticlesJan2018.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsJan2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)
# FEB 2018
articles_file_name = "ArticlesFeb2018.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsFeb2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)
# MARCH 2018
articles_file_name = "ArticlesMarch2018.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsMarch2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)
# APRIL 2018
articles_file_name = "ArticlesApril2018.csv"
keywords_articles_ids = getArticles(Clinton_keyword_list, articles_file_name)
comments_file_name = "CommentsApril2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, HILLARY_CLINTON)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, HILLARY_CLINTON)


# BARACK OBAMA

# JAN 2017
articles_file_name = "ArticlesJan2017.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsJan2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)
# FEB 2017
articles_file_name = "ArticlesFeb2017.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsFeb2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)
# MARCH 2017
articles_file_name = "ArticlesMarch2017.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsMarch2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)
# APRIL 2017
articles_file_name = "ArticlesApril2017.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsApril2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)
# MAY 2017
articles_file_name = "ArticlesMay2017.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsMay2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)
# JAN 2018
articles_file_name = "ArticlesJan2018.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsJan2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)
# FEB 2018
articles_file_name = "ArticlesFeb2018.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsFeb2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)
# MARCH 2018
articles_file_name = "ArticlesMarch2018.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsMarch2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)
# APRIL 2018
articles_file_name = "ArticlesApril2018.csv"
keywords_articles_ids = getArticles(Obama_keyword_list, articles_file_name)
comments_file_name = "CommentsApril2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BARACK_OBAMA)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BARACK_OBAMA)


# BERNIE SANDERS

# JAN 2017
articles_file_name = "ArticlesJan2017.csv"
keywords_articles_ids = getArticles(Sanders_keyword_list, articles_file_name)
comments_file_name = "CommentsJan2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BERNIE_SANDERS)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BERNIE_SANDERS)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BERNIE_SANDERS)
# FEB 2017
articles_file_name = "ArticlesFeb2017.csv"
keywords_articles_ids = getArticles(Sanders_keyword_list, articles_file_name)
comments_file_name = "CommentsFeb2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BERNIE_SANDERS)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BERNIE_SANDERS)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BERNIE_SANDERS)
# MARCH 2017
articles_file_name = "ArticlesMarch2017.csv"
keywords_articles_ids = getArticles(Sanders_keyword_list, articles_file_name)
comments_file_name = "CommentsMarch2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BERNIE_SANDERS)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BERNIE_SANDERS)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BERNIE_SANDERS)
# APRIL 2017
articles_file_name = "ArticlesApril2017.csv"
keywords_articles_ids = getArticles(Sanders_keyword_list, articles_file_name)
comments_file_name = "CommentsApril2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BERNIE_SANDERS)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BERNIE_SANDERS)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BERNIE_SANDERS)
# MAY 2017
articles_file_name = "ArticlesMay2017.csv"
keywords_articles_ids = getArticles(Sanders_keyword_list, articles_file_name)
comments_file_name = "CommentsMay2017.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BERNIE_SANDERS)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BERNIE_SANDERS)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BERNIE_SANDERS)
# JAN 2018
articles_file_name = "ArticlesJan2018.csv"
keywords_articles_ids = getArticles(Sanders_keyword_list, articles_file_name)
comments_file_name = "CommentsJan2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BERNIE_SANDERS)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BERNIE_SANDERS)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BERNIE_SANDERS)
# FEB 2018
articles_file_name = "ArticlesFeb2018.csv"
keywords_articles_ids = getArticles(Sanders_keyword_list, articles_file_name)
comments_file_name = "CommentsFeb2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BERNIE_SANDERS)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BERNIE_SANDERS)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BERNIE_SANDERS)
# MARCH 2018
articles_file_name = "ArticlesMarch2018.csv"
keywords_articles_ids = getArticles(Sanders_keyword_list, articles_file_name)
comments_file_name = "CommentsMarch2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BERNIE_SANDERS)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BERNIE_SANDERS)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BERNIE_SANDERS)
# APRIL 2018
articles_file_name = "ArticlesApril2018.csv"
keywords_articles_ids = getArticles(Sanders_keyword_list, articles_file_name)
comments_file_name = "CommentsApril2018.csv"
relevant_comments, comments_indices = getComments(keywords_articles_ids, comments_file_name)
getGeneralOpinions(relevant_comments, comments_file_name, ENTIRE_MONTH, BERNIE_SANDERS)
getLocationBasedOpinions(relevant_comments, comments_indices, comments_file_name, BERNIE_SANDERS)
getTimeBasedOpinions(relevant_comments, comments_indices, comments_file_name, BERNIE_SANDERS)
