# This is a sample Python script.

#TODO: REFERENCES
#https://theappsolutions.com/blog/development/app-like-yelp-development/
#https://www.kaggle.com/fahd09/yelp-dataset-surpriseme-recommendation-system
#https://towardsdatascience.com/yelp-restaurant-recommendation-system-capstone-project-264fe7a7dea1
#https://medium.com/web-mining-is688-spring-2021/yelp-restaurant-rating-prediction-and-recommendation-system-fffd26fcb57a
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import streamlit as st
import unicodedata
import json
import joblib
import string
import itertools
import re
from collections import Counter
from collections import OrderedDict
from tqdm import tqdm
import pandas as pd
import nltk
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import seaborn as sns

path2qualityindian='app_data/t3_indian_50_dishes.txt'

st.cache(suppress_st_warning=True)
def get_yelp_data():
    url_bsuisiness_id = 'app_data/j_ind_business_id.jlib'
    url_restaurant_name = 'app_data/j_ind_restaurant_name.jlib'
    url_res_city = 'app_data/j_ind_city.jlib'
    url_reviews = 'app_data/j_reviews.jlib'
    url_stars = 'app_data/ind_stars.jlib'
    url_rest_name = 'app_data/j_ind_rest_names.jlib'
    # Load data (deserialize)
    business_id = joblib.load(url_bsuisiness_id )
    restaurant_name = joblib.load(url_restaurant_name )
    res_city = joblib.load(url_res_city )
    reviews = joblib.load(url_reviews )
    stars = joblib.load(url_stars )
    rest_name = joblib.load(url_rest_name )
    return business_id, restaurant_name, res_city, reviews, stars, rest_name

st.cache()
def load_dishes(path):
    dishes_df = pd.read_csv(path)

    return dishes_df

st.cache()
def get_dish_mention_count(reviews, dish_list):
    dish_counter = dict(zip(dish_list, [0] * len(dish_list)))
    for dish in dish_list:
        for review in reviews:
            review = review.replace("\t", " ") \
                .replace("\n", "") \
                .replace("\r", "") \
                .lower().strip()
            dish_counter[dish] += review.count(dish)
    N = len(dish_list) + 1
    sorted_dish_counter = {k: v for k, v in sorted(dish_counter.items(), key=lambda item: item[1], reverse=True)}
    topN = dict(itertools.islice(sorted_dish_counter.items(), N))
    dish_count_df = pd.DataFrame(topN.items(), columns=["dish_name", "value"])
    return dish_count_df

st.cache()
def get_popularity_number_of_restaurants(rest_name, dish_list, reviews):
    unique_restaurants = list(set(rest_name))
    dish_rest_counter = dict(zip(dish_list, [0] * len(dish_list)))

    for dish in dish_list:
        # for each dish, keep track of the unique restaurants that mention the particular dish
        restaurant_tracker = dict(zip(unique_restaurants, [0] * len(unique_restaurants)))
        for i, review in enumerate(reviews):
            review = review.replace("\t", " ") \
                .replace("\n", "") \
                .replace("\r", "") \
                .lower().strip()

            if dish in review:
                restaurant_tracker[rest_name[i]] = 1
        dish_rest_counter[dish] += sum(restaurant_tracker.values())
    N = 75
    sorted_dish_counter = {k: v for k, v in sorted(dish_rest_counter.items(), key=lambda item: item[1], reverse=True)}
    topN = dict(itertools.islice(sorted_dish_counter.items(), N))
    dish_count_df = pd.DataFrame(topN.items(), columns=["dish_name", "value"])
    return dish_count_df

st.cache()
def get_popularity_average_rating(stars, reviews, dish_list):
    # initialize dataframe
    dish_rating_df = pd.DataFrame(columns=['dish_name', 'Total_Rating', "Review_Count"])
    dish_rating_df['dish_name'] = dish_list
    dish_rating_df['Total_Rating'] = [0] * len(dish_list)
    dish_rating_df['Review_Count'] = [0] * len(dish_list)

    for i, dish in enumerate(dish_list):
        for j, review in enumerate(reviews):
            if stars[j] == 3:  # skip "neutral" reviews by stars
                continue
            review = review.replace("\t", " ") \
                .replace("\n", "") \
                .replace("\r", "") \
                .lower().strip()
            if dish in review:
                dish_rating_df.loc[i, 'Review_Count'] += 1  # used for average
                dish_rating_df.loc[i, 'Total_Rating'] += stars[j]

    # compute average rating
    dish_rating_df['Average_Rating'] = dish_rating_df['Total_Rating'] / dish_rating_df['Review_Count']
    # Use # of reviews as tiebreaker if needed
    dish_rating_df = dish_rating_df.sort_values(by=['Average_Rating', 'Review_Count'], axis=0, ascending=False)
    dish_rating_df_chart = dish_rating_df[dish_rating_df['Average_Rating'] > 1]
    dish_rating_df_chart = dish_rating_df_chart.sort_values(by=['Average_Rating', 'Review_Count'], axis=0,
                                                            ascending=False)
    return dish_rating_df_chart

st.cache()
def get_popularty_sentiment(dish_list, reviews, stars):

    # initialize dataframe
    dish_rating_df = get_popularity_average_rating(stars, reviews, dish_list)
    dish_sentiment_df = pd.DataFrame(columns=['dish_name', 'Total_Sentiment', "Review_Count"])
    dish_sentiment_df['dish_name'] = dish_list
    dish_sentiment_df['Total_Sentiment'] = [0] * len(dish_list)
    dish_sentiment_df['Review_Count'] = [0] * len(dish_list)



    for i, dish in enumerate(dish_list):
        for j, review in enumerate(reviews):
            if stars[j] == 3:  # skip neutral reviews by stars
                continue
            review = review.replace("\t", " ") \
                .replace("\n", "") \
                .replace("\r", "") \
                .lower().strip()
            if dish in review:
                toAnalyze = TextBlob(review)  # sentiment analysis part
                sent = toAnalyze.sentiment.polarity
                scaled_sent = 5 * (sent + 1)
                # Could use nltk VADER for sentiment analysis instead
                # analyzer = SentimentIntensityAnalyzer()
                # analyzer.polarity_scores(review)
                dish_sentiment_df.loc[i, 'Review_Count'] += 1  # used for average
                dish_sentiment_df.loc[i, 'Total_Sentiment'] += scaled_sent

            # compute average rating
    dish_sentiment_df['Average_Sentiment'] = dish_sentiment_df['Total_Sentiment'] / dish_rating_df['Review_Count']
    # Sort and use # of reviews as tiebreaker if needed
    dish_sentiment_df = dish_sentiment_df.sort_values(by=['Average_Sentiment', 'Review_Count'], axis=0, ascending=False)
    return dish_sentiment_df

st.cache()
def get_rating_recommendations(rest_name, reviews, stars, selected_dishes, res_city):

    #selected_dishes = ['mango lassi', 'chicken tikka', 'chicken tikka masala', 'tikka masala', 'biryani', 'naan']
    unique_restaurants = list(set(rest_name))
    rest_total_rating = OrderedDict(zip(unique_restaurants, [0] * len(unique_restaurants)))
    rest_review_counter = OrderedDict(zip(unique_restaurants, [0] * len(unique_restaurants)))

    #st.write(reviews)
    for i, review in enumerate(reviews):
        review = review.replace("\t", " ") \
            .replace("\n", "") \
            .replace("\r", "") \
            .lower().strip()
        for dish in selected_dishes:

            if dish in review:
                rest_review_counter[rest_name[i]] += 1 # used for average

                rest_total_rating[rest_name[i]] += int(stars[i])

    rest_rating_df = pd.DataFrame(columns=['Restaurant_Name', 'Total_Rating', "Review_Count"])
    rest_rating_df['Restaurant_Name'] = list(rest_total_rating.keys())

    rest_rating_df['Total_Rating'] = list(rest_total_rating.values())
    rest_rating_df['Review_Count'] = list(rest_review_counter.values())
    rest_rating_df['Average_Rating'] = (rest_rating_df['Total_Rating'] + 1e-3) / (
                rest_rating_df['Review_Count'] + 1e-3)
    # sort restaurants by average rating and use # of reviews as tiebreaker if needed
    # drop restaurants with fewer than a predetermined minimum threshold of reviews
    # select the top 10 (as Yelp often displays the results)
    rest_rating_df = rest_rating_df.sort_values(by=['Average_Rating', 'Review_Count'], axis=0, ascending=False) \
        .reset_index(drop=True)

    top10_A = rest_rating_df[rest_rating_df['Review_Count'] >= 3].head(10)
    #st.write(rest_rating_df)
    return top10_A

st.cache()
def get_sentiment_recommendations(rest_name, reviews, stars, selected_dishes, res_city):
    unique_restaurants = list(set(rest_name))
    rest_total_sentiment = OrderedDict(zip(unique_restaurants, [0] * len(unique_restaurants)))
    rest_review_counter = OrderedDict(zip(unique_restaurants, [0] * len(unique_restaurants)))

    for i, review in enumerate(reviews):
        review = review.replace("\t", " ").replace("\n", "").replace("\r", "").lower().strip()
        # skip "neutral" reviews by stars
        for dish in selected_dishes:
            if dish in review:
                toAnalyze = TextBlob(review)  # sentiment analysis part
                sent = toAnalyze.sentiment.polarity
                scaled_sent = 5 * (sent + 1)
                rest_review_counter[rest_name[i]] += 1  # used for average
                rest_total_sentiment[rest_name[i]] += scaled_sent
    rest_sentiment_df = pd.DataFrame(columns=['Restaurant_Name', 'Total_Sentiment', "Review_Count"])
    rest_sentiment_df['Restaurant_Name'] = list(rest_total_sentiment.keys())
    rest_sentiment_df['Total_Sentiment'] = list(rest_total_sentiment.values())
    rest_sentiment_df['Review_Count'] = list(rest_review_counter.values())
    rest_sentiment_df['Average_Sentiment'] = (rest_sentiment_df['Total_Sentiment'] + 1e-3) / (
                rest_sentiment_df['Review_Count'] + 1e-3)
    # sort restaurants by average sentiment and use # of reviews as tiebreaker if needed
    # drop restaurants with fewer than a predetermined minimum threshold of reviews
    # select the top 10 (as Yelp often displays the results)
    rest_sentiment_df = rest_sentiment_df.sort_values(by=['Average_Sentiment', 'Review_Count'], axis=0, ascending=False) \
        .reset_index(drop=True)

    top10_B = rest_sentiment_df[rest_sentiment_df['Review_Count'] >= 3].head(10)
    return top10_B

st.cache()
def get_dish_list(path):
    # extract the "quality" Indian dishes from task 3
    dish_list = []
    with open(path, 'r') as f:
        for line in f:
            dish_list.append(unicodedata.normalize('NFKD', line).lower().strip())


    return dish_list

def main():
    st.title("Yelpy Dish India")
    menu = ["Home","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Home":
        dishes_df = load_dishes('app_data/indian_popular_dishes.csv')
        dish_list = get_dish_list('app_data/t3_indian_50_dishes.txt')
        business_id, restaurant_name, res_city, reviews, stars, rest_name = get_yelp_data()

        st.subheader('CS598 Data Mining Capstone - Summer 2021, Sembian2 at illinois.edu')
        st.write("Interactive Experience to explore Popular Indian Cuisine Dish & Restaurant Recommendations")
        st.sidebar.header('Yelply Dish Options')
        dish_form = st.sidebar.form(key='dish_form')
        #st.write(dishes_df)
        options = dish_form.radio('Find Popular Dish By',
                                 ['Count','Restaurant Mentions', 'Average Rating', 'Review Sentiment'])
        dishes_button = dish_form.form_submit_button(label="Show Popular Dishes")
        dish_selected = pd.DataFrame(columns=['dish_name', 'value'])
        if options == 'Count':
            dish_selected = get_dish_mention_count(reviews, dish_list)
        if options == 'Restaurant Mentions':
            dish_selected = get_popularity_number_of_restaurants(rest_name, dish_list, reviews)
        if options == 'Average Rating':
            dish_selected = get_popularity_average_rating(stars, reviews, dish_list)
        if options == 'Review Sentiment':
            dish_selected = get_popularty_sentiment(dish_list, reviews, stars)


        if dishes_button:
            st.sidebar.success("Show Dish Selected {} ".format(options))
            st.write(dish_selected['dish_name'])
        dish_options = st.multiselect('Select Popular Indian Dishes', dish_selected['dish_name'])

        rec_radio = st.radio('Restaurant Recommendations',
                                 ['Average Rating','Sentiment'])
        rec_button = st.button("Show Recommendations")
        if rec_button:
            if rec_radio == 'Average Rating':
                restaurants = get_rating_recommendations(rest_name, reviews, stars, dish_options, res_city)
                st.write("Showing Top 10 Restaurant recommendations based on Average Dish Rating")

                st.write(restaurants['Restaurant_Name'])
            if rec_radio == 'Sentiment':
                restaurants = get_sentiment_recommendations(rest_name, reviews, stars, dish_options, res_city)
                st.write("Showing Top 10 Restaurant recommendations based on Review Sentiment")


                st.write(restaurants['Restaurant_Name'])
    else:
        st.subheader("About")
        st.write(get_dish_list('app_data/t3_indian_50_dishes.txt'))
if __name__ == '__main__':
    main()


#TESTING
# st.success('Yelpy Success')
# st.warning('Yelpy Warning')
# st.info('Yelpy Info')
# st.error("Yelply Error")
# button1 = st.button("Click Me")

#MultiSelect  - https://www.youtube.com/watch?v=LU0ZUqCgNzw
#st.multiselect('Multiselect',[1,2,3])





# HIDING Streamlit Main Menu and Footer
hide_streamlit_style="""
<style>
#MainMenu{visibility:hidden}
footer{visibility:hidden}
</style>"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)