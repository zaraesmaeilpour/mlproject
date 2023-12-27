#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 14:32:50 2023

@author: zahra
"""

# MLProject
### Zahra
#!pip install turicreate
#!pip install matplotlib
#!pip install scikit-learn
#!pip install fastparquet


import pandas as pd
import numpy as np   
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


import turicreate as tc
from sklearn.model_selection import train_test_split

# Read the .parquet file from your local directory
parquet_file_path = '/Users/zahra/Documents/Git/machinelearning'
all_df = pd.read_parquet(parquet_file_path, engine='fastparquet')

print(all_df.head())

# Replace 'NA' values with the category above them
#items_df['cat_2'].fillna(items_df['cat_1'],inplace=True)
all_df['cat_2'][(all_df['cat_2'])=='NA']= all_df['cat_1']
all_df['cat_3'][(all_df['cat_3'])=='NA']= all_df['cat_2']


print(all_df.head())

interactions_df = all_df[['user_id', 'product_id', 'event_type']]
print(interactions_df.head())

print(interactions_df.loc[interactions_df['user_id'] == '513339512'])

items_df = all_df[['product_id', 'brand', 'price', 'cat_0', 'cat_1', 'cat_2', 'cat_3']].drop_duplicates().reset_index(drop=True)

print(items_df.head())

# Define the strength of events
event_type_strength = {
   'purchase': 5,
   'cart': 0,  
}

interactions_df['eventStrength'] = interactions_df['event_type'].apply(lambda x: event_type_strength.get(x, 0))

users_interactions_count_df = interactions_df.groupby(['user_id', 'product_id']).size().groupby('user_id').size()

print('# users: %d' % len(users_interactions_count_df))

users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 20].reset_index()[['user_id']]

print('# users with at least 20 interactions: %d' % len(users_with_enough_interactions_df))

interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df, 
                                                            how='right',
                                                            left_on='user_id',
                                                            right_on='user_id')

print('# of interactions from users with at least 20 interactions: %d' % len(interactions_from_selected_users_df))

interactions_full_df = interactions_from_selected_users_df.groupby(['user_id', 'product_id'])['eventStrength'].sum().reset_index()

print('# of unique user/item interactions: %d' % len(interactions_full_df))
print(interactions_full_df.head(10))

# Create dummy data
def create_data_dummy(data):
    data_dummy = data.copy()
    data_dummy['purchase_dummy'] = 1
    return data_dummy

data_dummy = create_data_dummy(interactions_full_df)

# Split the data into training, validation, and test sets
def split_data(data):
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    valid, test = train_test_split(test, test_size=0.5, random_state=42)
    train_data = tc.SFrame(train)
    valid_data = tc.SFrame(valid)
    test_data = tc.SFrame(test)
    return train_data, valid_data, test_data

train_data, valid_data, test_data = split_data(interactions_full_df)
train_data_dummy, valid_data_dummy, test_data_dummy = split_data(data_dummy)

# Function to build the recommendation model
def model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display):
    if name == 'popularity':
        model = tc.popularity_recommender.create(train_data, 
                                                 user_id=user_id, 
                                                 item_id=item_id, 
                                                 target=target)
    elif name == 'cosine':
        model = tc.item_similarity_recommender.create(train_data, 
                                                      user_id=user_id, 
                                                      item_id=item_id, 
                                                      target=target, 
                                                      similarity_type='cosine')
    elif name == 'pearson':
        model = tc.item_similarity_recommender.create(train_data, 
                                                      user_id=user_id, 
                                                      item_id=item_id, 
                                                      target=target, 
                                                      similarity_type='pearson')
    recom = model.recommend(users=users_to_recommend, k=n_rec)
    recom.print_rows(n_display)
    return model

# Define constants for the model
user_id = 'user_id'
item_id = 'product_id'
users_to_recommend = list(interactions_df['user_id'].unique())
n_rec = 10 # number of items to recommend
n_display = 30 # to display the first few rows in an output dataset

# Train different types of models
name = 'popularity'
target = 'eventStrength'
popularity = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

name = 'cosine'
cos = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

name = 'pearson'
pear = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

# More models can be added following the same structure

# Evaluation of the models
models_w_counts = [popularity, cos, pear]
names_w_counts = ['Popularity Model on Purchase Counts', 'Cosine Similarity on Purchase Counts', 'Pearson Similarity on Purchase Counts']

eval_counts = tc.recommender.util.compare_models(test_data, models_w_counts, model_names=names_w_counts)
