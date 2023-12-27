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

#items_df['cat_2'].fillna(items_df['cat_1'],inplace=True)
all_df['cat_2'][(all_df['cat_2'])=='NA']= all_df['cat_1']
all_df['cat_3'][(all_df['cat_3'])=='NA']= all_df['cat_2']
print(all_df.head())

interactions_df = all_df[['user_id','product_id','event_type']]
print(interactions_df.head())

print(interactions_df.loc[interactions_df['user_id'].isin(['513339512'])].head())

items_df = all_df[['product_id','brand','price','cat_0','cat_1','cat_2','cat_3']].drop_duplicates().reset_index(drop=True)
items_df=items_df.drop_duplicates(subset='product_id', keep="first").reset_index(drop=True)
print(items_df.head())

#Shrinking the data

#from sklearn.model_selection import train_test_split
#interactions_df_waste,interactions_df=train_test_split(interactions_df0,test_size=0.99,random_state=1234)

# 2. Some Basic Insight

event_type_strength = {
   'purchase': 5,
   'cart': 0,  
}

interactions_df['eventStrength'] = interactions_df['event_type'].apply(lambda x: event_type_strength[x])

users_interactions_count_df = interactions_df.groupby(['user_id', 'product_id']).size().groupby('user_id').size()
print('# users: %d' % len(users_interactions_count_df))
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 20].reset_index()[['user_id']]
print('# users with at least 10 interactions: %d' % len(users_with_enough_interactions_df))

print('# of interactions: %d' % len(interactions_df))
interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df, 
               how = 'right',
               left_on = 'user_id',
               right_on = 'user_id')
print('# of interactions from users with at least 10 interactions: %d' % len(interactions_from_selected_users_df))

interactions_full_df = interactions_from_selected_users_df.groupby(['user_id', 'product_id'])['eventStrength'].sum().reset_index()
print('# of unique user/item interactions: %d' % len(interactions_full_df))
interactions_full_df.head(10)

interactions_full_df.shape

def create_data_dummy(data):
    data_dummy = data.copy()
    data_dummy['purchase_dummy'] = 1
    return data_dummy
data_dummy = create_data_dummy(interactions_full_df)


def split_data(data):
    train, test = train_test_split(data, test_size = 0.2)
    valid, test = train_test_split(test, test_size = 0.5)
    train_data = tc.SFrame(train)
    valid_data = tc.SFrame(valid)
    test_data = tc.SFrame(test)
    return train_data, test_data,valid_data

train_data, test_data,valid_data = split_data(interactions_full_df)
train_data_dummy, test_data_dummy ,valid_data_dummy = split_data(data_dummy)


items_of_selected_users_df = interactions_full_df.merge(items_df, 
               how = 'inner',
               left_on = 'product_id',
               right_on = 'product_id')

del items_of_selected_users_df['user_id']
del items_of_selected_users_df['eventStrength']

items_of_selected_users_df=items_of_selected_users_df.drop_duplicates(subset='product_id', keep="first").reset_index(drop=True)
items_of_selected_users_df

data_dummy.head(10)

from tqdm import tqdm

chunk_size = 50000
chunks = [x for x in range(0, interactions_full_df.shape[0], chunk_size)]

for i in range(0, len(chunks) - 1):
    print(chunks[i], chunks[i + 1] - 1)
pivot_df = pd.DataFrame()

for i in tqdm(range(0, len(chunks) - 1)):
    chunk_df = interactions_full_df.iloc[ chunks[i]:chunks[i + 1] - 1]
    interactions = (chunk_df.groupby(['user_id', 'product_id'])['eventStrength']
      .sum()
      .unstack()
      .reset_index()
      .fillna(0)
      .set_index('user_id')
    )
    print (interactions.shape)
    pivot_df = pd.concat([pivot_df, interactions], ignore_index=True)
 

df_matrix=pivot_df

#And then I have to make a sparse matrix as input to lightFM recommendation model (run matrix-factorization algorithm). You can use it for any use case where unstacking is required. Using the following code, converted to sparse matrix-

#from scipy import sparse
#import numpy as np
#sparse_matrix = sparse.csr_matrix(df_new.to_numpy())

#df_matrix = pd.pivot_table(interactions_full_df, values='eventStrength', index='user_id', columns='product_id')

df_matrix.shape

df_matrix

# constant variables to define field names include:
Sitems_df = tc.SFrame(items_df)
user_id = 'user_id'
item_id = 'product_id'
users_to_recommend = list(test_data[user_id])
n_rec = 10 # number of items to recommend
n_display = 30 # to display the first few rows in an output dataset

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
    #Zahra->    
    elif name == 'factorization':
        model = tc.factorization_recommender.create(train_data, 
                                                    user_id=user_id, 
                                                    item_id=item_id, 
                                                    target=target)
        
    elif name == 'content':
        model = tc.item_content_recommender.create(observation_data=train_data,
                                                   item_data=Sitems_df, 
                                                    user_id=user_id, 
                                                    item_id=item_id, 
                                                    target=target)        


     #->Zahra   
    recom = model.recommend(users=users_to_recommend, k=n_rec)
    recom.print_rows(n_display)
    return model

#Sitems_df = tc.SFrame(items_df)
#m = tc.item_content_recommender.create(item_data=Sitems_df,item_id='product_id')
#m = tc.recommender.item_content_recommender.create(item_data=Sitems_df,item_id='product_id')
#m.recommend(item_data=Sitems_df,item_id='product_id')

name = 'content'
target = 'eventStrength'
content = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

name = 'content'
target = 'purchase_dummy'
content_dummy = model(train_data_dummy, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

name = 'factorization'
target = 'eventStrength'
factorization = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

name = 'factorization'
target = 'purchase_dummy'
fact_dummy = model(train_data_dummy, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

name = 'popularity'
target = 'eventStrength'
popularity = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

name = 'popularity'
target = 'purchase_dummy'
pop_dummy = model(train_data_dummy, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

name = 'cosine'
target = 'eventStrength'
cos = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)


name = 'cosine'
target = 'purchase_dummy'
cos_dummy = model(train_data_dummy, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

name = 'pearson'
target = 'eventStrength'
pear = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

name = 'pearson'
target = 'purchase_dummy'
pear_dummy = model(train_data_dummy, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

models_w_counts = [popularity, cos, pear,factorization,content]
models_w_dummy = [pop_dummy, cos_dummy, pear_dummy,fact_dummy,content_dummy]
names_w_counts = ['Popularity Model on Purchase Counts', 'Cosine Similarity on Purchase Counts', 'Pearson Similarity on Purchase Counts','Factorization Model on Purchase Counts','Content based Model on Purchase Counts']
names_w_dummy = ['Popularity Model on Purchase Dummy', 'Cosine Similarity on Purchase Dummy', 'Pearson Similarity on Purchase Dummy','Factorization Model on Purchase Dummy','Content based Model on Purchase Dummy']


eval_counts = tc.recommender.util.compare_models(test_data, models_w_counts, model_names=names_w_counts)
eval_dummy = tc.recommender.util.compare_models(test_data_dummy, models_w_dummy, model_names=names_w_dummy)

#models_w_counts = [popularity, cos, pear,factorization,content]
#models_w_dummy = [pop_dummy, cos_dummy, pear_dummy,fact_dummy,content_dummy]
#names_w_counts = ['Popularity Model on Purchase Counts', 'Cosine Similarity on Purchase Counts', 'Pearson Similarity on Purchase Counts','Factorization Model on Purchase Counts','Content based Model on Purchase Counts']
#names_w_dummy = ['Popularity Model on Purchase Dummy', 'Cosine Similarity on Purchase Dummy', 'Pearson Similarity on Purchase Dummy','Factorization Model on Purchase Dummy','Content based Model on Purchase Dummy']
eval_counts = tc.recommender.util.compare_models(valid_data, models_w_counts, model_names=names_w_counts)
eval_dummy = tc.recommender.util.compare_models(valid_data_dummy, models_w_dummy, model_names=names_w_dummy)