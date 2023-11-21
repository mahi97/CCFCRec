import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from tqdm import tqdm

def build_user_item_interaction_dict(train_csv='data/train_rating.csv',
                                     user_item_interaction_dict_save='pkl/user_item_interaction_dict.pkl'):
    if os.path.exists(user_item_interaction_dict_save) is True:
        print('从缓存中加载user_item_interaction_dict')
        pkl_file = open(user_item_interaction_dict_save, 'rb')
        data = pickle.load(pkl_file)
        return data['user_item_interaction_dict']
    if os.path.exists("pkl") is False:
        os.makedirs("pkl")
    df = pd.read_csv(train_csv)
    user_item_interaction_dict = {}
    for _, row in tqdm(df.iterrows()):
        movie = row['asin']
        user = row['reviewerID']
        res = user_item_interaction_dict.get(user)
        if res is None:
            user_item_interaction_dict[user] = [movie]
        else:
            res.append(movie)
            user_item_interaction_dict[user] = res
    with open(user_item_interaction_dict_save, 'wb') as file:
        pickle.dump({'user_item_interaction_dict': user_item_interaction_dict}, file)
    return user_item_interaction_dict

d = build_user_item_interaction_dict()
print(d)