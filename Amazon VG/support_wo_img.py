import random
import time

from torch.utils.data import Dataset
import sys
import os
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from preprocess import serial_asin_category
# from extract_img_feature import get_img_feature_pickle


def serialize_user(user_set):
    user_set = set(user_set)
    user_idx = 0
    # key: user原始下标，value: user有序下标
    user_serialize_dict = {}
    for user in user_set:
        user_serialize_dict[user] = user_idx
        user_idx += 1
    return user_serialize_dict


# 输入user和item的set，输出user和item从1到n有序的字典
def serialize_item(item_set):
    item_set = set(item_set)
    item_idx = 0
    item_serialize_dict = {}
    for item in item_set:
        item_serialize_dict[item] = item_idx
        item_idx += 1
    return item_serialize_dict


def sample_negative_user(user_set, interaction_user_set):
    users = set(interaction_user_set)
    candidate_users = set(user_set) - set(users)
    return random.sample(list(candidate_users), 1)[0]


# 新建一个user-item的交互字典
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


# 新建一个item-user的交互字典
def build_item_user_interaction_dict(train_csv='data/train_rating.csv',
                                     item_user_interaction_dict_save='pkl/item_user_interaction_dict.pkl'):
    if os.path.exists(item_user_interaction_dict_save) is True:
        print('从缓存中加载', item_user_interaction_dict_save)
        pkl_file = open(item_user_interaction_dict_save, 'rb')
        data = pickle.load(pkl_file)
        return data['item_user_interaction_dict']
    if os.path.exists("pkl") is False:
        os.makedirs("pkl")
    df = pd.read_csv(train_csv)
    item_user_interaction_dict = {}
    for _, row in tqdm(df.iterrows()):
        movie = row['asin']
        user = row['reviewerID']
        res = item_user_interaction_dict.get(movie)
        if res is None:
            item_user_interaction_dict[movie] = [user]
        else:
            res.append(user)
            item_user_interaction_dict[movie] = res
    with open(item_user_interaction_dict_save, 'wb') as file:
        pickle.dump({'item_user_interaction_dict': item_user_interaction_dict}, file)
    return item_user_interaction_dict


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RatingDataset:
    def __init__(self, train_csv, genres, category_num, user_serialize_dict,  positive_number, negative_number):
        self.train_csv = train_csv
        # 读其他内容
        # self.img_feature_dict = img_features
        self.genres_dict = genres
        # print(self.item_pn_df)
        self.user = self.train_csv["reviewerID"]
        self.item = self.train_csv["asin"]
        self.rating = self.train_csv["rating"]
        self.neg_user = self.train_csv['neg_user']
        self.item_set = set(self.item)
        # 序列化user和item
        self.user_serialize_dict = user_serialize_dict
        self.item_serialize_dict = serialize_item(self.item)
        # 返回个数时，返回全集的user数和训练集的item数
        self.user_number = len(user_serialize_dict)
        self.item_number = len(set(self.item))
        self.positive_number = positive_number
        self.negative_number = negative_number
        self.category_num = category_num
        self.user_item_interaction_dict = build_user_item_interaction_dict()
        self.item_user_interaction_dict = build_item_user_interaction_dict()
        print("整个数据集的user个数为:", self.user_number, "train_set中的用户数目为:", len(set(self.user)))

    def __len__(self):
        return len(self.train_csv)

    def run_all_vec(self):
        # Assuming self.train_csv is a Pandas DataFrame
        # Apply the vectorized function to each row
        results = self.train_csv.apply(self.run_index_vectorized, axis=1)

        # Convert the results into separate lists
        user_list = results.apply(lambda x: x[0]).tolist()
        item_list = results.apply(lambda x: x[1]).tolist()
        genres_list = results.apply(lambda x: x[2]).tolist()
        neg_user_list = results.apply(lambda x: x[3]).tolist()
        positive_items_list = results.apply(lambda x: x[4]).tolist()
        negative_item_list = results.apply(lambda x: x[5]).tolist()
        self_neg_list = results.apply(lambda x: x[6]).tolist()

        # Convert lists to tensors and move them to the specified device
        device = 'cuda'  # or 'cpu', depending on your setup
        user_tensor = torch.tensor(user_list)
        item_tensor = torch.tensor(item_list)
        genres_tensor = torch.tensor(genres_list)
        neg_user_tensor = torch.tensor(neg_user_list)
        positive_items_tensor = torch.tensor(positive_items_list)
        negative_item_tensor = torch.tensor(negative_item_list)
        self_neg_tensor = torch.tensor(self_neg_list)

        dataset = {
            'user': user_tensor,
            'item': item_tensor,
            'genres': genres_tensor,
            'neg_user': neg_user_tensor,
            'positive_items': positive_items_tensor,
            'negative_item': negative_item_tensor,
            'self_neg': self_neg_tensor
        }

        torch.save(dataset, 'data/dataset-25.pt')
        return user_tensor, item_tensor, genres_tensor, neg_user_tensor, positive_items_tensor, negative_item_tensor, self_neg_tensor

    def run_all(self):
        user_list = []
        item_list = []
        genres_list = []
        neg_user_list = []
        positive_items_list = []
        negative_item_list = []
        self_neg_list = []
        for index, _ in tqdm(self.train_csv.iterrows()):
            user, item, genres, neg_user, positive_items = self.run_index(index)
            user_list.append(user)
            item_list.append(item)
            genres_list.append(genres)
            neg_user_list.append(neg_user)
            positive_items_list.append(positive_items)
            # negative_item_list.append(negative_items)
            # self_neg_list.append(self_neg)

        user_tensor = torch.tensor(user_list)
        item_tensor = torch.tensor(item_list)
        genres_tensor = torch.tensor(genres_list)
        neg_user_tensor = torch.tensor(neg_user_list)
        positive_items_tensor = torch.tensor(positive_items_list)
        # negative_item_tensor = torch.tensor(negative_item_list)
        # self_neg_tensor = torch.tensor(self_neg_list)
        dataset = {
            'user': user_tensor,
            'item': item_tensor,
            'genres': genres_tensor,
            'neg_user': neg_user_tensor,
            'positive_items': positive_items_tensor,
            # 'negative_item': negative_item_tensor,
            # 'self_neg': self_neg_tensor
        }

        torch.save(dataset, 'data/dataset-new.pt')
        return user_tensor, item_tensor, genres_tensor, neg_user_tensor, positive_items_tensor #, negative_item_tensor, self_neg_tensor
    def run_index(self, index):
        user = self.user[index]
        item = self.item[index]
        # 处理 item genres
        genres = np.full((self.category_num, 1), -1)
        if item in self.genres_dict:
            genres_index = self.genres_dict.get(item)
            if len(genres_index) > 0:
                genres[genres_index] = 1
        # genres = genres.squeeze(dim=1)
        # 处理 item feature
        # img_feature = self.img_feature_dict.get(item)
        # get_item_start = time.time()
        # sample neg user spend a lot time.
        # interaction_user_set = self.item_user_interaction_dict.get(item)
        # neg_user = sample_negative_user(self.user, interaction_user_set)
        neg_user = self.neg_user[index]
        # print('sample neg user:', time.time()-get_item_start)
        # --------------------- #
        #  处理 positive items   #
        #  runtime sampling     #
        # --------------------- #
        positive_items_ = self.user_item_interaction_dict.get(user, [])
        # positive_items = list(np.random.choice(list(positive_items_), self.positive_number, replace=True))
        positive_items_list = [self.item_serialize_dict.get(item) for item in positive_items_][:20]
        # runtime sampling negative
        # negative_item_list = []
        # neg_item_set = list(self.item_set - set(positive_items_))
        # merge multi negative sample result
        negative_items_ = list(np.random.choice(neg_item_set, self.negative_number*(self.positive_number+1), replace=True))
        # negative_items_ = [self.item_serialize_dict.get(it) for it in negative_items_]
        # for i in range(self.positive_number):
        #     start_idx = self.negative_number*i
        #     end_idx = self.negative_number*(i+1)
        #     negative_item_list.append(negative_items_[start_idx:end_idx])
        # self neg list 完成 序列化, self的抽样放在和collaborative items中一起抽样负例子，最后分割出来就行了
        # self_neg_list = negative_items_[self.positive_number*self.negative_number:]
        # serialize
        user = self.user_serialize_dict.get(user)
        item = self.item_serialize_dict.get(item)
        neg_user = self.user_serialize_dict.get(neg_user)
        # user = torch.tensor(user).to(device)
        # item = torch.tensor(item).to(device)
        # neg_user = torch.tensor(neg_user).to(device)
        # positive_items_list = torch.tensor(positive_items_list).to(device)
        # negative_item_list = torch.tensor(negative_item_list).to(device)
        # self_neg_list = torch.tensor(self_neg_list).to(device)
        # genres = genres.to(device)
        return user, item, genres, neg_user, positive_items_list #, negative_item_list, self_neg_list

    def run_index_vectorized(self, row):
        # Access row elements
        user = row['reviewerID']
        item = row['asin']

        # Vectorized operation for genres
        # Assuming self.genres_dict is a dictionary that maps items to genre indexes
        genres = np.full((self.category_num, 1), -1)
        if item in self.genres_dict:
            genres_index = self.genres_dict[item]
            genres[genres_index] = 1

        # Handling negative user (assuming self.neg_user is a precomputed column)
        neg_user = row['neg_user']

        # Vectorized handling for positive items
        # Assuming 'positive_items_' is precomputed as a list of items for each user
        positive_items_ = self.user_item_interaction_dict.get(user, [])
        positive_items = np.random.choice(positive_items_, self.positive_number, replace=True)
        positive_items_list = [self.item_serialize_dict.get(item) for item in positive_items]

        # Vectorized handling for negative items
        neg_item_set = list(self.item_set - set(positive_items_))
        negative_items_ = np.random.choice(neg_item_set, self.negative_number * (self.positive_number + 1),
                                           replace=True)
        negative_items_ = [self.item_serialize_dict.get(it) for it in negative_items_]

        # Split negative items into negative_item_list and self_neg_list
        negative_item_list = [negative_items_[i * self.negative_number:(i + 1) * self.negative_number] for i in
                              range(self.positive_number)]
        self_neg_list = negative_items_[self.positive_number * self.negative_number:]

        # Serialize user, item, and neg_user
        user = self.user_serialize_dict.get(user)
        item = self.item_serialize_dict.get(item)
        neg_user = self.user_serialize_dict.get(neg_user)

        return user, item, genres, neg_user, positive_items_list, negative_item_list, self_neg_list


# 测试数据封装
if __name__ == '__main__':
    print("support.py")

    asin_category_int_map, category_ser_map = serial_asin_category()
    # img_feature_dict = get_img_feature_pickle()
    category_length = len(category_ser_map)
    # (self, train_csv, img_features, genres, user_serialize_dict, positive_number, negative_number)
    train_csv = pd.read_csv("data/train_withneg_rating.csv")
    print("ratings.length:", train_csv.__len__())
    total_user_set = train_csv['reviewerID']
    user_ser_dict = serialize_user(total_user_set)
    # all_ratings = pd.read_csv("data/ratings_filter.csv")
    # user_ser_dict = serialize_user(all_ratings["reviewerID"])
    dataset = RatingDataset(train_csv, asin_category_int_map, category_length, user_ser_dict, 10, 20)
    # dataIter = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    # it = dataIter.__iter__()
    user, item, genres, neg_user, positive_items_list, negative_item_list, self_neg_list = dataset.run_all()
    print(dataset.__len__())
    # """Train for a single epoch."""
