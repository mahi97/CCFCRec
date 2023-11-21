import pickle
import os
import pandas as pd


def serial_song_genres(pkl_name='data/yahoo_music_genre.pkl'):
    if os.path.exists(pkl_name) is True:
        pkl_file = open(pkl_name, "rb")
        data = pickle.load(pkl_file)
        # data['asin_category_int_map']， asin: category, category为经过顺序化后的属性
        # data['category_ser_map']， category: category_int_num, category对应的顺序编号
        return data
    asin_df = pd.read_csv("data/final_song_genre_hierarchy.csv")
    asin_category_int_map = {}
    for idx, row in asin_df.iterrows():
        cat = row['category'].split(',')
        asin = row['asin']
        tmp_list = []
        for i in cat:
            tmp_list.append(category_ser_map.get(i))
        asin_category_int_map[asin] = tmp_list
    with open(pkl_name, "wb") as file:
        pickle.dump(asin_category_int_map, file)



if __name__ == '__main__':
    serial_song_genres()