#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from data_utils import encode_by_count
import pickle
import os
from model import DT

class dataset(object):
    def __init__(self,raw_csv_data_file):
        self.data = pd.read_csv('data/data.csv')
        self.action_type = self.data['action_type']
        self.combined_shot_type = self.data['combined_shot_type']
        self.game_event_id = self.data['game_event_id']
        self.game_id = self.data['game_id']
        self.lat = self.data['lat']
        self.loc_x = self.data['loc_x']
        self.loc_y = self.data['loc_y']
        self.lon = self.data['lon']
        self.minutes_remaining = self.data['minutes_remaining']
        self.period = self.data['period']
        self.playoffs = self.data['playoffs']
        self.season = self.data['season']
        self.seconds_remaining = self.data['seconds_remaining']
        self.shot_distance = self.data['shot_distance']
        self.shot_made_flag = self.data['shot_made_flag']
        self.shot_type = self.data['shot_type']
        self.shot_zone_area = self.data['shot_zone_area']
        self.shot_zone_basic = self.data['shot_zone_basic']
        self.shot_zone_range = self.data['shot_zone_range']
        self.team_id = self.data['team_id']
        self.team_name = self.data['team_name']
        self.game_date = self.data['game_date']
        self.min_game_date = self.game_date.min()
        self.matchup = self.data['matchup']
        self.opponent = self.data['opponent']
        self.shot_id = self.data['shot_id']


        
def pretreatment():
    map_dict = {}
    data = pd.read_csv('data/data.csv')
    notnull_id = data.isnull().values == True
    print notnull_id.__class__
    print data.info()
    #数值类分析
    print data.describe()
    #目标类分析
    print data.describe(include=['O'])
    print set(data['action_type'].tolist())
    print set(data['shot_zone_range'].tolist())

    #raw data
    action_type = data['action_type']
    combined_shot_type = data['combined_shot_type']
    game_event_id = data['game_event_id']
    game_id = data['game_id']
    lat = data['lat']
    loc_x = data['loc_x']
    loc_y = data['loc_y']
    lon = data['lon']
    minutes_remaining = data['minutes_remaining']
    period = data['period']
    playoffs = data['playoffs']
    season = data['season']
    seconds_remaining = data['seconds_remaining']
    shot_distance = data['shot_distance']
    shot_made_flag = data['shot_made_flag']
    shot_type = data['shot_type']
    shot_zone_area = data['shot_zone_area']
    shot_zone_basic = data['shot_zone_basic']
    shot_zone_range = data['shot_zone_range']
    team_id = data['team_id']
    team_name = data['team_name']
    game_date = data['game_date']
    matchup = data['matchup']
    opponent = data['opponent']
    shot_id = data['shot_id']

    #pretreat_data
    indices = shot_id
    columes = ['action_type','combined_shot_type','loc_x','loc_y','period','playoffs','time_remaining',
               'shot_distance','shot_type','shot_zone_area','shot_zone_basic','game_date',
               'home_away','opponent','target']
    pretreated_data = pd.DataFrame(index=indices,columns=columes)


    action_type_str2id = encode_by_count(action_type)
    map_dict['action_type'] = action_type_str2id
    pretreated_data['action_type'] = [action_type_str2id[i] for i in action_type]

    combined_shot_type_str2id = encode_by_count(combined_shot_type)
    map_dict['combined_shot_type'] = combined_shot_type_str2id
    pretreated_data['combined_shot_type'] = [combined_shot_type_str2id[i] for i in combined_shot_type]

    pretreated_data['loc_x'] = loc_x.values
    pretreated_data['loc_y'] = loc_y.values
    pretreated_data['period'] = period.values
    pretreated_data['playoffs'] = playoffs.values


    time_remaining = minutes_remaining.values * 60 + seconds_remaining.values
    pretreated_data['time_remaining'] = time_remaining

    pretreated_data['shot_distance'] = shot_distance.values

    shot_type_str2id = encode_by_count(shot_type)
    map_dict['shot_type'] = shot_type_str2id
    pretreated_data['shot_type'] = [shot_type_str2id[i] for i in shot_type]

    shot_zone_area_str2id = encode_by_count(shot_zone_area)
    map_dict['shot_zone_area'] = shot_zone_area_str2id
    pretreated_data['shot_zone_area'] = [shot_zone_area_str2id[i] for i in shot_zone_area]

    shot_zone_basic_str2id = encode_by_count(shot_zone_basic)
    map_dict['shot_zone_basic'] = shot_zone_basic_str2id
    pretreated_data['shot_zone_basic'] = [shot_zone_basic_str2id[i] for i in shot_zone_basic]

    #时间差
    #tidel格式是TimedeltaIndex
    timedel = pd.to_datetime(game_date.values) - pd.to_datetime(game_date.min())
    daydel = timedel.days.values
    pretreated_data['game_date'] = daydel

    #
    home_away = [0 if '@' in m else 1 for m in matchup]
    pretreated_data['home_away'] = home_away

    opponent_str2id = encode_by_count(opponent)
    map_dict['opponent'] = opponent_str2id
    pretreated_data['opponent'] = [opponent_str2id[i] for i in opponent]

    pretreated_data['target'] = shot_made_flag.values

    notnull = pretreated_data['target'].notnull()
    isnull = ~ notnull
    train_set = pretreated_data[notnull]
    predict_set = pretreated_data[isnull]
    train_set.to_csv('data/train_set.csv')
    predict_set.to_csv('data/predict_set.csv')
    # print train_set,predict_set
    pretreated_data.to_csv('data/pretreatment.csv')
    if not os.path.exists('model'):
        os.mkdir('model')
    f = open('model/map_dict.pkl','wb')
    pickle.dump(map_dict,f)
    f.close()
    # pretreated_data.isnull().to_csv('data/pretreatment_nan.csv')

def main():
    # pretreatment()
    train_data = pd.read_csv('data/train_set.csv')
    dt_clf = DT(train_data)
    test_data = pd.read_csv('data/predict_set.csv')
    out = dt_clf.predict(test_data)
    out_path = 'output/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    OutDf = pd.DataFrame(index = test_data['shot_id'].values,columns=['shot_made_flag'])#index = test_data['shot_id'].values,
    # OutDf['shot_id'] = test_data['shot_id'].values
    OutDf['shot_made_flag'] = out
    print out
    OutDf.to_csv(out_path+'DT_out.csv')
    # a = pd.read_csv('data/pretreatment.csv')

if __name__ == '__main__':
    main()