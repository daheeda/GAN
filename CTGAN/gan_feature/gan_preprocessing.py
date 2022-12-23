import tqdm
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from tqdm import tqdm
import pandas as pd
import numpy as np
import datetime
import glob
from scipy.signal import butter, lfilter


'''
생성 데이터 전처리 순서

* 0 일차 *
1. raw data 형태로 바꿔줌 : 누적값 및 dat, obs_time 수정해주기
2. 예측모델 형태로 바꿔줌  

* n 일차 *
1. raw data 형태로 바꿔줌 : 누적값 및 dat, obs_time 수정해주기
2. 0 ~ n-1 일차 데이터랑 합치기 : raw data 형태로
3. 예측모델 형태로 바꿔줌 

'''

def make_raw (df, day) :
    df_numpy = df.values
    return_arr = []
    for i in range(len(df_numpy)) :
        now = df_numpy[i]
        for j in range(24):
            value = np.insert(now[j*8:j*8+8],0, int(day)) # dat
            value = np.insert(value , 1 ,int(j)) # obs_time
            value = np.insert(value , 0 ,int(i)) # case
            return_arr.append(value.tolist())

    return_df = pd.DataFrame(np.array(return_arr),columns = ['Case','DAT', 'obs_time', '내부온도관측치', '내부습도관측치','co2관측치', 
                                        'ec관측치', '시간당분무량', '시간당백색광량',  '시간당적색광량',  '시간당청색광량' ])
    
    return_df['시간당총광량'] = return_df['시간당백색광량'] + return_df['시간당적색광량'] + return_df['시간당청색광량']

    cumsum_list = ['일간누적분무량', '일간누적백색광량', '일간누적적색광량', '일간누적청색광량', '일간누적총광량']
    per_time_list = ['시간당분무량', '시간당백색광량', '시간당적색광량', '시간당청색광량', '시간당총광량']

    for i in range(0, 5):
        col1 = cumsum_list[i]
        col2 = per_time_list[i]
        return_df[col1] = 0
        return_df[col1] = return_df.groupby((return_df.obs_time == 0).cumsum()).agg(col2).cumsum()

    return_df = return_df[['Case', 'DAT', 'obs_time', '내부온도관측치', '내부습도관측치', 'co2관측치', 'ec관측치', '시간당분무량',
             '일간누적분무량', '시간당백색광량', '일간누적백색광량', '시간당적색광량', '일간누적적색광량', '시간당청색광량',
             '일간누적청색광량', '시간당총광량', '일간누적총광량']]

    return return_df

# raw 데이터 형태로 바꾸기


def make_raw_data(df):

    df['시간당총광량'] = df['시간당백색광량'] + df['시간당적색광량'] + df['시간당청색광량']

    cumsum_list = ['일간누적분무량', '일간누적백색광량', '일간누적적색광량', '일간누적청색광량', '일간누적총광량']
    per_time_list = ['시간당분무량', '시간당백색광량', '시간당적색광량', '시간당청색광량', '시간당총광량']

    for i in range(0, 5):
        col1 = cumsum_list[i]
        col2 = per_time_list[i]
        df[col1] = 0
        df[col1] = df.groupby((df.obs_time == 0).cumsum()).agg(col2).cumsum()

    df = df[['DAT', 'obs_time', '내부온도관측치', '내부습도관측치', 'co2관측치', 'ec관측치', '시간당분무량',
             '일간누적분무량', '시간당백색광량', '일간누적백색광량', '시간당적색광량', '일간누적적색광량', '시간당청색광량',
             '일간누적청색광량', '시간당총광량', '일간누적총광량']]

    return df

# 0 ~ n-1 일차 데이터랑 합치기


def cumsum_dataset(df1, df2):
    newdf = pd.concat([df1, df2]).reset_index(drop=True)
    return newdf


##################################################################################################################

def preprocessing(data):
    data['obs_time'] = data.index % 24  # 시간통일
    # 전처리 1 : 데이콘에서 제공된 제한범위로 1차 전처리
    df = abs(data)
    df.loc[(df['내부온도관측치'] > 40), '내부온도관측치'] = 40
    df.loc[(df['내부습도관측치'] > 100), '내부습도관측치'] = 100
    df.loc[(df['co2관측치'] > 1200), 'co2관측치'] = 1200
    df.loc[(df['ec관측치'] > 8), 'ec관측치'] = 8
    df.loc[(df['시간당분무량'] > 3000), '시간당분무량'] = 3000
    df.loc[(df['시간당백색광량'] > 120000), '시간당백색광량'] = 120000
    df.loc[(df['시간당적색광량'] > 120000), '시간당적색광량'] = 120000
    df.loc[(df['시간당청색광량'] > 120000), '시간당청색광량'] = 120000
    df.loc[(df['시간당총광량'] > 120000), '시간당총광량'] = 120000

    # 전처리 2 : 이상치처리
    # 해당경우는 14시에서 15시로 넘어갈때 내부온도 및 내부습도가 0으로 관찰되었음
    # 같은 케이스가 반복적으로 보여서 15시 값을 14시 값으로 바꿔줌
    df.loc[(df['내부온도관측치'] < 6.921053), '내부온도관측치'] = 6.921053
    df.loc[(df['내부습도관측치'] < 9.639473), '내부습도관측치'] = 9.639473

    # 전처리3 : 시간당총광량
    # 백색/적색/청색 값의 합이 총광량이기때문에 전처리 1에 영향을 받았을 가능성이 있어서 칼럼을 다시 만들어줌
    df['시간당총광량'] = df['시간당청색광량']+df['시간당백색광량']+df['시간당적색광량']

    col_list = data.columns
    for i in range(0, len(col_list)):
        col = col_list[i]
        if '누적' in col:
            data[col] = data.groupby((data.obs_time == 0).cumsum()).agg(
                col_list[i-1]).cumsum()

    return data


def make_dataset(all_input_list_path, all_target_list_path):
    all_input_list = sorted(glob.glob(all_input_list_path))
    all_target_list = sorted(glob.glob(all_target_list_path))

    df_all = pd.DataFrame()
    length = len(all_input_list)
    for idx in range(length):
        X = pd.read_csv(all_input_list[idx])
        y = pd.read_csv(all_target_list[idx])
        y['DAT'] = y['DAT']-1
        df_concat = pd.merge(X, y, on='DAT', how='left')
        df_concat['Case'] = idx+1
        df_all = pd.concat([df_all, df_concat])
    return df_all


def n_cumsum(df):
    cumsum_list = ['05시내부온도관측치누적', '19시내부온도관측치누적', '23시내부온도관측치누적', '05시내부습도관측치누적',
                   '19시내부습도관측치누적', '23시내부습도관측치누적', '05시co2관측치누적', '19시co2관측치누적',
                   '23시co2관측치누적', '05시ec관측치누적', '19시ec관측치누적', '23시ec관측치누적', '05시분무량누적',
                   '19시분무량누적', '23시분무량누적', '05시백색광누적', '19시백색광누적', '23시백색광누적', '05시적색광누적',
                   '19시적색광누적', '23시적색광누적', '05시청색광누적', '19시청색광누적', '23시청색광누적', '05시총광량누적',
                   '19시총광량누적', '23시총광량누적']

    for col in cumsum_list:
        df[col] = 0

    return df


def make_cumsum_columns(df):
    time_list = ['05시', '19시', '23시']
    col_list = ['내부온도관측치누적', '내부습도관측치누적', 'co2관측치누적',
                'ec관측치누적', '분무량누적', '백색광누적', '적색광누적', '청색광누적', '총광량누적']
    for col in col_list:
        for time in time_list:
            df[time+col] = 0


def drop_day_cumsum(df):
    return df.drop(df.filter(regex='일간누적').columns, axis=1)


def time_split(df):
    """시간 분할
    Args:
        df (DataFrame): train, test data
    Returns:
        DataFrame: df['6time']
    """
    df['시간대'] = 0
    df.loc[(df['obs_time'] < 6), '시간대'] = 1
    df.loc[(df['obs_time'] >= 6) & (df['obs_time'] < 20), '시간대'] = 2
    df.loc[(df['obs_time'] >= 20) & (df['obs_time'] < 23), '시간대'] = 3

    return df


def make_time_slot(df):
    df['시간대'] = 0
    df['시간대'][(df['obs_time'] >= 0) & (df['obs_time'] <= 5)] = 1
    df['시간대'][(df['obs_time'] > 5) & (df['obs_time'] < 20)] = 2
    df['시간대'][(df['obs_time'] >= 20) & (df['obs_time'] <= 23)] = 3


def weight_moist(df):
    df = df.reset_index()

    df['측정될수분량2'] = 0 
    df['측정될수분량1'] = 0 
    df['측정될수분량3'] = 0 

    for i in range(22, len(df), 24) : 

        s2 = df.loc[i, '시간당분무량'] + df.loc[i+1, '시간당분무량'] * ((df.loc[i, '내부습도관측치'] + df.loc[i+1, '내부습도관측치']) / 2)
        s1 = df.loc[i+1,'시간당분무량'] * df.loc[i+1,'내부습도관측치'] 
        s3 = (df.loc[i,'시간당분무량'] + df.loc[i+1,'시간당분무량']) * ((df.loc[i,'내부습도관측치'] + df.loc[i+1,'내부습도관측치'])/2)
        
        df.loc[i,'측정될수분량2'] = s2
        df.loc[i,'측정될수분량1'] = s1
        df.loc[i,'측정될수분량3'] = s3

    return df.set_index(keys=['index'], inplace=False, drop=True)



def train_ver(train):
    train_x = train.drop(['predicted_weight_g'], axis=1)
    train_y = train['predicted_weight_g']
    return train_x, train_y


def test_ver(df):
    try:
        test_x = df.drop(['predicted_weight_g'], axis=1)
    except:
        pass
    return test_x


def expanding_timeslot(test_x):
    test_x = test_x.groupby(['DAT', '시간대']).sum().reset_index()
    test_x = test_x.sort_values(by=['DAT', '시간대'], axis=0).reset_index()
    test_x.drop(['index'], axis=1, inplace=True)

    col_list = ['내부온도관측치', '내부습도관측치', 'co2관측치', 'ec관측치', '시간당분무량', '시간당백색광량',
                '시간당적색광량', '시간당청색광량', '시간당총광량']

    for col in col_list:
        x = test_x[col].expanding().sum()
        test_x[col] = x

    return test_x


def expanding_data(train, train_x, train_x_2):

    # train_x = train_x.drop(['predicted_weight_g'], axis=1)

    col_list_1 = train_x.columns[12:-1]
    col_list_2 = train_x.columns[2:11]

    train_x = train_x.groupby(['DAT']).sum().reset_index()  # 784 로
    train_x = train_x.sort_values(by=['DAT'], axis=0).reset_index()
    train_x.drop(['index'], axis=1, inplace=True)

    k = 0
    for col_2 in col_list_2:
        for i in range(0, 3):
            try:
                train_x[col_list_1[k+i]] = train_x_2.loc[train_x_2.index %3 == i, col_2].values
            except:
                pass
        k += 3

    train_x.drop(['obs_time', '시간대', '내부온도관측치', '내부습도관측치',
                 'co2관측치', 'ec관측치'], axis=1, inplace=True)
    train_x = train_x.drop(train_x.filter(regex='시간당').columns, axis=1)

    return train_x


def ec_spray(df):
    df['ec_x_분무05'] = (df['05시ec관측치누적']+1) * (df['05시분무량누적']+1)
    df['ec_x_분무19'] = (df['19시ec관측치누적']+1) * (df['19시분무량누적']+1)
    df['ec_x_분무23'] = (df['23시ec관측치누적']+1) * (df['23시분무량누적']+1)
    df['ec_x_분무평균'] = (df['하루평균ec']+1) * (df['하루평균분무량']+1)

    df['적색_+_청색05'] = (df['05시적색광누적']) + (df['05시청색광누적'])
    df['적색_+_청색19'] = (df['19시적색광누적']) + (df['19시청색광누적'])
    df['적색_+_청색23'] = (df['23시적색광누적']) + (df['23시청색광누적'])
    df['적색_+_청색평균'] = (df['하루평균적색광']) + (df['하루평균청색광'])

    return df


def day_mean_value(df):
    df['하루평균온도'] = (df['05시내부온도관측치누적'] + df['19시내부온도관측치누적'] +df['23시내부온도관측치누적']) / 3
    df['하루평균습도'] = (df['05시내부습도관측치누적'] + df['19시내부습도관측치누적'] +df['23시내부습도관측치누적']) / 3
    df['하루평균co2'] = (df['05시co2관측치누적'] + df['19시co2관측치누적'] +df['23시co2관측치누적']) / 3
    df['하루평균ec'] = (df['05시ec관측치누적'] + df['19시ec관측치누적'] + df['23시ec관측치누적']) / 3
    df['하루평균분무량'] = (df['05시분무량누적'] + df['19시분무량누적'] + df['23시분무량누적']) / 3
    df['하루평균백색광'] = (df['05시백색광누적'] + df['19시백색광누적'] + df['23시백색광누적']) / 3
    df['하루평균적색광'] = (df['05시적색광누적'] + df['19시적색광누적'] + df['23시적색광누적']) / 3
    df['하루평균청색광'] = (df['05시청색광누적'] + df['19시청색광누적'] + df['23시청색광누적']) / 3
    df['하루평균총광량'] = (df['05시총광량누적'] + df['19시총광량누적'] + df['23시총광량누적']) / 3

    return df


def kalman_filter(data):
    data = data.drop(data.filter(regex='총광').columns, axis=1)
    data = data.drop(data.filter(regex = '백색').columns, axis =1)
    col_list = data.columns

    # feature name 생성
    for i in range(2, len(col_list)): 
        data['kf_X_'+str(i)] = 0
        
        
    for j in range(len(col_list)):
        if ((j == 0)):  # dat, case 뺌
            continue
        current = 0
        sum_c = []
        z = data.loc[:, data.columns[j]]
        a = []  # 필터링 된 피쳐(after)
        b = []  # 필터링 전 피쳐(before)
        my_filter = KalmanFilter(dim_x=2, dim_z=1)  # create kalman filter
        # initial state (location and velocity)
        my_filter.x = np.array([[2.], [0.]])
        # state transition matrix
        my_filter.F = np.array([[1., 1.], [0., 1.]])
        my_filter.H = np.array([[1., 0.]])    # Measurement function
        my_filter.P *= 1000.                 # covariance matrix
        my_filter.R = 5                      # state uncertainty
        my_filter.Q = Q_discrete_white_noise(dim=2, dt=.1, var=.1)  # process uncertainty
        for k in z.values:
            my_filter.predict()
            my_filter.update(k)
            x = my_filter.x
            a.extend(x[0])
            b.append(k)
        sum_c = sum_c+a

        data['kf_X_'+str(i+1)] = sum_c
    return data


def kalman_filter_mb(data): # 완
    data = data.drop(data.filter(regex='총광').columns, axis=1)
    data = data.drop(data.filter(regex = '백색').columns, axis =1)
    col_list = data.columns

    # feature name 생성
    for idx in range(2, len(col_list)): 
        data['kf_X_'+str(idx)] = 0
        
    for i in range(len(col_list)):
        if i == 0:
            continue
        
        col_value = data.loc[:, data.columns[i]]
        after_feature = []  # 필터링 된 피쳐(after)
        my_filter = KalmanFilter(dim_x=2, dim_z=1)  # create kalman filter
        my_filter.x = np.array([[2.], [0.]])    # initial state (location and velocity)
        my_filter.F = np.array([[1., 1.], [0., 1.]])    # state transition matrix
        my_filter.H = np.array([[1., 0.]])    # Measurement function
        my_filter.P *= 1000.                 # covariance matrix
        my_filter.R = 5                      # state uncertainty
        my_filter.Q = Q_discrete_white_noise(dim=2, dt=.1, var=.1)  # process uncertainty
        
        for k in col_value.values:
            my_filter.predict()
            my_filter.update(k)
            x = my_filter.x
            after_feature.extend(x[0])
        
        data['kf_X_'+str(i)] = after_feature

                
    return data

def make_move_mean_median(df, set_amount):
    
    return_df = pd.DataFrame()

    case_list = df['Case'].unique()
    col_list = df.columns
    for c in case_list :
        target = df[df['Case']==c]
        for col in col_list :
            mean_arr = []
            median_arr = []
            column_list  = target[col].to_list()
            for i in range(set_amount):
                try:
                    mean_arr.append(column_list [i])
                    median_arr.append(column_list [i])
                except:
                    pass
            for i in range(set_amount, len(column_list )):
                try:
                    mean_arr.append(float(np.mean(column_list[i-set_amount:i])))
                    median_arr.append(float(np.median(column_list[i-set_amount:i])))  
                except:
                    pass
            target[f'{col}_mean_{set_amount}'] = mean_arr
            target[f'{col}_median_{set_amount}'] = median_arr
        return_df = pd.concat([return_df, target],axis=0)
    return return_df


def moving_ave_mid(data, window):
    return_df = pd.DataFrame()
    col_list = data.columns
    for col in col_list :
        mean_arr = []
        median_arr = []
        column_list  = data[col].to_list()
        for i in range(window):
            try:
                mean_arr.append(column_list [i])
                median_arr.append(column_list [i])
            except:
                pass
        for i in range(window, len(column_list )):
            try:
                mean_arr.append(float(np.mean(column_list[i-window:i])))
                median_arr.append(float(np.median(column_list[i-window:i]))) 
            except:
                pass
        data[f'{col}_mean_{window}'] = mean_arr
        data[f'{col}_median_{window}'] = median_arr
    return_df = pd.concat([return_df, data], axis=0)
    print(return_df.shape)
    return return_df


def make_move_mean_median_run(df, set1, set2):
    
    dfc = df.drop(df.filter(regex = 'kf').columns, axis = 1)
    raw_cols = dfc.columns

    df1 = make_move_mean_median(dfc,set1)
    df2 = make_move_mean_median(dfc,set2)

    df1 = df1.drop(raw_cols, axis=1)
    print(df1.shape)
    df2 = df2.drop(raw_cols, axis=1)
    print(df2.shape)
    df = pd.concat([df, df1],axis=1)
    df = pd.concat([df, df2],axis=1)

    return df


def LPF(df, low, order=1) :
    new_df = pd.DataFrame()
    df_x_fill = df.iloc[:, 1:41]
    print(df_x_fill)
    print(df_x_fill.shape)
    case_list = df['Case'].unique()
    b, a = butter(
                    N = order,
                    Wn = low,
                    btype = 'low',
                    )
    for c in case_list :
        target = df_x_fill[df_x_fill['Case'] == c ]
        lpf_series = lfilter(b, a, target)
        lpf_dataframe = pd.DataFrame(lpf_series)
        new_df = pd.concat([new_df,lpf_dataframe], axis = 0)
    new_df = new_df.add_suffix('_LPF')
    new_df = new_df.reset_index(drop=True)
    df = pd.concat([df,new_df], axis = 1)
    
    return df
    