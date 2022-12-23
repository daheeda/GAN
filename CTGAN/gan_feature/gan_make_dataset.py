from glob import glob
from gan_feature.T_preprocessing import *


def make_data(data):

    data = preprocessing(data)
    make_cumsum_columns(data)

    data = drop_day_cumsum(data)

    make_time_slot(data)

    data = weight_moist(data)

    data = test_ver(data)

    data_ex = expanding_timeslot_test_ver(data)

    data = expanding_data(data, data, data_ex)

    data = day_mean_value(data)

    data = ec_spray(data)

    data.drop(data.filter(regex = '백색'), axis = 1, inplace=True)
    data.drop(data.filter(regex = '청색'), axis = 1, inplace=True)

    data_kf = kalman_filter(data)

    data = make_move_mean_median_run(data_kf, 7, 14)

    data = LPF(data, 0.1, 1)

    return data
