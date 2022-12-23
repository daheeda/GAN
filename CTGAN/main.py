# CTGAN
from ctgan import CTGANSynthesizer
import pandas as pd
import warnings
import joblib
import os
from gan_feature.gan_make_dataset import make_data
from gan_feature.gan_preprocessing import *
from autogluon.tabular import TabularDataset, TabularPredictor

warnings.filterwarnings('ignore')

input_path = "./data/train_input/*.csv"
target_path = "./data/train_target/*.csv"
generated_path = "./generated_data/"


class CreateGE:
    def __init__(self):
        self.gan_model = CTGANSynthesizer()
        self.predict_model = joblib.load("save_model.pkl")
        self.load_path = "./3_45_model"
        self.raw_data = make_dataset(input_path, target_path)
        self.discrete_col = ['obs_time']

    @property
    def generated_data(self):
        print("Start generated_data")
        for i in range(28):
            print(f"=======Start Day{i}===========")
            raw_data = self.raw_data[self.raw_data["DAT"] == i]
            self.gan_model.fit(raw_data, self.discrete_col, epochs=100)
            new_data = self.gan_model.sample(240)
            print(new_data.shape)
            new_data.to_csv(os.path.join(generated_path, f'generate_day{i}.csv'), index=False)
            print(f"========End Day{i}=========\n\n")
        return None

    def growth_env(self):
        y_list = []
        for i in range(28):
            data = pd.read_csv(f"./generated_data/generate_day{i}.csv")  # 동적변경
            data = data.groupby(['obs_time']).mean().reset_index()
            preprocessed_data = make_data(data)
            y_prediction = self.predict_model.predict(preprocessed_data)
            print(y_prediction)
            break
            # y_list.append(y_prediction)
            # max(y_list) # y_preddiction중 max값 반환해서 저장할 예정


if __name__ == '__main__':
    cls = CreateGE()
    # cls.generated_data
    # cls.growth_env()
    load_path = "3_45_model"
    predictor = TabularPredictor.load(load_path, require_version_match=False)
    print(predictor)

