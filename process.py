import joblib
from sklearn import preprocessing
import pandas as pd
import config


def save_encoder(path):
    df = pd.read_csv(path)
    relation_encoder = preprocessing.LabelEncoder()
    encode_column = df['relation']
    encoded_ralations = relation_encoder.fit_transform(encode_column)

    meta_data = {
        'relation_encoder': relation_encoder
    }
    joblib.dump(meta_data, './meta_data.bin')


def get_encoded_relations(df):
    relation_encoder = joblib.load('./meta_data.bin')
    encoded_relations = relation_encoder['relation_encoder'].fit_transform(df['relation'])
    return encoded_relations


if __name__ == '__main__':
    # path = r'../input/process_to_csv/example.csv'
    # contexts, encoded_relations, relation_encoder = make_data(path)
    # print(relation_encoder.inverse_transform(encoded_relations),
    #       type(relation_encoder.inverse_transform(encoded_relations)[0])
    #       )
    # print(encoded_relations)
    # print(contexts)

    # save_encoder(config.TRAIN_PATH)   # 值保存一次
    # df = pd.read_csv(config.VALIDATION_PATH)
    #
    # encoded_relaitons = get_encoded_relations(df)
    # print(encoded_relaitons)
    # print(len(set(encoded_relaitons)))

    relation_encoder = joblib.load('./meta_data.bin')
    print('ok')
    # print(len(relation_encoder['relation_encoder'].classes_))


    # 经过统计， 总共有 37 种关系
