import config
from model import REModel
import numpy as np
import pandas as pd
from process import get_encoded_relations
from dataset import REDataset
from torch.utils import data
import torch
from tqdm import tqdm
from sklearn import metrics
import joblib
import logging

logging.basicConfig(
    level=logging.DEBUG,
    filename='logging.txt'
)

if __name__ == '__main__':
    # df_validation = pd.read_csv(config.VALIDATION_PATH)
    # df_validation_contexts = df_validation.context.values
    #
    # df_validation_encoded_relations = get_encoded_relations(df_validation)
    #
    # # 制作 dataset
    # validation_dataset = REDataset(df_validation_contexts, df_validation_encoded_relations)
    # validation_data_loader = data.DataLoader(validation_dataset, batch_size=config.VALIDATION_BATCH_SIZE,
    #                                          shuffle=False)
    #
    # device = config.DEVICE
    # model = REModel(config.NUM_CLASSES)
    # checkpoint = torch.load(config.SAVE_MODEL_PATH)
    # model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    # model.to(device)
    # model.eval()
    # preds = []
    # for data in tqdm(validation_data_loader):
    #     with torch.no_grad():
    #         for k, v in data.items():
    #             data[k] = v.to(device)
    #         output, _ = model(input_ids=data['input_ids'],
    #                           attention_mask=data['attention_mask'],
    #                           token_type_ids=data['token_type_ids'],
    #                           target_relation=data['relation'])
    #         logits = output.detach().cpu().numpy()
    #         pred = np.argmax(logits, axis=-1)
    #         preds.extend(pred)
    # acc = metrics.accuracy_score(df_validation_encoded_relations, np.array(preds))
    # print(acc)  # now is overfitting

    # start test
    sentences = [
        ['The founder of <e1>Peking University</e1> is <e2>Zhang Jianai<e2>'],
        ['I come form <e1>JiangSu<e2> and graduated from <e2>NUIST</e2>'],
        ['now <e1>XiaoMing</e1> is living in the <e2>Shanghai</e2>'],
        ['<e1>President</e1>,<e2>Xi Jinping</e2>, of the Peoples Republic of China'],
        ['The president of <e1>student council</e1> of Shanghai University of Technology is <e2>Li Jiawei</e2>']
    ]

    dataset = REDataset(contexts=sentences)
    data_loader = data.DataLoader(dataset=dataset, batch_size=4, shuffle=False)
    device = config.DEVICE
    model = REModel(num_classes=config.NUM_CLASSES)
    checkpoint = torch.load(config.SAVE_MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for data in data_loader:
            data['input_ids'], data['attention_mask'], data['token_type_ids'] = data['input_ids'].to(device), data[
                'attention_mask'].to(device), data['token_type_ids'].to(device)
            output = model(input_ids=data['input_ids'],
                           attention_mask=data['attention_mask'],
                           token_type_ids=data['token_type_ids']
                           )
            logits = output.detach().cpu().numpy()
            pred = np.argmax(logits, axis=-1)
            preds.extend(pred)
    checkpoint_encoder = joblib.load('./meta_data.bin')
    relation_encoder = checkpoint_encoder['relation_encoder']
    pred_relations = relation_encoder.inverse_transform(preds)

    for i in range(len(sentences)):
        print(f'sentence:{sentences[i]} , relation:{pred_relations[i]}')

