import config
from model import REModel
from dataset import REDataset
from engine import train_fn, validation_fn
from process import get_encoded_relations
import pandas as pd
from torch.utils import data
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
import torch
import joblib

if __name__ == '__main__':
    df_train = pd.read_csv(config.TRAIN_PATH)
    df_validation = pd.read_csv(config.VALIDATION_PATH)

    df_train_contexts = df_train.context.values
    df_validation_contexts = df_validation.context.values

    df_train_encoded_relations = get_encoded_relations(df_train)
    df_validation_encoded_relations = get_encoded_relations(df_validation)

    # 制作 dataset
    train_dataset = REDataset(df_train_contexts, df_train_encoded_relations)
    train_data_loader = data.DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)

    validation_dataset = REDataset(df_validation_contexts, df_validation_encoded_relations)
    validation_data_loader = data.DataLoader(validation_dataset, batch_size=config.VALIDATION_BATCH_SIZE,
                                             shuffle=False)
    device = config.DEVICE
    model = REModel(config.NUM_CLASSES)
    model.to(device)

    # model = REModel(config.NUM_CLASSES)
    # checkpoint = torch.load('../model/REmodel.pth')
    # model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    # model.to(device)

    # param_optimizer = list(model.named_parameters())
    # no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # optimizer_parameters = [
    #     {
    #         "params": [
    #             p for n, p in param_optimizer if not any(
    #                 nd in n for nd in no_decay
    #             )
    #         ],
    #         "weight_decay": 0.001,
    #     },
    #     {
    #         "params": [
    #             p for n, p in param_optimizer if any(
    #                 nd in n for nd in no_decay
    #             )
    #         ],
    #         "weight_decay": 0.0,
    #     },
    # ]
    #
    # num_train_steps = int(
    #     len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS
    # )
    # optimizer = AdamW(optimizer_parameters, lr=config.LEARNING_RATE)
    # # optimizer.load_state_dict(checkpoint['optimizer_state_dict)'])
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=num_train_steps
    # )

    optimizer  = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print('-' * 10, 'START_TRAINING', '-' * 10)
    best_score = np.inf
    train_loss = []
    validation_loss = []

    for epoch in range(config.EPOCHS):
        tr_loss = train_fn(data_loader=train_data_loader, model=model, optimizer=optimizer,
                           device=device)
        vl_loss = validation_fn(data_loader=validation_data_loader, model=model, device=device)

        print(f'{epoch+1}/{config.EPOCHS}', '-' * 5, f'train_loss={tr_loss}',
              '\t', f'validation_loss={vl_loss}')

        train_loss.append(tr_loss)
        validation_loss.append(vl_loss)

        if vl_loss < best_score:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'EPOCH': epoch
            }, config.SAVE_MODEL_PATH)
            best_score = vl_loss
    log_dict = {
        'train_loss': train_loss,
        'validation_loss': validation_loss
    }

    joblib.dump(log_dict, '../model/log.txt')