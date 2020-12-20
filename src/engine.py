import config
from tqdm import tqdm
import torch


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        _, loss = model(input_ids=data['input_ids'], attention_mask=data['attention_mask'],
                        token_type_ids=data['token_type_ids'], target_relation=data['relation'])
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()

    return final_loss/len(data_loader)


def validation_fn(data_loader, model, device):
    model.eval()
    final_loss = 0
    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader)):
            for k, v in data.items():
                data[k] = v.to(device)
            _, loss = model(**data)
            final_loss += loss.item()

    return final_loss/len(data_loader)
