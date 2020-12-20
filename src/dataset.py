import config
import torch
from torch.utils import data
import pandas as pd


class REDataset(data.Dataset):
    def __init__(self, contexts, relations):
        self.contexts = contexts
        self.relations = relations
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, index):
        context = str(self.contexts[index])
        relation = self.relations[index]

        # 现在只有使用encode_plus才能反对字典，encode现在只返回一个list
        # 注意，最好使用 tokenzier(**args)（seq_classification）  这种方法不易出错， 在NER时候， 还是 encode 好些
        # 报的错误常常是 batch 维度不一致
        tokenized_sentence = self.tokenizer(
            context,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        input_ids = tokenized_sentence['input_ids'].squeeze(0).long()
        attention_mask = tokenized_sentence['attention_mask'].squeeze(0).long()
        token_type_ids = tokenized_sentence['token_type_ids'].squeeze(0).long()

        # temp_len = len(input_ids)
        # padding_len = self.max_len - temp_len

        # input_ids.extend([0] * padding_len)     # 只能用extend 不然 [101,23,43,[0,0,0...]]
        # attention_mask.extend([0] * padding_len)
        # token_type_ids.extend([0] * padding_len)

        # input_ids = torch.tensor(input_ids).long()
        # attention_mask = torch.tensor(attention_mask).long()
        # token_type_ids = torch.tensor(token_type_ids).long()
        relation = torch.tensor(relation).long()   # 这里是已经 label_encoder 过后的 long

        return{
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'relation': relation
        }


if __name__ == '__main__':
    df = pd.read_csv(r'../input/process_to_csv/example.csv')
    print(df)
    dataset = REDataset(df['context'], df['relation'])
    print(dataset[0])

    # tokenizer = config.TOKENIZER.encode_plus('i have a pen', add_special_tokens=True)
    # print(tokenizer)