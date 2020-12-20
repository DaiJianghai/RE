import torch.nn as nn
import config
import transformers
import torch


# out[batch_size, num_classes], target[batch_size, 1]
def loss_fn(out, target_relation, num_classes):
    lfn = nn.CrossEntropyLoss()
    out_logits = out.view(-1, num_classes)
    target_relation = target_relation.view(config.TRAIN_BATCH_SIZE)
    return lfn(out, target_relation)


class REModel(nn.Module):
    def __init__(self, num_classes):
        super(REModel, self).__init__()
        self.num_classes = num_classes
        self.bert = transformers.BertModel.from_pretrained(config.BASE_MODEL_PATH, output_hidden_states=True,
                                                           return_dict=True)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(config.HIDDEN_SIZE * 2, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids, target_relation):
        output = self.bert(input_ids, attention_mask, token_type_ids)
        concat_2_hidden = torch.cat(tuple([output.hidden_states[i] for i in [-1, -2]]), dim=-1)  # [batch_size, seq_len, hidden*2]
        out_2_for_linear = concat_2_hidden[:, 0, :]     # [batch_size, 0, hidden*2]
        output_drop = self.dropout(out_2_for_linear)
        out = self.fc(output_drop)      # [batch_size, num_classes]

        loss = loss_fn(out, target_relation, self.num_classes)

        return out, loss