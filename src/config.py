import transformers
import torch

TRAIN_PATH = r'../input/process_to_csv/train.csv'
VALIDATION_PATH = r'../input/process_to_csv/validation.csv'
TRAIN_RAW_PATH = r'../input/raw/train.txt'
VALIDATION_RAW_PATH = r'../input/raw/dev.txt'

BASE_MODEL_PATH = r'E:\Huggingface_Model\BERT\bert-base-uncased'
SAVE_MODEL_PATH = r'../model/REmodel.pth'
HIDDEN_SIZE = 768
MAX_LEN = 128
TRAIN_BATCH_SIZE = 12
VALIDATION_BATCH_SIZE = 4
LEARNING_RATE = 3e-5
NUM_CLASSES = 37
TOKENIZER = transformers.AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
