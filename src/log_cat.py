import numpy as np
import joblib

checkpoint = joblib.load('../model/log.txt')
print(checkpoint['train_loss'])
print(checkpoint['validation_loss'])