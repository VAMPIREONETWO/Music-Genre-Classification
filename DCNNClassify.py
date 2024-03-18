import torch
from sklearn.model_selection import train_test_split
from DCNN import DCNN
import os
import librosa
import numpy as np
from torchinfo import summary
from torch import nn
from torch.optim import Adam
from Utils import create_dataloader

# limit GPU usage
torch.cuda.set_per_process_memory_fraction(0.625)

# read data
root = 'Data/genres_original'
genres = os.listdir(root)
x = []
y = []
length = []
sr = 22050
for genre in genres:
    genre_root = os.path.join(root, genre)
    audios = os.listdir(genre_root)
    for audio in audios:
        audio_path = os.path.join(genre_root, audio)
        signal, _ = librosa.load(audio_path, sr=sr)
        x.append(signal)
        length.append(len(signal))
        y.append(genres.index(genre))
min_length = min(length)

# segment and normalise
for i in range(1000):
    x[i] = x[i][0:min_length]
    x[i] = librosa.util.normalize(x[i])
x = np.asarray(x)
y = np.asarray(y)
print(x.shape,y.shape)
seg_length = 59049
frame_num = int(x.shape[1]/seg_length)
preprocessed_x = x[:, :frame_num*seg_length].reshape(frame_num*x.shape[0],1,seg_length)
preprocessed_y = (y.reshape(y.shape[0],1)*np.ones((y.shape[0],frame_num))).reshape(y.shape[0]*frame_num)
print(preprocessed_x.shape,preprocessed_y.shape)

# data split
x_train, x_test, y_train, y_test = train_test_split(preprocessed_x, preprocessed_y, test_size=0.2,
                                                    stratify=preprocessed_y,shuffle=True)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2,
                                                    stratify=y_train,shuffle=True)
print(x_train.shape,y_train.shape)
print(x_valid.shape,y_valid.shape)
batch_size = 64
dataloader_train = create_dataloader(x_train, y_train, batch_size=batch_size)
dataloader_valid = create_dataloader(x_valid, y_valid, batch_size=batch_size)
dataloader_test = create_dataloader(x_test, y_test, batch_size=batch_size)

# model construction
model = DCNN(10)
model.cuda()
loss_function = nn.CrossEntropyLoss()
opt = Adam(model.parameters(), lr=0.01)
summary(model,[(64,1,seg_length)])

# train
for i in range(10):
    print("-------epoch  {} -------".format(i + 1))
    loss_train = 0
    step_train = 0
    for batch_idx, (data, target) in enumerate(dataloader_train):
        model.train()
        outputs = model(data)
        loss = loss_function(outputs, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_train += loss.item()
        step_train += 1
    print("Loss: {}".format(loss_train/step_train))

    loss_valid = 0
    step_valid = 0
    for batch_idx, (data, target) in enumerate(dataloader_valid):
        model.eval()
        with torch.no_grad():
            outputs = model(data)
            loss = loss_function(outputs, target)
            loss_valid = loss_valid + loss.item()
            step_valid += 1

    print("test set Loss: {}".format(loss_valid/step_valid))