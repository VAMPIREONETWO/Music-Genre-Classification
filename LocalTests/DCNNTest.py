import torch
from sklearn.model_selection import train_test_split
from models.DCNN import DCNN
import os
import librosa
import numpy as np
from torchinfo import summary
from torch import nn
from torch.optim import Adam
from Utils import create_dataloader, k_fold_cross_validation
from torcheval.metrics import MulticlassAUROC, MulticlassF1Score
# limit GPU usage
torch.cuda.set_per_process_memory_fraction(0.625)

# read data
root = '../Data/genres_original'
genres = os.listdir(root)
x = []
y = []
length = []
sr = 16*1000
for genre in genres:
    genre_root = os.path.join(root, genre)
    audios = os.listdir(genre_root)
    for audio in audios:
        audio_path = os.path.join(genre_root, audio)
        signal, sr = librosa.load(audio_path, sr=sr)
        x.append(signal)
        length.append(len(signal))
        y.append(genres.index(genre))
min_length = min(length)
print("finsh reading data")

# segment and normalise
for i in range(1000):
    x[i] = x[i][0:min_length]
    x[i] = librosa.util.normalize(x[i])
x = np.asarray(x)
y = np.asarray(y)
# print(x.shape,y.shape)
seg_length = 59049
frame_num = int(x.shape[1]/seg_length)
preprocessed_x = x[:, :frame_num*seg_length].reshape(frame_num*x.shape[0],1,seg_length)
preprocessed_y = (y.reshape(y.shape[0],1)*np.ones((y.shape[0],frame_num))).reshape(y.shape[0]*frame_num)
# print(preprocessed_x.shape,preprocessed_y.shape)
print("finish segmentation and normalisation")

# data split
x_train, x_test, y_train, y_test = train_test_split(preprocessed_x, preprocessed_y, test_size=0.2,
                                                    stratify=preprocessed_y,shuffle=True)
# k-fold cross validation
k = 5
xs_train, ys_train, xs_valid, ys_valid = k_fold_cross_validation(x_train,y_train,k)
print("finish splitting data")
# create dataloaders
batch_size = 32
dataloaders_train = []
dataloaders_valid = []
for i in range(k):
    dataloaders_train.append(create_dataloader(xs_train[i], ys_train[i], batch_size=batch_size))
    dataloaders_valid.append(create_dataloader(xs_valid[i], ys_valid[i], batch_size=batch_size))
dataloader_test = create_dataloader(x_test, y_test, batch_size=batch_size)
print("finish creating dataloaders")

# model construction
model = DCNN(10)
model.cuda()
loss_function = nn.CrossEntropyLoss()
opt = Adam(model.parameters(), lr=0.01)
summary(model,[(64,1,seg_length)])
print("finish model construction")

# train
for i in range(10):
    print("-------epoch  {} -------".format(i + 1))
    for j in range(k):
        print(f'fold {j+1}:')
        loss_train = 0
        accuracy_train = 0
        train_size = 0
        for batch_idx, (data, target) in enumerate(dataloaders_train[j]):
            model.train()
            output = model(data)
            loss = loss_function(output, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_train += loss.item()*len(data)
            accuracy = (output.argmax(1) == target).sum()
            accuracy_train += accuracy
            train_size += len(data)
        print("train set loss: {}".format(loss_train/train_size))
        print("train set accuracy: {}".format(accuracy_train /train_size))

        loss_valid = 0
        accuracy_valid = 0
        valid_size = 0
        for batch_idx, (data, target) in enumerate(dataloaders_valid[j]):
            model.eval()
            with torch.no_grad():
                output = model(data)
                loss = loss_function(output, target)
                loss_valid += loss.item()*len(data)
                accuracy = (output.argmax(1) == target).sum()
                accuracy_valid += accuracy
                valid_size += len(data)
        print("valid set loss: {}".format(loss_valid/valid_size))
        print("valid set accuracy: {}".format(accuracy_valid/valid_size))
print("finish training")

# test
loss_test = 0
accuracy_test = 0
AUC_test = 0
f1_score_test = 0
test_size = 0
for batch_idx, (data, target) in enumerate(dataloader_test):
    model.eval()
    with torch.no_grad():
        output = model(data)
        loss = loss_function(output, target)
        loss_test += loss.item()*len(data)
        accuracy = (output.argmax(1) == target).sum()
        accuracy_test += accuracy
        test_size += len(data)
        auc = MulticlassAUROC(num_classes=10)
        auc.update(output, target)
        AUC_test += auc.compute()*len(data)
        f1 = MulticlassF1Score(num_classes=10)
        f1.update(output,target)
        f1_score_test += f1.compute()*len(data)
print("test set loss: {}".format(loss_test/test_size))
print("test set accuracy: {}".format(accuracy_test/test_size))
print("test set AUC: {}".format(AUC_test/test_size))
print("test set f1-score: {}".format(f1_score_test/test_size))