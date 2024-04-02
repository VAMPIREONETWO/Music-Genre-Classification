from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import numpy as np
from torcheval.metrics import MulticlassAUROC, MulticlassF1Score
import os
import librosa


def read_data(root='Data/genres_original', sr=16 * 1000):
    genres = os.listdir(root)
    x = []
    y = []
    length = []
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
    return x, y, min_length


def create_dataloader(x, y, batch_size=64):
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    data = TensorDataset(x, y)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return dataloader


def k_fold_cross_validation(x, y, k):
    fold_size = x.shape[0] // k
    xs_train = []
    ys_train = []
    xs_valid = []
    ys_valid = []
    for i in range(k - 1):
        xs_valid.append(x[fold_size * i:fold_size * (i + 1)])
        ys_valid.append(y[fold_size * i:fold_size * (i + 1)])
        xs_train.append(np.concatenate([x[:fold_size * i], x[fold_size * (i + 1):]], axis=0))
        ys_train.append(np.concatenate([y[:fold_size * i], y[fold_size * (i + 1):]], axis=0))
    xs_valid.append(x[fold_size * (k - 1):])
    ys_valid.append(y[fold_size * (k - 1):])
    xs_train.append(x[:fold_size * (k - 1)])
    ys_train.append(y[:fold_size * (k - 1)])
    return xs_train, ys_train, xs_valid, ys_valid


def train(model, loss_function, opt, dataloaders_train, dataloaders_valid, k, epoch=10):
    epochs_loss = []
    epochs_accuracy = []
    for i in range(epoch):
        print("-------epoch  {} -------".format(i + 1))
        epoch_loss = 0
        epoch_accuracy = 0

        for j in range(k):
            print(f'fold {j + 1}:')

            # train
            loss_train = 0
            accuracy_train = 0
            train_size = 0
            for batch_idx, (data, target) in enumerate(dataloaders_train[j]):
                data = data.cuda()
                target = target.cuda()
                model.train()
                output = model(data)
                loss = loss_function(output, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                data_size = len(data)
                loss_train += loss.item() * data_size
                accuracy_train += (output.argmax(1) == target).sum()
                train_size += data_size
            print("train set loss: {}".format(loss_train / train_size))
            print("train set accuracy: {}".format(accuracy_train / train_size))

            # valid
            loss_valid = 0
            accuracy_valid = 0
            valid_size = 0
            for batch_idx, (data, target) in enumerate(dataloaders_valid[j]):
                data = data.cuda()
                target = target.cuda()
                model.eval()
                with torch.no_grad():
                    output = model(data)
                    loss = loss_function(output, target)
                    data_size = len(data)
                    loss_valid += loss.item() * data_size
                    accuracy_valid += (output.argmax(1) == target).sum()
                    valid_size += data_size
            print("valid set loss: {}".format(loss_valid / valid_size))
            print("valid set accuracy: {}".format(accuracy_valid / valid_size))
            epoch_loss += loss_valid / valid_size
            epoch_accuracy += accuracy_valid / valid_size
        epoch_loss = round(epoch_loss / k, 3)
        epoch_accuracy = round(float(epoch_accuracy) / k, 3)
        print(f"epoch loss: {epoch_loss}")
        print(f"epoch accuracy: {epoch_accuracy}")
        epochs_loss.append(epoch_loss)
        epochs_accuracy.append(epoch_accuracy)
    return {"loss": epochs_loss, "accuracy": epochs_accuracy}


def test(model, loss_function, dataloader_test):
    loss_test = 0
    accuracy_test = 0
    AUC_test = 0
    f1_score_test = 0
    test_size = 0
    for batch_idx, (data, target) in enumerate(dataloader_test):
        data = data.cuda()
        target = target.cuda()
        model.eval()
        with torch.no_grad():
            output = model(data)
            loss = loss_function(output, target)
            loss_test += loss.item() * len(data)
            accuracy_test += (output.argmax(1) == target).sum()
            test_size += len(data)
            auc = MulticlassAUROC(num_classes=10)
            auc.update(output, target)
            AUC_test += auc.compute() * len(data)
            auc.reset()
            f1 = MulticlassF1Score(num_classes=10)
            f1.update(output, target)
            f1_score_test += f1.compute() * len(data)
            f1.reset()
    loss = round(loss_test / test_size, 3)
    accuracy = round(float(accuracy_test) / test_size,3)
    AUC = round(float(AUC_test) / test_size, 3)
    f1 = round(float(f1_score_test) / test_size,3)
    print(f"test set loss: {loss}")
    print(f"test set accuracy: {accuracy}")
    print(f"test set AUC: {AUC}")
    print(f"test set f1-score: {f1}")
    return loss, accuracy, AUC, f1
