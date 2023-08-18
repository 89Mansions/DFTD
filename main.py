import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn import preprocessing
from model.models import *
from utils import *
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import torch.backends.cudnn as cudnn


class CSVFileNotFound(Exception):

    def __init__(self, filename, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = filename

    def __str__(self):
        return f'The path {self.filename} does not exist'


def concat_csv(location):
    df = pd.read_csv(f'./csv/{location}.csv')
    feature_num = df.shape[1]
    return df, feature_num


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2
batch_size = 256
input_length = 57
output_length = 4
loss_function = 'MSE'
learning_rate = 1e-3
weight_decay = 0.0001
tcn_channel_list = [32, 64, 128]
kernel_size = 3
dropout = 0.1
scalar = True
scalar_contain_labels = True
target_value = 'GHI'
location = ''
year = ''
epochs = 100
if output_length > 1:
    forecasting_model = 'multi_steps'
else:
    forecasting_model = 'one_steps'

df, feature_num = concat_csv(location=location)
data_length = len(df)
features_num = feature_num
if features_num > 1:
    features_ = df.values
else:
    features_ = df[target_value].values
labels_ = df[target_value].values
split_train_val, split_val_test = int(len(features_) * train_ratio), \
                                  int(len(features_) * train_ratio) + int(len(features_) * val_ratio)

if scalar:
    train_features_ = features_[:split_train_val]
    val_test_features_ = features_[split_train_val:]
    scalar = preprocessing.MinMaxScaler()
    if features_num == 1:
        train_features_ = np.expand_dims(train_features_, axis=1)
        val_test_features_ = np.expand_dims(val_test_features_, axis=1)
    train_features_ = scalar.fit_transform(train_features_)
    val_test_features_ = scalar.transform(val_test_features_)
    features_ = np.vstack([train_features_, val_test_features_])
    if scalar_contain_labels:
        labels_ = features_[:, -1]

if len(features_.shape) == 1:
    features_ = np.expand_dims(features_, 0).T
features, labels = get_rolling_window_multistep(output_length, 0, input_length,
                                                features_.T, np.expand_dims(labels_, 0))

labels = torch.squeeze(labels, dim=1)
features = features.to(torch.float32)
labels = labels.to(torch.float32)
split_train_val, split_val_test = int(len(features) * train_ratio), int(len(features) * train_ratio) + int(
    len(features) * val_ratio)
train_features, train_labels = features[:split_train_val], labels[:split_train_val]
val_features, val_labels = features[split_train_val:split_val_test], labels[split_train_val:split_val_test]
test_features, test_labels = features[split_val_test:], labels[split_val_test:]

train_Datasets = TensorDataset(train_features.to(device), train_labels.to(device))
train_Loader = DataLoader(batch_size=batch_size, dataset=train_Datasets)
val_Datasets = TensorDataset(val_features.to(device), val_labels.to(device))
val_Loader = DataLoader(batch_size=batch_size, dataset=val_Datasets)
test_Datasets = TensorDataset(test_features.to(device), test_labels.to(device))
test_Loader = DataLoader(batch_size=batch_size, dataset=test_Datasets)

TCNMain_model = Model(input_features_num=features_num, input_len=input_length, output_len=output_length,
                      tcn_OutputChannelList=tcn_channel_list, tcn_Dropout=dropout)

TCNMain_model.to(device)
if loss_function == 'MSE':
    loss_func = nn.MSELoss(reduction='mean')

optimizer = torch.optim.AdamW(TCNMain_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs // 3, eta_min=0.00001)

train_losses = []
val_losses = []
print("——————————————————————Training Starts——————————————————————")
for epoch in range(epochs):
    TCNMain_model.train()
    train_loss_sum = 0
    step = 1
    for step, (feature_, label_) in enumerate(train_Loader):
        optimizer.zero_grad()
        feature_ = feature_.permute(0, 2, 1)
        prediction = TCNMain_model(feature_)
        loss = loss_func(prediction, label_)
        train_losses.append(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm(TCNMain_model.parameters(), 0.15)
        optimizer.step()
        train_loss_sum += loss.item()
    print("epochs = " + str(epoch))
    print('train_loss = ' + str(train_loss_sum))

    TCNMain_model.eval()
    val_loss_sum = 0
    val_step = 1
    for val_step, (feature_, label_) in enumerate(val_Loader):
        feature_ = feature_.permute(0, 2, 1)
        with torch.no_grad():
            prediction = TCNMain_model(feature_)
            val_loss = loss_func(prediction, label_)
            val_losses.append(val_loss)
        val_loss_sum += val_loss.item()
    if epoch == 0:
        val_best = val_loss_sum
        print('val_loss = ' + str(val_loss_sum))
    else:
        print('val_loss = ' + str(val_loss_sum))
        if val_best > val_loss_sum:
            val_best = val_loss_sum
            torch.save(TCNMain_model.state_dict(), './weights/best.pth')
            print("val_best change")
print("best val loss = " + str(val_best))
print("——————————————————————Training Ends——————————————————————")

TCNMain_model.load_state_dict(torch.load('./weights/best.pth'))
test_loss_sum = 0
print("——————————————————————Testing Starts——————————————————————")
for step, (feature_, label_) in enumerate(test_Loader):
    feature_ = feature_.permute(0, 2, 1)
    with torch.no_grad():
        if step == 0:
            prediction = TCNMain_model(feature_)
            pre_array = prediction.cpu()
            label_array = label_.cpu()
            loss = loss_func(prediction, label_)
            test_loss_sum += loss.item()
        else:
            prediction = TCNMain_model(feature_)
            pre_array = np.vstack((pre_array, prediction.cpu()))
            label_array = np.vstack((label_array, label_.cpu()))
            loss = loss_func(prediction, label_)
            test_loss_sum += loss.item()
print("test loss = " + str(test_loss_sum))
print("——————————————————————Testing Ends——————————————————————")

print("——————————————————————Post-Processing——————————————————————")
if scalar_contain_labels and scalar:
    pre_inverse = []
    test_inverse = []
    if features_num == 1 and output_length == 1:
        for pre_slice in range(pre_array.shape[0]):
            pre_inverse_slice = scalar.inverse_transform(np.expand_dims(pre_array[pre_slice, :], axis=1))
            test_inverse_slice = scalar.inverse_transform(np.expand_dims(label_array[pre_slice, :], axis=1))
            pre_inverse.append(pre_inverse_slice)
            test_inverse.append(test_inverse_slice)
        pre_array = np.array(pre_inverse).squeeze(axis=-1)
        test_labels = np.array(test_inverse).squeeze(axis=-1)
    elif features_num > 1:
        if isinstance(pre_array, np.ndarray):
            pre_array = torch.from_numpy(pre_array)
        for pre_slice in range(pre_array.shape[0]):
            pre_inverse_slice = scalar.inverse_transform(torch.cat(
                (torch.zeros(pre_array[0].shape[0], features_num - 1), torch.unsqueeze(pre_array[pre_slice], dim=1)),
                1))[:, -1]
            test_inverse_slice = scalar.inverse_transform(torch.cat((torch.zeros(test_labels[0].shape[0],
                                                                                 features_num - 1),
                                                                     torch.unsqueeze(test_labels[pre_slice], dim=1)),
                                                                    1))[:, -1]
            pre_inverse.append(pre_inverse_slice)
            test_inverse.append(test_inverse_slice)
        pre_array = np.array(pre_inverse)
        test_labels = np.array(test_inverse)
    else:
        for pre_slice in range(pre_array.shape[0]):
            pre_inverse_slice = scalar.inverse_transform(np.expand_dims(pre_array[pre_slice, :], axis=1))
            test_inverse_slice = scalar.inverse_transform(np.expand_dims(label_array[pre_slice, :], axis=1))
            pre_inverse.append(pre_inverse_slice)
            test_inverse.append(test_inverse_slice)
        pre_array = np.array(pre_inverse).squeeze(axis=-1)
        test_labels = np.array(test_inverse).squeeze(axis=-1)
    plt.figure(figsize=(40, 20))
    if forecasting_model == 'multi_steps':
        plt.plot(pre_array[0], 'g', label='Predicted')
        plt.plot(test_labels[0], "r", label='Actual')
        plt.legend(loc='upper right', fontsize='large')
        plt.savefig('mymodel_{}_{}_{}.png'.format(location, input_length, output_length))

        # plt.show()
    else:
        plt.plot(pre_array, 'g', label='Predicted')
        plt.plot(test_labels, "r", label='Actual')
        plt.legend(loc='upper right', fontsize='large')
        # plt.savefig('{location}_{input_length}_{output_length}.png')
        plt.savefig('mymodel_{}_{}_{}.png'.format(location, input_length, output_length))

    MSE_l = mean_squared_error(test_labels, pre_array)
    MAE_l = mean_absolute_error(test_labels, pre_array)
    MAPE_l = mean_absolute_percentage_error(test_labels, pre_array)
    R2 = r2_score(test_labels, pre_array)
    print('MSE loss=%s' % MSE_l)
    print('MAE loss=%s' % MAE_l)
    print('MAPE loss=%s' % MAPE_l)
    print('R2=%s' % R2)

else:
    plt.figure(figsize=(40, 20))
    if forecasting_model == 'multi_steps':
        plt.plot(pre_array[0], 'g', label='Predicted')
        plt.plot(test_labels[0].cpu(), "r", label='Actual')
        plt.legend(loc='upper right', fontsize='large')
        # plt.show()
        plt.savefig('acc.png')
    else:
        plt.plot(pre_array, 'g', label='Predicted')
        plt.plot(test_labels.cpu(), "r", label='Actual')
        plt.legend(loc='upper right', fontsize='large')
        # plt.show()
        plt.savefig('acc.png')
    MSE_l = mean_squared_error(test_labels.cpu(), pre_array)
    MAE_l = mean_absolute_error(test_labels.cpu(), pre_array)
    MAPE_l = mean_absolute_percentage_error(test_labels.cpu(), pre_array)
    R2 = r2_score(test_labels.cpu(), pre_array)
    print('MSE loss=%s' % MSE_l)
    print('MAE loss=%s' % MAE_l)
    print('MAPE loss=%s' % MAPE_l)
    print('R2=%s' % R2)
