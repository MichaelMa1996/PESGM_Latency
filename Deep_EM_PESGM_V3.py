
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import os
from sklearn.mixture import GaussianMixture
import math
def reset_random_seeds():
    os.environ['PYTHONHASHSEED']=str(1)
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(0)
reset_random_seeds()
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)
class FuckMyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).float()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        a=1
        return x, y
    def __len__(self):
        return len(self.data)

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.linear1 = nn.Linear(in_features=x_train_datasize, out_features=60)
        self.linear2 = nn.Linear(in_features=60, out_features=60)
        self.linear3 = nn.Linear(in_features=60, out_features=60)
        self.linear4 = nn.Linear(in_features=60, out_features=2)
        # self.linearn = nn.Linear(in_features=20, out_features=14)

    def forward(self, input):  # Input is a 1D tensor
        y = self.linear1(input)
        y = F.relu(self.linear2(y.clone()))
        y = F.relu(self.linear3(y.clone()))
        y = F.relu(self.linear4(y.clone()))
        y = F.relu(y.clone())
        # y = F.relu(y)
        return y

# adding the gausian white noise is standard
noise_mean = 0
noise_std = 0.009

# using 200bus data event 1
filename = '4_bus_60Hz_10s_trimmed.csv'
df = pd.read_csv(filename, header=None)

# Parameters
num_columns_not_shift = 3  # Number of columns not to shift (e.g., the time column)
max_shift = 1  # Maximum number of rows to shift

# Initialize the maximum shift value
max_shift_applied = 0

# Shift the data
for col in df.columns[num_columns_not_shift:]:
    shift_by = np.random.randint(0, max_shift + 1)
    df[col] = df[col].shift(shift_by)
    max_shift_applied = max(max_shift_applied, shift_by)

# Trim the DataFrame to align the data
df1 = df.iloc[max_shift_applied:].reset_index(drop=True)
trimmed_original_df = df.iloc[:len(df)].reset_index(drop=True)

# train_data_2 =  pd.read_csv("C:\\Users\\zhihao's home\\Downloads\\testing.csv")
Dataset1 = df1.iloc[:,1:].values
Dataset0 =trimmed_original_df.iloc[:,1:].values
#add noise fron above so that it has uniform noise before any treament

# y_train = train_data.iloc[[10],[1]].values
PMU_for_training = 2 # number of PMU is selected while other converted to AMI data
PMU_selection_len = 10 # utilize 10 datapoint before to 10 datapoint after
Smart_mater_resolution = 33 # 15 --> one sample per 15min
# smart_meter_index = list(range(15, Dataset1.shape[0], Smart_mater_resolution))
smart_meter_index = list(range(15, 290, Smart_mater_resolution))
all_time_step_index  = list(range(15, 280, 1))

interpolate_remaining = [i for i in all_time_step_index if i not in smart_meter_index] # remove all the true dataset

if Dataset1.shape[0]-smart_meter_index[-1] < PMU_selection_len:
    smart_meter_index = smart_meter_index[0:-2] # make sure the last one have enough lenght

sc = MinMaxScaler()
sct = MinMaxScaler()
X_train = list()
y_train = list()
X_test = list()
Y_test = list()
Y_test_no_laten = list()
for i in smart_meter_index:
    X_train_sub = Dataset1[i-PMU_selection_len:i+PMU_selection_len,0:PMU_for_training]
    # X_train_sub = X_train_sub.reshape(-1,1)
    # X_train_sub = X_train_sub.astype(np.float)
    y_train_sub = Dataset1[i, PMU_for_training:]
    # y_train_sub = y_train_sub.reshape(-1,1)
    # y_train_sub = y_train_sub.astype(np.float)
    X_train.append(X_train_sub)
    y_train.append(y_train_sub)
for i in interpolate_remaining:
    X_test_sub = Dataset1[i-PMU_selection_len:i+PMU_selection_len,0:PMU_for_training]
    # X_test_sub = X_test_sub.reshape(-1,1)
    # X_test_sub = X_test_sub.astype(np.float)
    y_test_sub = Dataset1[i, PMU_for_training:]

    y_test_no_latent_sub = Dataset0[i, PMU_for_training:]
    # y_test_sub = y_test_sub.reshape(-1,1)
    # y_test_sub = y_test_sub.astype(np.float)
    X_test.append(X_test_sub)
    Y_test.append(y_test_sub)
    Y_test_no_laten.append(y_test_no_latent_sub)

# y_train =sct.fit_transform(y_train.reshape(-1,1))

x_train_datasize = 2*PMU_selection_len*PMU_for_training
y_train_datasize = Dataset1.shape[1]- PMU_for_training

#############################################################
###########################################################

X_test_total = torch.from_numpy(np.array(X_test).reshape(len(interpolate_remaining),x_train_datasize))
y_test_total = torch.from_numpy(np.array(Y_test).reshape(len(interpolate_remaining),y_train_datasize))
y_no_latency_total = torch.from_numpy(np.array(Y_test_no_laten).reshape(len(interpolate_remaining),y_train_datasize))

X_train_total = torch.from_numpy(np.array(X_train).reshape(len(smart_meter_index),x_train_datasize)).float()
y_train_total = torch.from_numpy(np.array(y_train).reshape(len(smart_meter_index),y_train_datasize)).float()
# total_dataset = FuckMyDataset(xtrain,ytrain)

# loader = torch.utils.data.DataLoader(dataset=total_dataset,batch_size=50,shuffle=True,num_workers=0)



model = network()

learning_rate = 1E-5
torch.autograd.set_detect_anomaly(True)
l = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr =learning_rate )
iteration_step = 1
y_test_predict_tensor = torch.empty(1,2) # initialize the predit

training_loss_list = list()
testing_loss_list = list()

for iter in range(iteration_step):
    # adam
    optimizer.zero_grad()
    if iter == 0:
        new_X_train = X_train_total.float().clone()
        new_Y_train = y_train_total.float().clone()
    if iter > 0:
        new_X_train = torch.cat((X_train_total.float().clone(),X_test_total.float().clone()))
        new_Y_train = torch.cat((y_train_total.float().clone(),y_test_predict_tensor.float().clone()))

        #load_saved_model
        state = torch.load("C:/Users/mazhi/PESGM_EM/"+str(iter)+".pth")
        model = network()
        model.load_state_dict(state['model'])

        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        optimizer.load_state_dict(state['optimizer'])
        for parameter in model.parameters():
            parameter.requires_grad = True
    new_dataset = torch.utils.data.TensorDataset(new_X_train,new_Y_train.clone())
    loader = torch.utils.data.DataLoader(dataset=new_dataset,batch_size=50,shuffle=True,num_workers=0)


    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = l(outputs, labels)


            loss.backward(retain_graph=True)

            optimizer.step()
            #clear out the gradients from the last step loss.backward()

            # y_predict = model(X_train_total.float()).detach().numpy()
            # y_true = np.array(y_train).reshape(len(smart_meter_index),y_train_datasize)

            y_predict_tensor = model(X_train_total.float())
            y_true_tensor = y_train_total
            training_loss_MSE = nn.MSELoss()
            # training_loss = training_loss_MSE(y_true_tensor,y_predict_tensor).item()
            # training_loss_list.append(training_loss)

            testing_loss_MSE = nn.MSELoss()
            y_test_predict_tensor = model(X_test_total.float()).clone()

            # code for Ds-f(xp)
            true_noise = y_true_tensor.numpy()-y_predict_tensor.detach().numpy()
            gm = GaussianMixture(n_components=1,random_state=random.randint(1, 1000)).fit(np.reshape(true_noise.T,(-1,1)))
            Z_size = y_test_predict_tensor.size()[0]*y_test_predict_tensor.size()[1]
            random_input_noise = gm.sample(Z_size)
            random_input_noise_reshape = np.reshape(random_input_noise[0],(y_test_predict_tensor.size()[0],y_test_predict_tensor.size()[1]))
            y_test_predict_tensor_float = y_test_predict_tensor.detach().numpy() + random_input_noise_reshape*0.01
            y_test_predict_tensor = torch.from_numpy(y_test_predict_tensor_float).float()
            # if (epoch+1) % 10 == 0:
            #     print('epoch {}'.format(epoch))

    testing_loss = testing_loss_MSE(y_test_predict_tensor,y_no_latency_total).item()
    print(testing_loss)
    testing_loss_list.append(testing_loss)
    testing_loss_bb = np.array(testing_loss_list)

    a=1
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    model_save_path = "C:/Users/mazhi/PESGM_EM/"+str(iter+1)+".pth"
    torch.save(state, model_save_path)
    model.zero_grad()
    a =1
a=1
# this is the remaining iteration begin
# iteration_number = 1
#
# for iter in range(iteration_number):
#8
#     # xtrain = np.array(X_train).reshape(len(smart_meter_index),x_train_datasize)
#     # ytrain = np.array(y_train).reshape(len(smart_meter_index),y_train_datasize)
#     # total_dataset = FuckMyDataset(xtrain,ytrain)
#     # trainloader = torch.utils.data.DataLoader(X_train, batch_size=4,
#     #                                           shuffle=True, num_workers=0)
#     loader = torch.utils.data.DataLoader(dataset=new_dataset,batch_size=50,shuffle=True,num_workers=0)
#     # network.reset_parameters()
#
#     y_test_predict_tensor = model2(X_test_total.float())        # print('epoch {}, loss {}'.format(epoch, loss.item()))
# a = 1
