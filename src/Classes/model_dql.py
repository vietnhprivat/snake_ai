import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device='cpu', grid_mode=False):
        super().__init__()
        self.device = device
        self.grid_mode = grid_mode

        ## If state representation is not in the grid form, initialize linear layers (one hidden)
        if not self.grid_mode:
            self.linear1 = nn.Linear(input_size, hidden_size).to(self.device)
            self.linear2 = nn.Linear(hidden_size, output_size).to(self.device)
        
        ## Else, create two convolution layers before adding linear layers
        else:
            self.input_size = input_size
            self.conv1 = nn.Conv2d(1,6,kernel_size=3)
            self.conv2 = nn.Conv2d(6,8,kernel_size=3)
            self.linear1 = nn.Linear(512, 128).to(self.device)
            self.linear2 = nn.Linear(128, output_size).to(self.device)

    def forward(self, x):
        if not self.grid_mode:
            x = F.relu(self.linear1(x))
            x = self.linear2(x)
            return x
        else:
            return self.forward_grid(x)
        
    def forward_grid(self, x):
        ## Formatting of the input to match shape (Batch size, channels, height, width)
        if len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        else:
            if len(x.shape) == 3:
                if x.shape[1] == 1:
                    x = x.transpose(0,1)
                x = x.unsqueeze(1)

        ## Pass through the net
        x = F.relu(self.conv1(x))
        x = torch.nn.MaxPool2d(kernel_size=2)(x)
        x = F.relu(self.conv2(x))
        x = torch.flatten(x,1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth', index=0):
        model_folder_path = './DQL_models/model' 
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        complete_file_name = f"{index}_{file_name}"
        file_path = os.path.join(model_folder_path, complete_file_name)
        
        torch.save(self.state_dict(), file_path)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.device = model.device
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(self.device)
        action = torch.tensor(np.array(action), dtype=torch.long).to(self.device)
        reward = torch.tensor(np.array(reward), dtype=torch.float).to(self.device)
        # (n, x)

        if len(state.shape) == (1 if not self.model.grid_mode else 2):
            # (1, x)
            state = torch.unsqueeze(state, 0)#(0 if not self.model.grid_mode else 1))
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)
        target = pred.clone()

        ## If we have more than one state-action pair to train on, do so in a loop.
        ## If not, we will only process one pair
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()

        ## Calculate MSE loss on target and prediction
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()