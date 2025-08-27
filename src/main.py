from primitives import *
from operations import *
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

class QuadraticDataset(Dataset):
    def __init__(self, n_samples=1000000, x_range=(-1, 1), y_range=(-1, 1)):
        super().__init__()
        self.n_samples = n_samples
        self.x_vals = torch.rand(n_samples) * (x_range[1] - x_range[0]) + x_range[0]
        self.y_vals = torch.rand(n_samples) * (y_range[1] - y_range[0]) + y_range[0]
        
        # compute labels
        # labels = 2*self.x_vals**2 - 3*self.x_vals*self.y_vals + 2*self.y_vals
        labels = self.x_vals * self.y_vals

        # duplicate to make 2D output
        self.labels = torch.stack([labels, labels], dim=1)
        
        # stack inputs
        self.data = torch.stack([self.x_vals, self.y_vals], dim=1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class Composer(nn.Module):
    def __init__(self, input_num_nns=2, layer_config=[]):
        super().__init__()
        
        self.layer_config = layer_config
        self.input_num_nns = input_num_nns

        cur_inp = input_num_nns
        self.weights = []
        for k in layer_config:
            self.weights.append(nn.Parameter(torch.randn((cur_inp, k)) * 0.01))
            cur_inp = k
        
        self.weights.append(nn.Parameter(torch.randn((cur_inp, 1))))
        self.weights = nn.ParameterList(self.weights)

    def forward(self, x):
        # x: list of NNs
        for w in self.weights:
            x = linear(x, w)
            x = act(x)

        return x[0]

def main():
    input_num_nns   = 2
    primitive_list  = [MultiplyOp(), AddOp()]

    composerNN = Composer(input_num_nns)
    composerNN.train()

    dataset = QuadraticDataset()
    train_dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, pin_memory=True)

    optimizer = optim.AdamW(composerNN.parameters(), lr=0.0001)
    
    for epoch in range(1000):
        for batch_idx, batch in enumerate(train_dataloader):
            x, y = batch

            # get the composed neural network
            nn_mod = composerNN(primitive_list)

            y_hat = nn_mod(x)
            
            loss = F.mse_loss(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('epoch: {} | batch: {} | loss: {}'.format(epoch, batch_idx, loss.item()))
                print(y[0, 0].numpy(), y_hat[0, 0].detach().numpy())
                print(composerNN.weights[0].data.detach().numpy())

if __name__ == '__main__':
    main()