from torch import no_grad, stack
from torch.utils.data import DataLoader
from torch.nn import Module

from torch.nn import Parameter, Linear
from torch import optim, tensor, tensordot, empty, ones
from torch.nn.functional import cross_entropy, relu, mse_loss
from torch import movedim



class DriverClassificationModel(Module):
    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        input_size = 56
        output_size = 1
        self.batch_size = 5
        self.lr = 0.01
        self.hidden = 300
        
        self.m1 = Linear(input_size, self.hidden)
        self.m2 = Linear(self.hidden, self.hidden)
        self.m3 = Linear(self.hidden, output_size)


    def run(self, x):
        return self.m3(relu(self.m2(relu(self.m1(x)))))


    def get_loss(self, x, y):
        ystar = self.run(x)
        return mse_loss(ystar, y)

        

    def train(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        for epoch in range(2000):
            for data in dataloader:
                optimizer.zero_grad()
                loss = self.get_loss(data['x'], data['label'])   
                loss.backward()
                optimizer.step()
                
            




