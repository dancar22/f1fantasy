from torch import no_grad, stack
from torch.utils.data import DataLoader
from torch.nn import Module


"""
Functions you should use.
Please avoid importing any other torch functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch.nn import Parameter, Linear
from torch import optim, tensor, tensordot, empty, ones
from torch.nn.functional import cross_entropy, relu, mse_loss
from torch import movedim



class DriverClassificationModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        input_size = 45
        output_size = 1
        "*** YOUR CODE HERE ***"
        self.batch_size = 2
        self.lr = 0.01
        self.hidden = 100
        
        self.m1 = Linear(input_size, self.hidden)
        self.m2 = Linear(self.hidden, self.hidden)
        self.m3 = Linear(self.hidden, output_size)




    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a tensor with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        """ YOUR CODE HERE """
        return self.m3(relu(self.m2(relu(self.m1(x)))))


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        """ YOUR CODE HERE """
        print("true value: ", y)
        print("guess: ", self.run(x))
        print("\n")
        return mse_loss(self.run(x), y)

        

    def train(self, dataset):
        """
        Trains the model.
        """
        """ YOUR CODE HERE """
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        
        for epoch in range(1000):
            for data in dataloader:
                optimizer.zero_grad()
                loss = self.get_loss(data['x'], data['label'])   
                loss.backward()
                optimizer.step()
                
            




