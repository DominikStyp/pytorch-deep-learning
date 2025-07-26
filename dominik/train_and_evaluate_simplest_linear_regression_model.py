import torch
# nn contains all of PyTorch's building blocks for neural networks
from torch import nn
import matplotlib.pyplot as plt

weight = 2
bias = 1

def create_data():
    # Create a simple dataset
    # Create data
    start = 0
    end = 1
    step = 0.02
    # tensor with X values
    X = torch.arange(start, end, step).unsqueeze(dim=1)
    # tensor with Y values, multiplies all X values by weight and adds bias
    Y = (weight * X) ** 2 + bias
    return X, Y

# split size tells you how much of the data should be used for training 
# split_size=0.8 means 80% for training and 20% for testing
def split_data_to_train_and_test(X, Y, split_size=0.8):
    # Split the data into training and testing sets
    split = int(len(X) * split_size)

    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    return X_train, Y_train, X_test, Y_test

def create_and_split_data():
    # Create and split the data
    X, Y = create_data()
    X_train, Y_train, X_test, Y_test = split_data_to_train_and_test(X, Y)

    return X_train, Y_train, X_test, Y_test

def plot_predictions(title, X_train, Y_train, X_test, Y_test, Y_predictions):
    # Plot the training data, testing data, and predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, Y_train, color='blue', label='Training Data')
    plt.scatter(X_test, Y_test, color='orange', label='Testing Data')
    plt.scatter(X_test, Y_predictions, color='red', label='Predictions')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.show()

# Create a Linear Regression model class
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__() 

        random_weight = torch.randn(1, dtype=torch.float)
        random_bias = torch.randn(1, dtype=torch.float)
        
        self.weights = nn.Parameter(random_weight, requires_grad=True)  # can we update this value with gradient descent?
        self.bias = nn.Parameter(random_bias, requires_grad=True)

    # Forward defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.weights * x) ** 2 + self.bias
    

###########################################################################

def train (model: nn.Module, X_train: torch.Tensor, Y_train: torch.Tensor, learning_rate=0.01, epochs=100):
    # Define the loss function and optimizer

    # Picking Mean Absolute Error (MAE) as the loss function
    # MAE is the average of the absolute differences between predictions and actual values
    loss_fn = nn.L1Loss() # MAE loss is same as L1Loss
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # Stochastic Gradient Descent

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set the model to training mode

        # Forward pass: compute predictions
        Y_pred = model(X_train)

        # Compute the loss, not necessary for training but useful for monitoring
        loss = loss_fn(Y_pred, Y_train)

        # Zero gradients, backward pass, and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")


def run_all():
    print("Running Linear Regression Model...")
    
    # Set manual seed since nn.Parameter are randomly initialized
    torch.manual_seed(22) # this will ensure that the random values are the same every time you run the code

    # Create and split the data
    X_train, Y_train, X_test, Y_test = create_and_split_data()

    # Create an instance of the model
    model_0 = LinearRegressionModel()

    # Check predictions of the model without training
    # inference_mode is used to disable gradient tracking, which is useful for inference
    with torch.inference_mode(): 
        Y_preds = model_0(X_test)

    # Plot untrained model predictions
    plot_predictions('Predictions before training', X_train, Y_train, X_test, Y_test, Y_preds)

    # CRUCIAL! Train the model
    train(model_0, X_train, Y_train, learning_rate=0.02, epochs=100)

    # Evaluate the model after training
    model_0.eval()

    # Get predictions after training
    with torch.inference_mode():
        Y_preds = model_0(X_test)
        
    # Plot trained model predictions
    plot_predictions('Predictions after training', X_train, Y_train, X_test, Y_test, Y_preds)



####### Run the function to execute the code
run_all()