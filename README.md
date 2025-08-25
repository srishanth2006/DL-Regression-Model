# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: SRISHANTH J

### Register Number: 212223240160

```import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(71)
X=torch.linspace(1,50,50).reshape(-1,1)
e=torch.randint(-8,9,(50,1),dtype=torch.float)
y=2* X + 1 + e

plt.scatter(X,y,color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated data for Linear Regression')
plt.show()

from logging import logProcesses
class Model(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.linear=nn.Linear(in_features,out_features)

    def forward(self,x):
        return self.linear(x)

torch.manual_seed(59)
model=Model(1,1)

initial_weight=model.linear.weight.item()
initial_bias=model.linear.bias.item()

print("\nName: SRISHANTH J")
print("Register No: (b012)")
print(f"Initial weight: {initial_weight:.8f}, Initial bias: {initial_bias:.8f}\n")


loss_function=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.001)

epochs=100
losses=[]
for epoch in range(1,epochs+1):
  optimizer.zero_grad()
  y_pred=model(X)
  loss=loss_function(y_pred,y)
  losses.append(loss.item())
  loss.backward()
  optimizer.step()


  print(f'epoch: {epoch:2}  loss: {loss.item():10.8f}  '
          f'weight: {model.linear.weight.item():10.8f}  '
          f'bias: {model.linear.bias.item():10.8f}')

plt.plot(range(epochs),losses,color='blue')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title("Loss Curve")
plt.show()


final_weight=model.linear.weight.item()
final_bias=model.linear.bias.item()
print("\nName:SRISHANTH J")
print("Register No: 212223240160")
print(f"Final weight: {final_weight:.8f}, Final bias: {final_bias:.8f}")

x1=torch.tensor([X.min().item(),X.max().item()])
y1=x1*final_weight+final_bias

plt.scatter(X,y,label="Original Data")
plt.plot(x1,y1,'r',label="Best-Fit Line")
plt.xlabel('x')
plt.ylabel('y')
plt.title("Trained Model: Best-Fit Line")
plt.legend()
plt.show()

x_new = torch.tensor([[120.0]])
y_new_pred=model(x_new).item()
print("\nName: SRISHANTH J")
print("Register No: (b012)")
print(f"Predicted value for x={x_new.item()}: {y_new_pred:.8f}")
```

### Dataset Information
<img width="802" height="639" alt="Screenshot 2025-08-25 184625" src="https://github.com/user-attachments/assets/0ba0dcea-e941-46b1-9acc-c80843bdf7e2" />



### OUTPUT
Training Loss Vs Iteration Plot
<img width="803" height="632" alt="Screenshot 2025-08-25 184833" src="https://github.com/user-attachments/assets/bb6ff9a5-6d76-44f3-af5a-7c5c1968a616" />


Best Fit line plot



<img width="808" height="629" alt="Screenshot 2025-08-25 184911" src="https://github.com/user-attachments/assets/b608dc32-bccd-490a-9fa9-455bd91cd26b" />



### New Sample Data Prediction

<img width="480" height="87" alt="Screenshot 2025-08-25 184949" src="https://github.com/user-attachments/assets/39a1314c-b367-437c-a2c1-f3d2690cf4e9" />





## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
