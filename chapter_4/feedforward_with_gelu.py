import torch.nn as nn
import torch
import json

# Compared to ReLU, GELU and SwiGLUE are more complex and smooth activation functions incorporating Gaussian and signmoid gated
# linear units. # Compared to ReLU, GELU and SwiGLUE are more complex and smooth activation functions incorporating Gaussian and signmoid gated
# linear units. # Compared to ReLU, GELU and SwiGLUE are more complex and smooth activation functions incorporating Gaussian and signmoid gated
# linear units. 
# It is defined as:
# GELU(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))

# with open('gpt_config.json', 'r') as f:
#     cfg = json.load(f)
# cfg = cfg['GPT_CONFIG_124M']
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # NOTE: x is a tensor.
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

class FeedForward(nn.Module):
    
    # The feedforward network is a simple two-layer perceptron with a GELU activation function.
    def __init__(self, cfg):
        super().__init__()
        
        # The intermediate dimension before doing GELU is larger (by 4x) than the embedding dimension.
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]) 
        )
        
    def forward(self, x):
        return self.layers(x)

# EXample to illustrate shortcut / residual connections.
class ExampleShortcut(nn.Module):
    def __init__(self, cfg, use_shortcut=True):
        super().__init__()
        
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(cfg["emb_dim"], cfg["emb_dim"]), GELU()),
            nn.Sequential(nn.Linear(cfg["emb_dim"], cfg["emb_dim"]), GELU()),
            nn.Sequential(nn.Linear(cfg["emb_dim"], cfg["emb_dim"]), GELU()),
            nn.Sequential(nn.Linear(cfg["emb_dim"],              1), GELU()) # one class output.
        ])
        
        
    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output # This is correct because the added result is used as the input to the next layer (and also for the skip connection).
            else:
                x = layer_output
        return x

# Testing the ExampleShortcut class.
# torch.manual_seed(123)
# model = ExampleShortcut(cfg, use_shortcut=False) # Setting use_shortcut to True will mitigate the vanishing gradient problem. Print the gradients to see this in effect.
# sample_input = torch.randn(1, cfg['emb_dim'])

# out = model(sample_input)
# target = torch.tensor([[0.]])
# loss = nn.MSELoss()(out, target)
# loss.backward()

# for name, param in model.named_parameters():
#     if 'weight' in name:
#         print(f"{name} has gradient mean of {param.grad.mean()} and std of {param.grad.std()}")
 
# Plotting the GELU activation function and comparing it to ReLU.
# import matplotlib.pyplot as plt
# gelu, relu = GELU(), nn.ReLU()

# # Get sample x values.
# x = torch.linspace(-3, 3, 100)
# y_gelu, y_relu = gelu(x), relu(x)
# plt.figure(figsize=(8, 3))

# for i, (y, label) in enumerate(zip([y_gelu, y_relu], ['GELU', 'ReLU']), 1):
#     plt.subplot(1, 2, i)
#     plt.plot(x, y)
#     plt.title(f"{label} activation function")
#     plt.xlabel("x")
    