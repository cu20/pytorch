import torch

class Tudui(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        return x + 1

tudui = Tudui()
input = torch.tensor(1.0)
output = tudui(input)
print(output)