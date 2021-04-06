import torch
a= torch.arange(30).reshape(5,6)
print(a)
print('b:',a.repeat(2,1))
