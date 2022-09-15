import torch
torch.cuda.is_available()

X_train = torch.FloatTensor([0., 1., 2.])
X_train.is_cuda
