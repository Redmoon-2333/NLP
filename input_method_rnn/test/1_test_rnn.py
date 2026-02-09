import torch.nn as nn
import torch

rnn=nn.RNN(input_size=3,hidden_size=4,num_layers=2,batch_first=True,bidirectional=True)
# input.shape:[batch_size,seq_len,input_size]
input=torch.randn(2,4,3)
rnn(input)

# output.shape:[batch_size,seq_len,hidden_size*num_directions]
# hn.shape:[num_layers*num_directions,batch_size,hidden_size]

output,hn=rnn(input)
print(output.shape)
print(hn.shape)
