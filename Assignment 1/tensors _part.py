import torch
import numpy as np

# part1

shape=(3,4)
ten1=torch.ones(shape)
print(' 2D tensor of size 3 ×4 filled with ones',ten1)

# part2.

shape2=(5,3)
ten2=torch.rand(shape)
# print(ten2)
print('random tensor of size 5 ×3 with its transpose.',ten2.T)

# part3

ten3= torch.arange(12)
print(' tensor with values from 0 to 11 and reshape it into a 3×4 matrix',ten3.reshape(3,4))

# part4

ten4=torch.rand(4,4)
print(' tensor of size 4×4 with random values ',ten4)
ten4[ten4<0.5]=0
print('tensor of size 4×4 with random values and replace all values less than 0.5 with zeros.',ten4)

# part5

ten1= torch.rand(2,3)
print('1st random tensor',ten1)
ten2= torch.rand(2,3)
print('2nd random tensor', ten2)
ten3=torch.cat([ten1,ten2], dim=0)
print('Concatenate two tensors of size 2 × 3 along the first dimension',ten3)

# part6

ten1= torch.rand(3,3)
print(ten1)
ten2= torch.rand(3,3)
print(ten2)
ten3= torch.mul(ten1, ten2)
print(' element-wise product of two tensors of the same size.',ten3)

# part7

ten1= torch.rand(2,3)
print(' a 2 × 3 tensor ',ten1)
ten2= torch.rand(3,4)
print(' a 3×4 tensor', ten2)
ten3= torch.matmul(ten1,ten2)
print(' matrix multiplication between a 2 × 3 tensor and a 3×4 tensor.',ten3)
print(ten3.shape)

# part8

T= torch.rand(4,4)
# print(T)
print(' mean of a tensor with random values.',torch.mean(T))
print('standard deviation of a tensor with random values',torch.std(T))

