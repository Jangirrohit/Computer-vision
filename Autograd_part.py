import torch
import numpy as np
import math as m

# part1

ten1= torch.tensor([[1.,2.],[3.,4.]], requires_grad=True)
print(' tensor with requires grad=True',ten1)
# print(ten1.grad)
result_tensor=ten1*2+5
# print(result_tensor)
# result_tensor.backward()
result_tensor1=result_tensor.sum()
# print(result_tensor1)
result_tensor1.backward()
print('gradient of the resulting tensor with respect to the original tensor.',ten1.grad)



# part2

x= torch.tensor([2.],requires_grad=True)
y=x**2+3*x+2
y.backward()
print(' derivative of f(x) at x = 2. is :', x.grad)


# part3

a= torch.rand((3,3), requires_grad=True)
b= torch.rand((3,3), requires_grad=True)
c=a*b
sum_result=c.sum()
sum_result.backward()
print(' gradient of a is :', a.grad)
print(' gradient of b is :', b.grad)


# part4
x=torch.rand((1,3),requires_grad=True)
W=torch.ones((3,1),requires_grad=True)
b= torch.tensor([0.],requires_grad=True)
z=torch.matmul(x,W)+b
def sigmoid(a):
    y= 1/(1+torch.exp(-a))
    return y
Y_hat= sigmoid(z)
print('pridicated value of output',Y_hat)
Y_actual= torch.tensor([1.0],requires_grad=True) # let's say 1 for apple and 0 for orange    and for this case the actual output is one
loss = - (Y_actual * torch.log(Y_hat) + (1 - Y_actual) * torch.log(1 - Y_hat))
loss.backward()
grad_w=W.grad
grad_b=b.grad
grad_x=x.grad
W= W-0.1*grad_w
b=b-0.1*grad_b
print(W,b)


#part5

x= torch.tensor([1.],requires_grad=True)
y= torch.tensor([2.],requires_grad=True)
g=(x**2)*y+y**3
g.backward()
print(' gradient of the function g(x,y) = x2y + y3 with respect to x and y at the point x = 1 and y = 2 is:',x.grad,y.grad)


#part6

# part7

ten= torch.tensor([[3.,2.,3.],[4.,5.,6.]],requires_grad=True)
norm_ten= torch.norm(ten)
# print(ten)
norm_ten.backward()
print(f"Gradients of the norm with respect to the tensor:\n{ten.grad}")


# part8

x= torch.tensor([1.], requires_grad=True)
y= torch.tensor([2.], requires_grad=True)
z= torch.tensor([3.], requires_grad=True)
h=x**2+y**2+z**2
h.backward()
print(f'gradient with respect to x, y, and z at the point x = 1, y = 2, and z = 3 is : {x.grad},{y.grad},{z.grad}')






