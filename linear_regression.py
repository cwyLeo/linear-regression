from fileinput import filename
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

def loss(data,w,b): #损失函数的计算
    loss_val = 0
    x = data[:,:-1]
    y = data[:,-1].reshape(data.shape[0],1)
    loss_val = np.sum((y - (np.dot(x,w.T) + b))**2)
    return loss_val/(2*float(len(data)))

def step_gradient(data,w,b,μ,w_grad_b): #梯度下降法实现数据拟合
    x = data[:,:-1]
    y = data[:,-1].reshape(data.shape[0],1)
    N = float(len(data))
    w_grad = -1/N * x.T.dot((y - (np.dot(x,w.T) + b)))
    b_grad = -1/N*np.sum(y - (np.dot(x,w.T) + b))
    egt = 0.9 * w_grad_b + 0.1 * (w_grad**2)
    return [w - μ*(w_grad/np.sqrt(egt)).T,b - μ*b_grad,egt]

def grad_run(data,w,b,μ,iterat): #设置迭代次数并存储每次的损失函数值
    loss_sum = []
    gt = 0
    for i in range(iterat):
        w,b,gt = step_gradient(np.array(data),w,b,μ,gt)
        loss_sum.append(loss(data,w,b))

    plt.figure(1)
    plt.subplot(211)
    plt.plot(range(iterat),loss_sum)
    plt.xlabel('iteration times')
    plt.ylabel('loss value')
    plt.title('connection between iteration and loss value')

    return [w,b]

#导入数据，设置学习率，迭代次数，并输出损失函数值变化图像和拟合情况 
filename = input("enter your data_filename:")
data = np.genfromtxt(filename,delimiter=",")[1:,]
μ = 0.05
w = np.zeros(data.shape[1] - 1).reshape(1,data.shape[1] - 1)
b = 0
iterat = 200
print("start gradient descent at w = {0},b = {1},error={2}".format(w,b,loss(data,w,b)))
print("running")
[w,b] = grad_run(data,w,b,μ,iterat)
print("after {0} iteration w = {1},b = {2},error = {3}".format(iterat,w,b,loss(data,w,b)))
if data.shape[1] == 2:
# 绘制拟合图像
    c = np.linspace(1.0, 100.0, 20) # np.linspace()函数为在指定的间隔内返回均匀间隔的数字，把区间等分
    d = np.zeros(20)
    d = w[0] * c + b

    x = data[:,0]
    y = data[:,1]
    plt.subplot(212)
    plt.scatter(x, y, c='r', s=4.0)
    plt.plot(c, d)
    plt.title('数据拟合图像')
    plt.tight_layout()
plt.show()