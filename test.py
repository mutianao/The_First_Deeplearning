import numpy as np
N,D_in,H,D_out = 64,1000,100,10
#N代表训练数据，D_in代表输入数据的维度，D_out代表输出数据的维度，H代表隐藏层的维度


x = np.random.randn(N,D_in)
y = np.random.randn(N,D_out)

w1 = np.random.randn(D_in,H)
w2 = np.random.randn(H,D_out)
#np.random.randn(x,y)返回一个或一组样本，具有标准正态分布。x,y表示每个维度即x行y列。返回值为指定维度的array

learning_rate = 1e-6
#learning_rate,学习率，学习率过大时导致无法收敛，学习率过小时导致收敛缓慢

for it in range(500):
#range(500)等价于range(0,500);这里代表训练500次

    #Forward pass正向传播
    #计算预测值y_pred
    h = x.dot(w1)
    #x.dot(y),代表两个向量（一维数组）的向量积，卷积里学过的
    h_relu = np.maximum(h,0)
    #maxmium(x,y)逐个比较两个向量（一维数组）的大小，返回最大值组成的向量，这里符合relu函数返回最大值。
    y_pred = h_relu.dot(w2)
    #y_pred代表预测值
    
    
    #Compute loss计算损失
    loss = np.square(y_pred - y).sum()
    print(it,loss)
    #输出每次训练后的与每次收敛的数值
    
    #Backward pass反向传播
    #compute the gradient计算w1,w2对loss的梯度
    #就是求导
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    #h_relu.T就是对矩阵h_relu进行转置
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h<0] = 0
    grad_w1 = x.T.dot(grad_h)
    
    #update weights of w1 and w2更新权重
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
