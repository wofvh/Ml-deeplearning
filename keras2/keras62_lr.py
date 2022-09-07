
x = 100 #임의로 지정한 x값
y = 10 #목표로로 지정한 y값
w = 100 #임의로 지정한 w값
lr = 0.001 #학습률
epochs = 5000000 #반복횟수

for i in range(epochs):
    predict = x *w
    loss = (predict - y)**2
    
    print('loss:',round(loss,4),"\tpredict:",round(predict,4))

    up_predict = x * (w + lr)
    up_loss = (up_predict - y)**2

    down_predict = x * (w - lr)
    down_loss = (y - down_predict)**2

    if(up_loss > down_loss):
        w = w - lr
    else:
        w = w + lr