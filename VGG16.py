from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models
from torchvision.models.vgg import model_urls
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
   
def init_weights(modules):# for weight initialization
    for m in modules: # m is each models in models
        if isinstance(m, nn.Conv2d): # why isinstance?? -> isinstance(object, classinfo) 왜 conv2d ?? -> conv2d는 convolution layer를 의미  #why nn.Conv2d? : 2차원 컨볼루션 레이어 isinstance : 첫번째 인자가 두번째 인자의 인스턴스인지 확인
            init.xavier_uniform_(m.weight.data)#xavier_uniform_ : Xavier initialization 는 가중치를 균일 분포로 초기화
            if m.bias is not None: # why m.bias is not None? : bias가 존재하면 # why m.bias? : bias는 뉴런의 출력값에 더해지는 상수 
                m.bias.data.zero_() # why zero_? : zero_ : 텐서의 모든 원소를 0으로 만듬
        elif isinstance(m, nn.BatchNorm2d): # why nn.BatchNorm2d? : 배치 정규화
            m.weight.data.fill_(1) # fill_(1): 텐서의 모든 원소를 1로 만듬
            m.bias.data.zero_()  #why zero_? : zero_ : 텐서의 모든 원소를 0으로 만듬
        elif isinstance(m, nn.Linear): # why nn.Linear? : 선형 레이어
            m.weight.data.normal_(0, 0.01) # why normal_? : normal_ : 텐서의 모든 원소를 정규 분포로 초기화 # how? : 평균 0, 표준편차 0.01
            m.bias.data.zero_() #why zero_? : zero_ : 텐서의 모든 원소를 0으로 만듬 # for? : bias는 뉴런의 출력값에 더해지는 상수

# model = models.vgg16_bn(pretrained=True) #vgg16_bn : vgg16의 batch normalization 버전

class vgg16_bn(torch.nn.Module): # what is torch.nn.Module? : 모델을 정의하는데 필요한 기본적인 클래스
    def __init__(self, pretrained=True, freeze=True): # what is __init__? : 클래스의 생성자
        super(vgg16_bn, self).__init__()
        model_urls['vgg16_bn'] = model_urls['vgg16_bn'].replace('https://', 'http://') # why replace? : https:// 를 http:// 로 바꿈
        vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained).features # what is models.vgg16_bn? : vgg16_bn 모델을 불러옴
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(12):         # conv2_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 19):         # conv3_3
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 29):         # conv4_3
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(29, 39):         # conv5_3
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        # fc6, fc7 without atrous conv
        self.slice5 = torch.nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
                nn.Conv2d(1024, 1024, kernel_size=1)
        )

        if not pretrained: # why not pretrained? : pretrained가 아닌 경우
            init_weights(self.slice1.modules()) #why modules? : 모듈을 반환
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())

        init_weights(self.slice5.modules())       # why init_weights? : weight initialization

        if freeze: # why freeze? : freeze가 True인 경우 # what is freeze? : 학습을 하지 않는다
            for param in self.slice1.parameters():      # only first conv
                param.requires_grad= False #what is requires_grad? : requires_grad가 True인 경우, gradient를 계산하고, False인 경우, gradient를 계산하지 않음

    def forward(self, X):
        h = self.slice1(X)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        vgg_outputs = namedtuple("VggOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2']) # what is namedtuple? : 튜플의 서브클래스로, 필드 이름을 갖는 튜플을 정의
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out
        