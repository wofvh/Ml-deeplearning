
#class 맴버 : class 내부에 포함되는 변수
#class 함수 : class 내부의 클래스 내부의 포함되는 함수 (매소드 라고부름)

from unicodedata import name
class Car:           #class란 어떤 사물을 객채로 컴퓨터가 받아 드릴수있게 지정해줌 
    #클래스의 생성자
    def __init__(self, name , color, high,peple):    #  __init__ <<(생성자 ,self 라는 것을 기본적으로 가지고있음)
        self.name = name   #classs 맴머  #(생성자)__init__문 클래스로 정의된 객체를 프로그램 상에서 이용할수있게만든변수
        self.color = color  #class 맴버   
        self.high = high
        self.peple = peple
    #class 소멸자  del << 변수 마지막에 사용 사용시 그전 변수값이 다 사라짐 
    #def __del__(self):
        print("인스턴스를 소멸시킵니다.")
    #class의 매소드
    def show_info(self):
        print("소나타:",self.name, "/색상:",self.color,"/높이:",self.high,"/인용:",self.peple)
        #Setter 메소드 (set_name) < 이름을 바꾸는 매소드 
    def set_name(self, name):
        self.name = name
         
car1 = Car("소나타","검은색","너무낮다","4인용")
car1.set_name("아반떄")
car1.show_info()
del car1        

class 상속개념 다른클래스의 멤버 변수와 메소드를 물려 받아 사용하는 기법
상속 class 에는 부모와 자식관계가 존재함 
자식 클래스 : 부모 클래스를 상속 받은 클래스
class Unit:
    def __init__(self,name,power):
        self.name = name
        self.power = power
    def attack(self):
        print(self.name,"이(가) 공격을 수행합니다,[전투력:",self.power,"]")
        
class Monster(Unit):
    def __init__(self, name, power, type):
        self.name = name
        self.power = power
        self.type = type
    def show_info(self):
        print("몬스터 이름:", self.name, "/ 몬스터 종류:", self.type)
        
Monster = Monster("슬라임",10 ,"초급")
Monster.show_info()
Monster.attack()


        
car2 = Car("아반떄","검은색","너무낮다","4인용")
car2.show_info()

car4 = Car("구형아반떄","녹색","조금낮다","4인용") 
car4.show_info()
        
        
        def show_info(self):
            print("이름", self,name, "/색상:", self,color):
                
car1 = Car("소나타", "빨간색")
car1.show_info()

list [1,2,3,4,5,6,7,8,9,10,11,
      312,23,23,4,34,]

for i in list:
    

def add_num(*args):
    return args

print((add_num(1,2,3,4,5,6,7,'김플')[-1]))


def input_me(**kwargs):
    return kwargs
print(input_me(name='panda', age=20))
print(input_me(name='lee', age=20, food = 'pizza'))

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import time

class Exam_Model_cls(): # Class 정의하기
    def __init__(self, output_nums, output_actfunc): # 생성자 => class 가 시작할떄 인자로 받으며 class 안에서의 변수로 지정됨 (self.output 같이)
        self.output = output_nums # 기억해두기 1
        self.output_func = output_actfunc # 기억해두기 2
        
    def make_model(self): # class 만들때 생성자에서 정의한 변수들로 모델 구성
        model = Sequential()  
        model.add(Dense(9, input_dim=13))
        model.add(Dense(19, activation='sigmoid'))
        model.add(Dense(11, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(self.output, activation = self.output_func)) # <<= 위에서 생성자로 정의한 self.output, self.output_func 두개 써서 모델 만들어줌
        return model # 그리고 꺼내주기
    
    
if __name__ == '__main__': # 파이썬을 바로 시작했을때 ( python exam.py <= 이렇게 실행하면 __name__ 이 __main__임 print(__name__) 으로 실험해보는게 좋음)
    ## import 하면 __name__은 __main__ 이 아님
    exam_class = Exam_Model_cls(output_nums= 1, output_actfunc= 'softmax') 
    # ㄴ위에 만들어준 클래스 안에 생성자로 받는 파라미터 두개 넣어줌 output_nums랑 actfunc가 여기서 받아서 self.xx로 변환되는건 위에서 확인
    # 파라미터 받아서 만들어진 클래스가 변수로 지정됨
    model = exam_class.make_model() # 위에 생성자로 self.xx 로 변수가 지정됐으니까 지정된 클래스 안에 변수 (make_model)로 모델 꺼내서 지정해줌
    model.summary()
