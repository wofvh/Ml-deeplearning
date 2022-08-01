
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
    #class의 매소드
    def show_info(self):
        print("소나타:",self.name, "/색상:",self.color,"/높이:",self.high,"/인용:",self.peple)
        #Setter 메소드 (set_name) < 이름을 바꾸는 매소드 
    def set_name(self. name):
        self.name = name



                
car1 = Car("아반떄","검은색","너무낮다","4인용")
print(car1.name, "을(를)구매했습니다!")
        
# car2 = Car("아반떄","검은색","너무낮다","4인용")
# car2.show_info()

# car4 = Car("구형아반떄","녹색","조금낮다","4인용") 
# car4.show_info()
    
        
        
        
        
        
        
        
        
        
        
        
        
#         def show_info(self):
#             print("이름", self,name, "/색상:", self,color):
                
# car1 = Car("소나타", "빨간색")
# car1.show_info()


# def add_num(*args):
#     return args

# print((add_num(1,2,3,4,5,6,7,'김플')[-1]))


# def input_me(**kwargs):
#     return kwargs
# print(input_me(name='panda', age=20))
# print(input_me(name='lee', age=20, food = 'pizza'))

