# a_string = 'like this' #str = string 문자  
# a_number = 3           #number 숫자
# a_float = 3.12         #float 소수점 있는 숫자  
# a_boolean  = False     #boolean True or False 진실 or 거짓 
#                        #int = integer 정수를 의미함 소수점 없는 숫자     
# a_none = None          #None 비어있다는걸 의미함 

# print(type(a_none))  

# days =  ("mon","tue","wed","thur","fri","sat" )
 
# print(days)
# days.append("sat")
# # days.reverse()          #reverse 리스안에 문자를 역방향으로 바꿔줄때사용 
# print(type(days))

# nico = {                    #dictionary 한 딕셔너리 안에 정보를 넣어사전 처럼 만듬 
#     "name" : "nico",
#     "age": 29 ,
#     "korean" : True,
#     "fav_food": ["kimchi",
#     "sashimi"]
# }

# print(nico)
# nico["handsome"] =True
# print(nico)


# something = ("tntt", True,15,None,False,"laslaslal")

# print(something)

# age = "18"
# print(age)
# print(type(age))
# n_age = (int(age))
# print(n_age)
# print(type(n_age))


# def p_plus(a ,b):
#     print(a + b)
    
# def r_plus(a, b):
#     return a + b 

# p_result  = p_plus(2,3)
# r_result  = r_plus(2,3)

# print(p_result,r_result)





# def plus(a,b):
#     return a + b

# result = plus(b = 30, a = 1 )
# print(result)
 

# def say_hello(name,age, are_from, fav_food):
#     return f"hello{name} you are {age} you are from{are_from}\
#     you like {fav_food}"    
    
# hello = say_hello(name="nico", age= "12",are_from="colombia",fav_food ="kimchi")
# print(hello)




# # def plus(a,b):
# #     if True:
# #         return None
# #     eles:
# #         return a + b
    
# #     plus(12,"10")
# # name = 0
# # while name <13 :
# #     print("파이썬 최고")
# #     name += 3

# # ########## if 문##############
# # score = 70
# # if score >= 60:
# #     message = "success"
# # else:
# #     message = "failure"
    
# # message = "success" if score >= 60 else "failure"
# # print(message)

# # a = 3
# # b = 4
# # if a >= b:
# #     print("nononononoo")
# #     print("nononononoo")
# #     print("nononononoo")
# # else:
# #     print("황도영")

# ####### while 문###############
# # treeHit = 0
# # while treeHit < 10:
# #     treeHit = treeHit +1
# #     print("나무를 %d 번 찍어습니다."% treeHit)
# #     if treeHit == 10:
# #         print("나무를 넘어갑니다.")


# coffee = 10
# money = 300
# while money:
#     print("돈을 받았으니 커피를 줍니다.")
#     coffee = coffee -1
#     print("남은 커피의 양은 %d개입니다." % coffee)
#     if not coffee:
#         print("커피가 다 떨어졌습니다. 판매를 중지합니다")
#         break



# a = 0
# while a <10:
#     a = a+1
#     if a % 2 ==0:
#         continue
#     print(a)

# while True:
#     print("안녕하세요")

# ################for문###############################

# test_list = ['one','two','three']
# for i in test_list:
#     print(i)
    

# a = [(1,2),(3,4),(5,6)]
# for (first, last) in a:
#     print(first + last)

# marks = [90,25,67,45,80]
# number = 0
# for mark in marks:
#     number = number +1
#     if mark <= 60: continue
#     print("%d번 학생은 합격입니다." % number)


# #############2중 for 문########################
# for i in range(2,10):
#     for j in range(1,10):
#         print(i*j, end=" ")
#     print(' ')




# a = [1,2,3,4,5,6,7,8]

# result = [num * 3 for num in a if num % 2 == 0 ]
# result = []
# for num in a:
#     if num%2 == 0:
#         result.append(num*3)




team = ["초원","지수","형권"]

number = 0
for mark in team:
    number = number +1
    if mark <= 60: continue
    print("초록 형권 지수 한팀" )
