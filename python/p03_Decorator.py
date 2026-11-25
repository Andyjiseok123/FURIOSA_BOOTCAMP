# 데코레이터는 기존 함수를 수정하지 않고, 함수에 새로운 기능을 추가하는 문법
def hello():
    print("안녕하세요")

hello()

print("============================= 1 ==============================")
def deco(func):
    def wrapper():
        print("함수 실행전")
        func()
        print("함수 실행후")
    return wrapper

@deco
def hello2():
    print("안녕하세요")

hello2()
