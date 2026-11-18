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
