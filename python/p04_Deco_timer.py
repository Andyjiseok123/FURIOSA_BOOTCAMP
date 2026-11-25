import time

def timer(func):
    def wrapper():
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        print("걸린시간 : ",round(end - start,6),'초')
    return(wrapper)

@timer
def hello():
    print("안녕하세요")
    time.sleep(1)

hello()

@timer
def study():
    print("NPU 공부중이다")
    time.sleep(1.5)

study()