from multiprocessing import Process

print("test outside main")

def f(x):
    
    print(x);
    
if __name__ == '__main__':
    print("test inside main");
    p = Process(target=f, args=(2,));
    p.start();
    p2 = Process(target=f, args=(3,));
    p2.start();
    
