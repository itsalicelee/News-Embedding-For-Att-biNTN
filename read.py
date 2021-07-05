import pickle
import sys

target = ['ttv.pkl', 'tdt.pkl', 'tagging.pkl']
try:
    arg  = str(sys.argv[1])
    thefile = './results/' + arg
    if arg in target:
        with open(thefile, 'rb') as f:
            lst = pickle.load(f)

        print("Length of " + arg + ": " + str(len(lst)))
        for i in range(5):
            print(lst[i])
    else:
        s = ""
        for i in range(len(target)):
            s += target[i] 
            if i != len(target)-1:
                s += ", "
        print("Choose from [" + s + "]")
except:
    print("Usage: python read.py [file]")
    

