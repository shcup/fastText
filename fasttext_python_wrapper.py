import ctypes
so = ctypes.cdll.LoadLibrary
lib = so("./fasttext.so")
print 'LoadModel'
lib.LoadModel("./result/india.bin")

