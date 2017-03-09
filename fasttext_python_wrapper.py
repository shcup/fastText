import sys
import base64
import ctypes
reload(sys)
sys.setdefaultencoding('utf-8')

so = ctypes.cdll.LoadLibrary
lib = so("./fasttext.so")
print 'LoadModel'
lib.LoadModel("./model.bin", 0)
print "Finish Model Loading"

preprocess_function=lib.PreProcess
preprocess_function.restype=ctypes.c_char_p

for line in open("galaxy_sample"):
  id, b64body = line.strip().split('|')[:2]
  text = base64.b64decode(b64body)
  print id + "\t" + text
  processed_text = preprocess_function(text)
  print id + "\t" + processed_text

