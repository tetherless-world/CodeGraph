import ast
import sys
import os

good=0
bad=0

for f in os.listdir(sys.argv[1]):
  try:
    ast.parse(open(os.path.join(sys.argv[1], f)).read(), f)
    print(f)
    good += 1
  except SyntaxError:
    bad += 1

print (good)
print (bad)



