import fivehrParallelParseFiles as sp
import pickleFiles as pf
import os.path as osPath

if osPath.isfile("allPickledFiles.p"):
	print("Found pickled data.")
	dataTuple = pf.load_text("allPickledFiles.p")
	print(dataTuple[1][0])
	print(len(dataTuple[0]))
	print(len(dataTuple[1]))
else:
	print("No pickled data, parsing text.")
	dataTuple = sp.parse_text()
	pf.store_text(dataTuple, "allPickledFiles.p")
	print(dataTuple[1][0])
