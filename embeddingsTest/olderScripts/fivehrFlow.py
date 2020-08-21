import fivehrParseFiles as sp
import pickleFiles as pf
import os.path as osPath

if osPath.isfile("fivehrPickledFiles.p"):
	print("Found pickled data.")
	dataTuple = pf.load_text("fivehrPickledFiles.p")
	print(dataTuple[1][0])
else:
	print("No pickled data, parsing text.")
	dataTuple = sp.parse_text()
	pf.store_text(dataTuple, "fivehrPickledFiles.p")
	print(dataTuple[1][0])
