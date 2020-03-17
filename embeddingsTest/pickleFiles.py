import pickle


def load_text(inputFilename):
	pickleInput = open(inputFilename, 'rb')
	try:
		return pickle.load(pickleInput)
	except pickle.UnpicklingError:
		return False
	pickleInput.close()

def store_text(dataTuple, outputFilename):
	pickleOutput = open(outputFilename, 'wb')
	try:
		pickle.dump(dataTuple, pickleOutput)
		return True
	except pickle.PicklingError:
		return False
	pickleOutput.close()
