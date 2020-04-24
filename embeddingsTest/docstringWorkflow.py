import docstringSoupClean as dssc
import pickleFiles as pf
from multiprocessing import Pool
import docstringSanitize as dsSan

if __name__ == '__main__':
	documents = dssc.clean_docstrings()
	workerPool = Pool(48)
	cleanedDocuments = workerPool.map(dsSan.sanitize_text, documents)
	print(cleanedDocuments[0][1])
	print(cleanedDocuments[0][0])
	pf.store_text(cleanedDocuments, 'cleanedDocStringText.p')
