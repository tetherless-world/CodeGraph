if __name__ == '__main__':
        embeddingspath = '../../data/codeGraph/embeddings.csv'
        embeddingsToLabelPath='../../data/codeGraph/embeddingtolabel.csv'
        textToLabelPath='../../data/codeGraph/labeltotext.csv'
        with open(embeddingspath, 'r') as embeddings, open(textToLabelPath,'r') as embeddingsToLabels,open(embeddingsToLabelPath,'r') as textToLabels:
                i = 0
                for (line1,line2,line3) in zip(embeddings,embeddingsToLabels,textToLabels):
                        if i == 2:
                                break
                        newline = line1.rstrip()
                        parsedline = newline.split(',')
                        print(parsedline)
                        newline = line2.rstrip()
                        parsedline = newline.split(',')
                        print(parsedline)

                        newline = line3.rstrip()
                        parsedline = newline.split(',')
                        print(parsedline)
                        i += 1

