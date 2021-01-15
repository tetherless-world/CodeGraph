
import sys
from utils.util import evaluate_classification

if __name__ == '__main__':
    dataSetPath = sys.argv[1]
    embed_type = sys.argv[2]
    model_path = sys.argv[3]
    evaluate_classification(embed_type, model_path, dataSetPath, 'text1', 'text2', 'class', 'relevant')
