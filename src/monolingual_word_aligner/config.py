# flake8: noqa
ppdbDict = {}
ppdbSim = 0.9
theta1 = 0.9


#import nltk
#try:
#    from nltk.corpus import stopwords
#except:
#    from nltk import download
#    download('stopwords')

import os
from pathlib import Path
from nltk import SnowballStemmer
stemmer = SnowballStemmer('english')

def load_stop_words(language):
    base_path = Path(os.path.dirname(os.path.abspath(__file__)), "stopwords")
    full_path = os.path.join(base_path, language)
    try:
        with open(full_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            # Strip newline characters
            return [line.strip() for line in lines]
    except FileNotFoundError:
        print(f"Stop words file for language '{language}' not found.")
        return []
    except Exception as e:
        print(f"An error occurred while trying to load stopword file: {e}")
        return []

punctuations = ['(','-lrb-','.',',','-','?','!',';','_',':','{','}','[','/',']','...','"','\'',')', '-rrb-']
#stopwords = stopwords.words('english')

stopwords = load_stop_words("english")

