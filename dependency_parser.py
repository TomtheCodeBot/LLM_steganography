import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag, word_tokenize, RegexpParser
text = "Reliance Retail acquires majority stake in designer brand Abraham & Thakore."
tokens = word_tokenize(text)
tags = pos_tag(tokens)
print(tags)
