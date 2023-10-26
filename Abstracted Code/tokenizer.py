import tokenize
from io import StringIO
import nltk
from nltk.tokenize import RegexpTokenizer
word_list = ['model', '.', 'add', '(', '()', ',','=(', '),',')', '="', '",', '=', "'", "',", "='", '))', '"))', '())', "'])", "'))", '(),', "=['"]
def special(token):
    if token in word_list:
        return False
    else:
        return True
    
#nltk.download('punkt')
def tokenize_code(code):

    # Create a tokenizer that matches words, decimal numbers, and special characters
    tokenizer = RegexpTokenizer(r'\w+|\d+\.\d+|[^\w\s]+')

#text = "This is an example sentence. It contains multiple words."
    tokens = tokenizer.tokenize(code)
    tokens = [token for token in tokens if special(token)]
#    print(tokens)
    print(" ".join(tokens))

with open('Model#1_1.py', 'r') as file:
    for line in file:
        tokenize_code(line.strip())  # strip() removes the newline character


#print(words)
#