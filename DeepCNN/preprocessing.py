import re
import os
import glob
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
    return " ".join(tokens)

def check_fit(line):
    pattern = r"\*\.fit\(.*\).*\bkeras\b"

    if re.search(pattern, line):
        return True
    else:
        return False

def check_sequential(line):
    pattern = r"\bSequential\(\)"
    if re.search(pattern, line):
        return True
    else:
        return False
def check_summary(line):
    pattern = r"\bsummary\(\)"
    if re.search(pattern, line):
        return True
    else:
        return False
    
#for i in range(1,31):
#    flag = False
#    with open('FR/Model#6_{0}.py'.format(i), 'r') as input_file:
#        with open('ML/Model#6_{0}.py'.format(i), 'w') as output_file:
#            for line in input_file:
#                # Do some processing on the input line
#                if check_sequential(line):
#                    flag = True
#                
#                if flag and not check_summary(line): 
#                    output_file.write(line)
#                
#                if check_fit(line):
#                    flag = False
#        
#        output_file.close()
#    input_file.close()



DU_Counter = 0
FR_Counter = 0
KI_Counter = 0
KL_Counter = 0
PL_Counter = 0
SE_Counter = 0
totalModel = 17






#==================================================================<<DU>>=====================================================     
for i in range(1,totalModel):
    # Set the directory path
    directory_source_DU = "/Users/wardat/Downloads/CompleteCode/Model#{0}/mutation/DU".format(i)
    directory_dist_DU = "/Users/wardat/Downloads/LargeDataset/DU"
#    print(os.listdir(directory_source_DU))
    # loop through each file in the directory
    for filename in os.listdir(directory_source_DU):
        
        # check if the file is a Python file
        if filename.endswith('.py'):
            flag = False
            DU_Counter = DU_Counter + 1
            with open("{0}/{1}".format(directory_source_DU,filename), 'r') as input_file:
                with open("{0}/{1}.txt".format(directory_dist_DU,DU_Counter), mode='w', encoding='utf-8') as output_file:
                    for line in input_file:
                        # Do some processing on the input line
                        if check_sequential(line):
                            flag = True
                        
                        if flag and not check_summary(line): 
                            output_file.write(tokenize_code(line)+" ")
                        
                        if check_fit(line):
                            flag = False
                output_file.close()
            input_file.close()
#==================================================================<<FR>>=====================================================     
for i in range(1,totalModel):
    # Set the directory path
    directory_source_FR = "/Users/wardat/Downloads/CompleteCode/Model#{0}/mutation/FR".format(i)
    directory_dist_FR = "/Users/wardat/Downloads/LargeDataset/FR"
#    print(os.listdir(directory_source_DU))
    # loop through each file in the directory
    for filename in os.listdir(directory_source_FR):
        
        # check if the file is a Python file
        if filename.endswith('.py'):
            flag = False
            FR_Counter = FR_Counter + 1
            with open("{0}/{1}".format(directory_source_FR,filename), 'r') as input_file:
                with open("{0}/{1}.txt".format(directory_dist_FR,FR_Counter), mode='w', encoding='utf-8') as output_file:
                    for line in input_file:
                        # Do some processing on the input line
                        if check_sequential(line):
                            flag = True
                        
                        if flag and not check_summary(line): 
                            output_file.write(tokenize_code(line)+" ")
                        
                        if check_fit(line):
                            flag = False
                output_file.close()
            input_file.close()

#==================================================================<<KI>>=====================================================     
for i in range(1,totalModel):
    # Set the directory path
    directory_source_KI = "/Users/wardat/Downloads/CompleteCode/Model#{0}/mutation/KI".format(i)
    directory_dist_KI = "/Users/wardat/Downloads/LargeDataset/KI"
#    print(os.listdir(directory_source_DU))
    # loop through each file in the directory
    for filename in os.listdir(directory_source_KI):
        
        # check if the file is a Python file
        if filename.endswith('.py'):
            flag = False
            KI_Counter = KI_Counter + 1
            with open("{0}/{1}".format(directory_source_KI,filename), 'r') as input_file:
                with open("{0}/{1}.txt".format(directory_dist_KI,KI_Counter), mode='w', encoding='utf-8') as output_file:
                    for line in input_file:
                        # Do some processing on the input line
                        if check_sequential(line):
                            flag = True
                        
                        if flag and not check_summary(line): 
                            output_file.write(tokenize_code(line)+" ")
                        
                        if check_fit(line):
                            flag = False
                output_file.close()
            input_file.close()

#==================================================================<<KL>>=====================================================     
for i in range(1,totalModel):
    # Set the directory path
    directory_source_KL = "/Users/wardat/Downloads/CompleteCode/Model#{0}/mutation/KL".format(i)
    directory_dist_KL = "/Users/wardat/Downloads/LargeDataset/KL"
#    print(os.listdir(directory_source_DU))
    # loop through each file in the directory
    for filename in os.listdir(directory_source_KL):
        
        # check if the file is a Python file
        if filename.endswith('.py'):
            flag = False
            KL_Counter = KL_Counter + 1
            with open("{0}/{1}".format(directory_source_KL,filename), 'r') as input_file:
                with open("{0}/{1}.txt".format(directory_dist_KL,KL_Counter), mode='w', encoding='utf-8') as output_file:
                    for line in input_file:
                        # Do some processing on the input line
                        if check_sequential(line):
                            flag = True
                        
                        if flag and not check_summary(line): 
                            output_file.write(tokenize_code(line)+" ")
                        
                        if check_fit(line):
                            flag = False
                output_file.close()
            input_file.close()

#==================================================================<<PL>>=====================================================     
for i in range(1,totalModel):
    # Set the directory path
    directory_source_PL = "/Users/wardat/Downloads/CompleteCode/Model#{0}/mutation/PL".format(i)
    directory_dist_PL = "/Users/wardat/Downloads/LargeDataset/PL"
#    print(os.listdir(directory_source_DU))
    # loop through each file in the directory
    for filename in os.listdir(directory_source_PL):
        
        # check if the file is a Python file
        if filename.endswith('.py'):
            flag = False
            PL_Counter = PL_Counter + 1
            with open("{0}/{1}".format(directory_source_PL,filename), 'r') as input_file:
                with open("{0}/{1}.txt".format(directory_dist_PL,PL_Counter), mode='w', encoding='utf-8') as output_file:
                    for line in input_file:
                        # Do some processing on the input line
                        if check_sequential(line):
                            flag = True
                        
                        if flag and not check_summary(line): 
                            output_file.write(tokenize_code(line)+" ")
                        
                        if check_fit(line):
                            flag = False
                output_file.close()
            input_file.close()
#==================================================================<<SE>>=====================================================     
for i in range(1,totalModel):
    # Set the directory path
    directory_source_SE = "/Users/wardat/Downloads/CompleteCode/Model#{0}/mutation/SE".format(i)
    directory_dist_SE = "/Users/wardat/Downloads/LargeDataset/SE"
#    print(os.listdir(directory_source_DU))
    # loop through each file in the directory
    for filename in os.listdir(directory_source_SE):
        
        # check if the file is a Python file
        if filename.endswith('.py'):
            flag = False
            SE_Counter = SE_Counter + 1
            with open("{0}/{1}".format(directory_source_SE,filename), 'r') as input_file:
                with open("{0}/{1}.txt".format(directory_dist_SE,SE_Counter), mode='w', encoding='utf-8') as output_file:
                    for line in input_file:
                        # Do some processing on the input line
                        if check_sequential(line):
                            flag = True
                        
                        if flag and not check_summary(line): 
                            output_file.write(tokenize_code(line)+" ")
                        
                        if check_fit(line):
                            flag = False
                output_file.close()
            input_file.close()