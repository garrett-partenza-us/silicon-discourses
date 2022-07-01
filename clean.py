import argparse
import io
import re

parser = argparse.ArgumentParser(description="flags for training tokenizer")
parser.add_argument("--path", required=True, type=str, help="path to .txt file")

def clean(path):
    with open(path, "r") as file:
        #read
        file = file.read()
        #ensure lowercase
        file = file.lower()
        #ensure remove new lines
        file = file.replace('\n', ' ')
        #strip
        file = file.strip()
        #ensure remove all punctuation but periods and question marks
        pattern = r"[{}]".format("-!\"#$%&'()*+,/:;<=>@[\]^_`{|}~") 
        file = re.sub(pattern, "", file)
        #single whitespace
        file = " ".join(file.split())
        #add back new line for tokenizer
        file = file.replace(". ", ". \n")
        #write
        with open("{}-cleaned.txt".format(path.split(".")[0]), "w") as new_file:
            new_file.write(file)
    
if __name__ == '__main__':
    
    args = parser.parse_args()
    clean(args.path)