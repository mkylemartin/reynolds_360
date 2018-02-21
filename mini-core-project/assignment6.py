import operator
from glob import glob
import re

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import FreqDist

MINI_CORE = glob('Mini-CORE/*')
LINKRE = (r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|('
          r'?:%[0-9a-fA-F][0-9a-fA-F]))+')


def register(file):
    """
    Gets the register of a given filename from glob
    """
    regx = r'1\+(\w\w)'
    register = re.findall(regx, file)
    return register[0]


def get_registers(MINI_CORE):
    """
    Dynamically creates a list of all the registers
    """
    all_regs = []
    for file in MINI_CORE:
        all_regs.append(register(file))
    all_regs = sorted(list(set(all_regs)))
    return all_regs


def get_function_words():
    """
    Gets all the function words stored in the local file 'function_words.txt'
    """
    function_word_set = None
    with open('function_words.txt', 'r') as f:
        f = f.read()
        function_word_set = set(word_tokenize(f.lower()))
    return function_word_set


def cleanse(file):
    """
    Purification of the file in question to prepare it for tokenization
    e.g. strips HTML and anything in a tag
    """
    cleansed = ''
    # with open(file, 'r') as f:
    text = file.read()
    # removes any XML tags
    no_html = re.sub('<[^<]+?>', '', text)
    # removes any links
    cleansed = re.sub(LINKRE, '', no_html)
    return cleansed

regs = get_registers(MINI_CORE)

f_words = get_function_words()

all_data = {}
for reg in regs:
    fd = FreqDist()
    for file in MINI_CORE:
        fi_reg = register(file)
        if fi_reg == reg:
            with open(file, 'r') as f:
                f_cleansed = cleanse(f)
                tok_all = word_tokenize(f_cleansed.lower())
                filt_tok = [word for word in tok_all if word not in f_words]
                fd.update(filt_tok)
                all_data[reg] = fd
            print('.', end='', flush=True)

# print(all_data)

for key, value in all_data.items():
    with open('word-lists/' + key + '.txt', 'w+') as o:
        v_sort = sorted(value.items(), key=operator.itemgetter(1), reverse=True)
        for item in v_sort:
            print(item[0], item[1], sep='\t\t', file=o)

