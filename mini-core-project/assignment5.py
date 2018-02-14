from glob import glob
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from nltk.tokenize import RegexpTokenizer
# Regex tokenizer (leaves out punctuation)
re_tknzr = RegexpTokenizer(r'\w+').tokenize


Mini_CORE = glob('Mini-CORE/*')


# get register
def register(file):
    regx = r'1\+(\w\w)'
    register = re.findall(regx, file)
    return register[0]


def short_file(file):
    '''
    get short_file name for pretty tabs
    '''
    start = file[10:15]
    end = file[-4:]
    short_file = start + '...' + end
    return short_file


def sent_density(file):
    '''
    returns sentence density which is the number of sentences
    divided by the number of words.
    words, or tokens, are filtered.
    '''
    # with open(file, 'r').read() as f:
    lc_text = f.lower()
    filtered_tokens = re_tknzr(lc_text)
    word_count = len(filtered_tokens)
    sent_count = len(sent_tokenize(f))
    try:
        sent_density = round((sent_count / word_count), 3) * 1000
    except ZeroDivisionError:
        sent_density = 0
    return sent_density


def ttr_basic(file):
    '''
    basic type to token ratio, though I filter out the
    '''
    # with open(file, 'r').read() as f:
    lc_text = f.lower()
    filtered_tokens = re_tknzr(lc_text)
    word_count = len(filtered_tokens)
    unique_word_count = len(set(filtered_tokens))
    try:
        ttr_basic = round((unique_word_count / word_count), 3) * 1000
    except ZeroDivisionError:
        ttr_basic = 0
    return ttr_basic


# sentence length
def ave_s_len(file):
    '''
    divides total number of words by total number of sentences
    '''
    # with open(file, 'r').read() as f:
    num_tokens = len(re_tknzr(f.lower()))
    num_sents = len(sent_tokenize(f))
    ave_s_len = round(num_tokens / num_sents, 3) * 1000
    return ave_s_len


# opens all the files and writes a tsv to the cd
with open('my_output_file.tsv', 'w+') as o:
    print('filename',
          'sent_density',
          'ttr_basic',
          'ave_s_len',
          'register',
          sep='\t', file=o)
    for file in Mini_CORE:
        print('.', end='', sep='', flush=True)
        with open(file, 'r') as f:
            f = f.read()
            print(short_file(file),
                  sent_density(f),
                  ttr_basic(f),
                  ave_s_len(f),
                  register(file), sep='\t', file=o)
