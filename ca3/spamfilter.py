"""SpamFilter [a.b] 17.12.2018

Spamfilter auf Basis von Whitelisting, Blacklisting und Naive Bayes Filter

Parameter aus spamfilter.params // params.py

Ablaufsteuerung (Windows): run_spamfilter.cmd
"""
# ----------------------------------------------
# Module laden
from os import listdir
from os.path import isfile, join, abspath
from itertools import chain

# Parameter laden
from ca3.params import *

KEY_OCCURRENCES = 'occurrences'
KEY_MAILS = 'mails'
KEY_PROBABILITY = 'probability'

# logging
...

# ----------------------------------------------
# Funktionen allgemein
...


# ----------------------------------------------

def load_lines(path):
    with open(path, 'r') as file:
        return [line for line in file.read().split('\n') if line != '']


# Blacklist laden
black_list = load_lines(filename_blacklist)

# Whitelist laden
white_list = load_lines(filename_whitelist)


# Mails laden
def load_mails(dir):
    mails = {}
    for mail_file_name in [file for file in listdir(dir) if isfile(join(dir, file))]:
        with open(dir + mail_file_name, mode='r') as mail_file:
            string = mail_file.read()
            for char_to_replace in char_replaces:
                string = string.replace(char_to_replace, char_replaces[char_to_replace])
            mails[mail_file_name] = [word for word in string.split(' ')
                                     if not any(ignore in word for ignore in words_ignore) and word != '']
    return mails


# Spam-Mails laden
spam_mails = load_mails(dir_spam)

# NoSpam-Mails laden
no_spam_mails = load_mails(dir_nospam)


# ----------------------------------------------
# Bewertungstabellen fuer Naive Bayes erstellen und in eigene Datei protokollieren
def create_evaluation_table(mails):
    table = {}
    for mail_key in mails:
        for word in mails[mail_key]:
            if word in table:
                table[word][KEY_OCCURRENCES] += 1
            else:
                table[word] = {KEY_OCCURRENCES: 1, KEY_MAILS: 0}
        for word in set(mails[mail_key]):
            if word in table:
                table[word][KEY_MAILS] += 1
            else:
                table[word][KEY_MAILS] = 1
    return {word: {KEY_MAILS: table[word][KEY_MAILS],
                   KEY_OCCURRENCES: table[word][KEY_OCCURRENCES],
                   KEY_PROBABILITY: table[word][KEY_MAILS] / len(mails)} for word in table}


def save_evaluation_table(evaluation_table, file_name):
    table_file = open(abspath(dir_output + filename_nbwordtable + dir_separator + file_name), 'w')
    for word in evaluation_table:
        table_file.write('{}: {}\n'.format(word, evaluation_table[word]))
    table_file.close()


spam_table = create_evaluation_table(spam_mails)
save_evaluation_table(spam_table, 'spam.txt')
no_spam_table = create_evaluation_table(no_spam_mails)
save_evaluation_table(no_spam_table, 'no_spam.txt')

nb_table = {word: ((spam_table[word][KEY_PROBABILITY] if word in spam_table else 0) / (
        (spam_table[word][KEY_PROBABILITY] if word in spam_table else 0)
        + (no_spam_table[word][KEY_PROBABILITY] if word in no_spam_table else 0))) for word in
            set(list(chain.from_iterable(
                [list(chain.from_iterable(list(spam_mails.values()))),
                 list(chain.from_iterable(list(no_spam_mails.values())))])))}
save_evaluation_table(nb_table, 'total_probabilities.txt')

# ----------------------------------------------
# MailInput laden, bewerten und nach MailOutput schreiben
# Bewertungsklassifikation: WhiteList, NoSpam, undetermined, Spam, BlackList
input_mails = load_mails(dir_input)

for filename, words in input_mails.items():

    # classification
    x_spam_probability = sum([nb_table[word] if word in nb_table else 0 for word in set(words)]) / len(set(words))
    if x_spam_probability >= nb_spam_class['spam'][1]:
        x_spam = 'spam'
    elif x_spam_probability <= nb_spam_class['nospam'][0]:
        x_spam = 'nospam'
    else:
        x_spam = 'undetermined'

    # write results
    with open(dir_output + filename, 'w+') as output_file:
        with open(dir_input + filename, 'r') as input_file:
            content = input_file.read()
        output_file.write('XSpam: {}\nXSpamProbability: {}'
                          .format(x_spam, x_spam_probability).rstrip('\r\n') + '\n' + content)

# ----------------------------------------------
# Bewertungsübersicht fuer Mail-Eingang ausgeben
...

# ----------------------------------------------
# cleanup
...
