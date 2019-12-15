"""SpamFilter [a.b] 17.12.2018

Spamfilter auf Basis von Whitelisting, Blacklisting und Naive Bayes Filter

Parameter aus spamfilter.params // params.py

Ablaufsteuerung (Windows): run_spamfilter.cmd
"""
# ----------------------------------------------
# Module laden
from datetime import date
from os import listdir
from os.path import isfile, join, abspath, realpath
import statistics

# Parameter laden
from ca3.params import *

VERSION = 1.0

# different variables which are used as dictionary keys or magic values
KEY_SENDER = 'sender_email'
KEY_HEADER = 'header_section'
KEY_TOTAL = 'total_occurrences'
KEY_MAILS = 'mail_occurrences'
KEY_PROBABILITY = 'mean'

KEY_SP_MAILS = 'spam_mails'
KEY_NS_MAILS = 'no_spam_mails'
KEY_SP_WORDS = 'spam_mail_words'
KEY_NS_WORDS = 'no_spam_mail_words'
KEY_WORDS_PROBABILITY = 'words_probability'
KEY_MAIL_PROBABILITY = 'mails_probability'

KEY_XSPAM_PROBABILITY = 'x_spam_probability'
KEY_WORDS = 'words'
KEY_XSPAM = 'x_spam'
KEY_BLACK_LIST_FLAG = 'black_list_flag'
KEY_WHITE_LIST_FLAG = 'white_list_flag'

X_SPAM_VALUE_WHITELIST = 'whitelist'
X_SPAM_VALUE_BLACKLIST = 'blacklist'
X_SPAM_VALUE_SPAM = 'spam'
X_SPAM_VALUE_UNDETERMINED = 'undetermined'
X_SPAM_VALUE_NOSPAM = 'nospam'

# logging
log_lines = []


# ----------------------------------------------
# Funktionen allgemein
def load_lines(path):
    with open(path, 'r') as file:
        return [line for line in file.read().split('\n') if line != '']


def replace_chars(string):
    for char_to_replace in char_replaces:
        string = string.replace(char_to_replace, char_replaces[char_to_replace])
    return string


# Blacklist laden
black_list = load_lines(filename_blacklist)
log_lines.append('blacklist:')
log_lines.append(str(black_list))
log_lines.append('\n')

# Whitelist laden
white_list = load_lines(filename_whitelist)
log_lines.append('whitelist:')
log_lines.append(str(white_list))
log_lines.append('\n')


# Mails laden
def load_mails(dir):
    mails = {}
    for mail_file_name in [file for file in listdir(dir) if isfile(join(dir, file))]:
        with open(dir + mail_file_name, mode='r') as mail_file:

            content_string = mail_file.read()
            header = content_string[:content_string.find('\n\n')]
            body = content_string[content_string.find('\n\n'):]
            mails[mail_file_name] = {KEY_HEADER: header, KEY_SENDER: None, KEY_WORDS: []}

            for line in header.split('\n'):
                if line.startswith('Von:'):
                    mails[mail_file_name][KEY_SENDER] = line[line.find('<') + 1: line.find('>')]
                elif line.startswith('Betreff:'):
                    mails[mail_file_name][KEY_WORDS] += replace_chars(line)[line.find(':') + 2:].split(' ')
            mails[mail_file_name][KEY_WORDS] \
                += ([word for word in replace_chars(body).split(' ')
                     if not any(ignore in word for ignore in words_ignore) and word != ''])
    log_lines.append('directory_laden({}):'.format(dir))
    log_lines.append(str(mails.keys()))
    log_lines.append('\n')
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
        for word in mails[mail_key][KEY_WORDS]:
            if word in table:
                table[word][KEY_TOTAL] += 1
            else:
                table[word] = {KEY_TOTAL: 1, KEY_MAILS: 0}
        for word in set(mails[mail_key][KEY_WORDS]):
            if word in table:
                table[word][KEY_MAILS] += 1
            else:
                table[word][KEY_MAILS] = 1

    return table


spam_table = create_evaluation_table(spam_mails)
no_spam_table = create_evaluation_table(no_spam_mails)

# get a merged list of all words
words = []
for spam_mail_dict in spam_mails.values():
    for word in spam_mail_dict[KEY_WORDS]:
        words.append(word)
for no_spam_mail_dict in no_spam_mails.values():
    for word in no_spam_mail_dict[KEY_WORDS]:
        words.append(word)
words = [word for word in words if not word.isnumeric()]

# create and save nb word table
nb_table = {word: {
    KEY_SP_MAILS: spam_table[word][KEY_MAILS] if word in spam_table else 0,
    KEY_SP_WORDS: spam_table[word][KEY_TOTAL] if word in spam_table else 0,
    KEY_NS_MAILS: no_spam_table[word][KEY_MAILS] if word in no_spam_table else 0,
    KEY_NS_WORDS: no_spam_table[word][KEY_TOTAL] if word in no_spam_table else 0
} for word in list(set(words))}

for word in nb_table:
    temp = nb_table[word][KEY_SP_MAILS] + nb_table[word][KEY_NS_MAILS]
    nb_table[word][KEY_MAIL_PROBABILITY] = nb_table[word][KEY_SP_MAILS] / temp if temp != 0 else 0  # prevent / 0
    temp = nb_table[word][KEY_SP_WORDS] + nb_table[word][KEY_NS_WORDS]
    nb_table[word][KEY_WORDS_PROBABILITY] = nb_table[word][KEY_SP_WORDS] / temp if temp != 0 else 0  # prevent / 0

nb_table_file_name = dir_results + filename_nbwordtable
table_file = open(abspath(nb_table_file_name), 'w')
for word in sorted(nb_table):
    table_file.write('{}: {}\n'.format(word, nb_table[word]))
table_file.close()

log_lines.append('wordtable naive bayes ({}): {} entries'.format(nb_table_file_name, len(nb_table.keys())))
log_lines.append('\n')

# ----------------------------------------------
# MailInput laden, bewerten und nach MailOutput schreiben
# Bewertungsklassifikation: WhiteList, NoSpam, undetermined, Spam, BlackList
input_mail_dict = load_mails(dir_input)
input_mails = {filename: {KEY_HEADER: mail_dict[KEY_HEADER], KEY_SENDER: mail_dict[KEY_SENDER],
                          KEY_WORDS: mail_dict[KEY_WORDS], KEY_XSPAM: None, KEY_XSPAM_PROBABILITY: None}
               for filename, mail_dict in input_mail_dict.items()}


# evaluation functions -> return whitelist, blacklist, spam, undetermined, nospam)


def naive_bayes(mail_dict):
    """
    changes the xspam value if a word is at the whitelist
    :param mail_dict: dictionary which represents one mail
    :param naive_bayes_not_last: not used in this method (must be there because of other method calls in dict)
    :return:
    """
    word_set = set(mail_dict[KEY_WORDS])
    mail_dict[KEY_MAIL_PROBABILITY] = sum([nb_table[word][KEY_MAIL_PROBABILITY] if word in nb_table else 0
                                           for word in word_set]) / len(word_set)
    mail_dict[KEY_WORDS_PROBABILITY] = sum([nb_table[word][KEY_WORDS_PROBABILITY] if word in nb_table else 0
                                            for word in word_set]) / len(word_set)

    x_spam_probability = statistics.mean([mail_dict[KEY_MAIL_PROBABILITY], mail_dict[KEY_WORDS_PROBABILITY]])
    if x_spam_probability >= nb_spam_classes['spam'][1]:
        x_spam = 'spam'
    elif x_spam_probability <= nb_spam_classes['nospam'][0]:
        x_spam = 'nospam'
    else:
        x_spam = 'undetermined'

    if mail_dict[KEY_XSPAM] is None:  # prevent override of black and whitelist
        mail_dict[KEY_XSPAM] = x_spam
    mail_dict[KEY_XSPAM_PROBABILITY] = x_spam_probability


def whitelist(mail_dict):
    """
    changes the xspam value if a word is at the whitelist
    only applied if naive_bayes is the last evaluation method or naive_bayes evaluation predicts 'undetermined'
    :param mail_dict: dictionary which represents one mail
    :param naive_bayes_not_last: flag if naive_bayes evaluation is last in the priority order
    :return:
    """
    if any(mail_dict[KEY_SENDER] in word for word in white_list):
        mail_dict[KEY_XSPAM] = X_SPAM_VALUE_WHITELIST


def blacklist(mail_dict):
    """
    changes the xspam value if a word is at the blacklist
    only applied if naive_bayes is the last evaluation method or naive_bayes evaluation predicts 'undetermined'
    :param mail_dict: dictionary which represents one mail
    :return:
    """
    if any(mail_dict[KEY_SENDER] in word for word in black_list):
        mail_dict[KEY_XSPAM] = X_SPAM_VALUE_BLACKLIST


# "whitelist", "blacklist", "naive_bayes"
evaluations = {'whitelist': whitelist,
               'blacklist': blacklist,
               'naive_bayes': naive_bayes}

# do evaluation and write classification results

log_lines.append('processing mails:')
for filename, mail_dict in input_mails.items():

    for evaluation_method in priorityorder:
        evaluations[evaluation_method](mail_dict)

    # write results
    with open(dir_output + filename, 'w+') as output_file:
        with open(dir_input + filename, 'r') as input_file:
            content = input_file.read()
        output_file.write('XSpam: {}\nXSpamProbability: {}'
                          .format(mail_dict[KEY_XSPAM], mail_dict[KEY_XSPAM_PROBABILITY]).rstrip(
            '\r\n') + '\n' + content)

    log_lines.append('[{}] {}'.format(mail_dict[KEY_XSPAM], filename))

log_lines.append('processed mails {}'.format(len(input_mails)))

# ----------------------------------------------
# Bewertungsübersicht fuer Mail-Eingang ausgeben
with open(abspath(dir_results + filename_results), 'w') as spamfilter_results_file:
    spamfilter_results_file.write('{} [{}] {}\n'.format(__file__, str(VERSION), date.today().strftime("%d.%m.%Y")))
    spamfilter_results_file.write('priorityorder: {}\n'.format(priorityorder))
    spamfilter_results_file.write('nb_spam_class: {}\n\n'.format(nb_spam_classes))

    spamfilter_results_file.write('Evaluated emails:\n')
    for filename, mail_dict in input_mails.items():
        spamfilter_results_file.write('=' * 60 + '\n\n')
        spamfilter_results_file.write(mail_dict[KEY_HEADER] + '\n')
        spamfilter_results_file.write('-' * 10 + '\n')
        spamfilter_results_file.write(' ' * 3 + 'mail_probability: {}\n'.format(mail_dict[KEY_MAIL_PROBABILITY]))
        spamfilter_results_file.write(' ' * 3 + 'words_probability: {}\n'.format(mail_dict[KEY_WORDS_PROBABILITY]))
        spamfilter_results_file.write(' ' * 3 + 'spam_probability: {}\n'.format(mail_dict[KEY_XSPAM_PROBABILITY]))
        spamfilter_results_file.write(' ' * 3 + 'spam_class: {}\n'.format(mail_dict[KEY_XSPAM]))

# Log ausgeben
with open(abspath(dir_results + filename_logfile), 'w') as log_file:
    for line in log_lines:
        log_file.write(line + ('\n' if line != '\n' else ''))

# ----------------------------------------------
# cleanup
...
