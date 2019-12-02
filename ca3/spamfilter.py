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

# different variables which are used as dictionary keys or magic values

KEY_TOTAL_OCCURRENCES = 'total_occurrences'
KEY_MAIL_OCCURRENCES = 'mail_occurrences'
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
                table[word][KEY_TOTAL_OCCURRENCES] += 1
            else:
                table[word] = {KEY_TOTAL_OCCURRENCES: 1, KEY_MAIL_OCCURRENCES: 0}
        for word in set(mails[mail_key]):
            if word in table:
                table[word][KEY_MAIL_OCCURRENCES] += 1
            else:
                table[word][KEY_MAIL_OCCURRENCES] = 1
    return {word: {KEY_MAIL_OCCURRENCES: table[word][KEY_MAIL_OCCURRENCES],
                   KEY_TOTAL_OCCURRENCES: table[word][KEY_TOTAL_OCCURRENCES],
                   KEY_XSPAM_PROBABILITY: table[word][KEY_MAIL_OCCURRENCES] / len(mails)} for word in table}


def save_evaluation_table(evaluation_table, file_name):
    """
    saves a evaluation table
    :param evaluation_table:
    :param file_name:
    :return:
    """
    table_file = open(abspath(dir_results + filename_nbwordtable + dir_separator + file_name), 'w')
    for word in evaluation_table:
        table_file.write('{}: {}\n'.format(word, evaluation_table[word]))
    table_file.close()


spam_table = create_evaluation_table(spam_mails)
save_evaluation_table(spam_table, 'spam.txt')
no_spam_table = create_evaluation_table(no_spam_mails)
save_evaluation_table(no_spam_table, 'no_spam.txt')

nb_table = {word: ((spam_table[word][KEY_XSPAM_PROBABILITY] if word in spam_table else 0) / (
        (spam_table[word][KEY_XSPAM_PROBABILITY] if word in spam_table else 0)
        + (no_spam_table[word][KEY_XSPAM_PROBABILITY] if word in no_spam_table else 0))) for word in
            set(list(chain.from_iterable(
                [list(chain.from_iterable(list(spam_mails.values()))),
                 list(chain.from_iterable(list(no_spam_mails.values())))])))}
save_evaluation_table(nb_table, 'total_probabilities.txt')

# ----------------------------------------------
# MailInput laden, bewerten und nach MailOutput schreiben
# Bewertungsklassifikation: WhiteList, NoSpam, undetermined, Spam, BlackList
input_mail_word_dict = load_mails(dir_input)
input_mails = {filename: {KEY_WORDS: words, KEY_XSPAM: None, KEY_XSPAM_PROBABILITY: None}
               for filename, words in input_mail_word_dict.items()}


# evaluation functions -> return whitelist, blacklist, spam, undetermined, nospam)


def naive_bayes(mail_dict, naive_bayes_not_last=None):
    """
    changes the xspam value if a word is at the whitelist
    :param mail_dict: dictionary which represents one mail
    :param naive_bayes_not_last: not used in this method (must be there because of other method calls in dict)
    :return:
    """
    x_spam_probability = sum([nb_table[word] if word in nb_table else 0
                              for word in set(mail_dict[KEY_WORDS])]) / len(set(mail_dict[KEY_WORDS]))
    if x_spam_probability >= nb_spam_classes['spam'][1]:
        x_spam = 'spam'
    elif x_spam_probability <= nb_spam_classes['nospam'][0]:
        x_spam = 'nospam'
    else:
        x_spam = 'undetermined'
    # if is executed before evaluation with black and whitelist
    # if mail_dict[KEY_XSPAM] not in [X_SPAM_VALUE_BLACKLIST, X_SPAM_VALUE_WHITELIST]:
    mail_dict[KEY_XSPAM] = x_spam
    mail_dict[KEY_XSPAM_PROBABILITY] = x_spam_probability


def whitelist(mail_dict, naive_bayes_not_last=True):
    """
    changes the xspam value if a word is at the whitelist
    only applied if naive_bayes is the last evaluation method or naive_bayes evaluation predicts 'undetermined'
    :param mail_dict: dictionary which represents one mail
    :param naive_bayes_not_last: flag if naive_bayes evaluation is last in the priority order
    :return:
    """
    if naive_bayes_not_last and mail_dict[KEY_XSPAM] == X_SPAM_VALUE_UNDETERMINED or not naive_bayes_not_last:
        if any(word in white_list for word in mail_dict[KEY_WORDS]):
            mail_dict[KEY_XSPAM] = X_SPAM_VALUE_WHITELIST


def blacklist(mail_dict, naive_bayes_not_last=True):
    """
    changes the xspam value if a word is at the blacklist
    only applied if naive_bayes is the last evaluation method or naive_bayes evaluation predicts 'undetermined'
    :param mail_dict: dictionary which represents one mail
    :param naive_bayes_not_last: flag if naive_bayes evaluation is last in the priority order
    :return:
    """
    if naive_bayes_not_last and mail_dict[KEY_XSPAM] == X_SPAM_VALUE_UNDETERMINED or not naive_bayes_not_last:
        if any(word in black_list for word in mail_dict[KEY_WORDS]):
            mail_dict[KEY_XSPAM] = X_SPAM_VALUE_BLACKLIST


evaluations = {EVALUATION_METHOD_WHITELIST: whitelist,
               EVALUATION_METHOD_BLACKLIST: blacklist,
               EVALUATION_METHOD_NAIVE_BAYES: naive_bayes}
naive_bayes_not_last = priority_order[-1] != EVALUATION_METHOD_NAIVE_BAYES

# do evaluation and write classification results

for filename, mail_dict in input_mails.items():

    for evaluation_method in priority_order:
        evaluations[evaluation_method](mail_dict, naive_bayes_not_last=naive_bayes_not_last)

    # write results
    with open(dir_output + filename, 'w+') as output_file:
        with open(dir_input + filename, 'r') as input_file:
            content = input_file.read()
        output_file.write('XSpam: {}\nXSpamProbability: {}'
                          .format(mail_dict[KEY_XSPAM], mail_dict[KEY_XSPAM_PROBABILITY]).rstrip(
            '\r\n') + '\n' + content)

# ----------------------------------------------
# Bewertungsübersicht fuer Mail-Eingang ausgeben
with open(abspath(dir_results + filename_results), 'w') as spamfilter_results_file:
    spamfilter_results_file.write('Priority order: {}\n\n'.format(priority_order))
    spamfilter_results_file.write('Naive bayes spam classes:\n'.format(nb_spam_classes))
    for _class in nb_spam_classes:
        spamfilter_results_file.write('{} - {}\n'.format(_class, nb_spam_classes[_class]))
    spamfilter_results_file.write('\nEvaluated emails:\n')
    for filename, mail_dict in input_mails.items():
        spamfilter_results_file.write('{}: {} ({})\n'
                                      .format(filename, mail_dict[KEY_XSPAM], mail_dict[KEY_XSPAM_PROBABILITY]))

# Log ausgeben
with open(abspath(dir_results + filename_logfile), 'w') as log_file:
    for line in log_lines:
        log_file.write(line)

# ----------------------------------------------
# cleanup
...
