"""Parameter fuer spamfilter.py"""
spf_version="[1.4]"
spf_vdate="10.12.2018"
dir_separator="\\"
log_suffix=".log"
dir_root="."+dir_separator
dir_results="dir.filter.results"+dir_separator
dir_input="dir.mail.input"+dir_separator
dir_nospam="dir.nospam"+dir_separator
dir_spam="dir.spam"+dir_separator
dir_output="dir.mail.output"+dir_separator
dir_temp="dir.temp"+dir_separator
filename_blacklist="blacklist"
filename_whitelist="whitelist"
filename_results="spamfilter.results"
filename_nbwordtable="nb.wordtable"
filename_logfile="spamfilter"+log_suffix
priorityorder="blacklist", "whitelist", "naive_bayes"
nb_wordtable={}
nb_spam_level=0.5               #nb_level greater_or_equal is spam
nb_nospam_level=0.2             #nb_level loweror equal is nospam
                                 #in between is undetermined
nb_spam_class={"spam":(1.0,nb_spam_level), 
               "undetermined":(nb_spam_level,nb_nospam_level), 
               "nospam":(nb_nospam_level,0.0)}
char_replaces={'"':' ', "\n":" ", "_":" ", ",":" ", "-":" ", "+":" ", "„":" ", "’":" ", "“":" ",
               "%":" ", ".":" ", "\t":" ", "[":" ", "]":" ", "<":" ", ">":" ", "/":" ", "=":" ",
               "(":" ", ")":" ", "…":" ", "  ":" "}
words_ignore=["", " ", ":", "*", "#", "!", "&", "/", ";", "?", "@", "|", "©", "®", "´", "·"]
