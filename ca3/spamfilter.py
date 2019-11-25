"""SpamFilter [a.b] 17.12.2018

Spamfilter auf Basis von Whitelisting, Blacklisting und Naive Bayes Filter

Parameter aus spamfilter.params // params.py

Ablaufsteuerung (Windows): run_spamfilter.cmd
"""
#----------------------------------------------
#Module laden
import os

#Parameter laden
from params import *

#logging
...

#----------------------------------------------
#Funktionen allgemein
...

#----------------------------------------------
#Blacklist laden
...

#Whitelist laden
...

#Spam-Mails laden
...

#NoSpam-Mails laden
...

#----------------------------------------------
#Bewertungstabellen fuer Naive Bayes erstellen und in eigene Datei protokollieren
...

#----------------------------------------------
#MailInput laden, bewerten und nach MailOutput schreiben
#Bewertungsklassifikation: WhiteList, NoSpam, undetermined, Spam, BlackList
...

#----------------------------------------------
#Bewertungsübersicht fuer Mail-Eingang ausgeben
...

#----------------------------------------------
#cleanup
...
