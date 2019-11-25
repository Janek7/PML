@cls
@ECHO run_spamfilter [2.0]
@ECHO.
@ECHO === cleanup last run ===
del /Q dir.mail.output\*.*
del /Q dir.filter.results\*.*
@ECHO.
@ECHO === prepare params ===
copy spamfilter.params params.py
@ECHO.
@ECHO === run filter ===
spamfilter.py
@ECHO.
:@GOTO ende
@ECHO === show results ===
FOR %%f IN (dir.filter.results\*.*) DO "C:\Program Files\Notepad++\Notepad++.exe" %%f
@ECHO.
@ECHO === scanned mails ===
@DIR /B dir.mail.output\*.*
:@Pause
:ende
@REM (C) 2019 by Rainer Gerten