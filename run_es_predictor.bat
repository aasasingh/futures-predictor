@echo off
:: ES Predictor v6 - Daily Auto-Run
:: Scheduled via Windows Task Scheduler
::
:: FRED key comes from the FRED_API_KEY env var, or fred_key.txt (gitignored).
:: PYTHON points at the install that has the packages; plain "py" now resolves
:: to a fresh 3.14 install with nothing in it.

set PYTHON=C:\Users\singh\AppData\Local\Programs\Python\Python314\python.exe

cd /d C:\Users\singh\recession_indicator
if not defined FRED_API_KEY if exist fred_key.txt set /p FRED_API_KEY=<fred_key.txt
"%PYTHON%" es_predictor_v6.py >> es_v6_run_log.txt 2>&1
