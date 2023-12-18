pip install nltk
pip install sqlparse
python nltk_downloader.py
python evaluation.py --gold dev_gold.sql --pred predicted_sql.txt --etype all --db database --table data/tables.json