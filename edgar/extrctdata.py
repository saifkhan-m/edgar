import re

dir = 'sec-edgar-filings/MSFT/10-K/0001564590-20-034944/full-submission.txt'
with open(dir, 'r') as f:
    note = f.read()
