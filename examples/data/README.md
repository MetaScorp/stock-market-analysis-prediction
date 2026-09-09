# Real data fixture

`AAPL_1987.csv` is real, unmodified daily OHLCV data for Apple stock
for calendar year 1987, pulled from the public dataset at
https://github.com/Pubmarks/datasets (published under their license,
sourced from Yahoo Finance's historical data). It's checked into the repo
so `07_real_data_case_study.py` and its test have something genuinely real
to run against without needing network access.

1987 was picked on purpose, not cherry-picked for a good result - it's
just old enough to be a small file, and it happens to include Black
Monday (Oct 19, 1987), which is a good stress test for the data-quality
checks and the VaR/regime code: real crashes look different from anything
the synthetic generator produces.
