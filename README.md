# trading-bot

Bot created to learn about more about trading, parallelism, and concurrency. 

### Strategy
130/30 long-short equity strategy with mean reversion. Ranks stocks in the S&P 500 by how many standard deviations the price is from their previously calculated average and buys/sells accordingly. \
Some backtesting shows that the strategy performs 2.33% better than the S&P 500, but more testing is needed.

Learned more about multiprocessing and multithreading to improve performance. Backtesting with multiprocessing performed 3.7x faster than without multiprocessing.
