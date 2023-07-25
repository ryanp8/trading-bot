import numpy as np
import pandas as pd
import yfinance as yf
import math
import time
import multiprocessing

class LongShort:

    def __init__(self, start, end, balance=1000):
        self.sma_len = 30
        self.start = start
        self.end = end
        self.balance = balance
        self.price_record = {}
        self.z_scores = {}

        self.ls_backtest_results = []
        self.sp_backtest_results = []

    
    def add_price(self, symbol, price):
        # track self.sma_len prices 
        if symbol in self.price_record:
            target_prices = self.price_record[symbol]
            if target_prices[-1] != -1:
                target_prices[:-1] = target_prices[1:]
                target_prices[-1] = price
            else:
                target_prices[np.argmax(target_prices < 0)] = price
        else:
            self.price_record[symbol] = np.array([-1 for _ in range(self.sma_len)], dtype=float)
            self.price_record[symbol][0] = price


    def z_score(self, price, record):        
        std = np.std(record)
        avg = record.mean()
        if std == 0 or avg == 0:
            return 0

        return (price - avg) / std
    
    
    def process_prices(self, prices):
        valid_symbols = []
        z_scores = {}
        for symbol, price in prices.items():
            
            # only analyze the stock if data on it is provided for the current time
            if not math.isnan(price):
                self.add_price(symbol, price)
                if symbol in self.price_record and np.count_nonzero(self.price_record[symbol] == -1) == 0:
                    valid_symbols.append(symbol)
                    z_scores[symbol] = self.z_score(price, self.price_record[symbol])
        return (valid_symbols, z_scores)
    
    
    def clear_short(self, index, prices, z_scores, valid_symbols, short_positions, end):
        for symbol in short_positions:
            if symbol in valid_symbols:
                if short_positions[symbol] > 0:

                    # When a position is within a close range to the mean, buy back the shorted stock
                    if z_scores[symbol] < 1.5:
                        max_quantity = self.balance // prices[symbol]
                        buy_quantity = min(max_quantity, short_positions[symbol])
                        self.balance -= prices[symbol] * buy_quantity
                        short_positions[symbol] -= buy_quantity

                    # clear short positions regardless of prices when there is no more data left
                    if index == end - 1:
                        self.balance -= prices[symbol] * short_positions[symbol]
                        short_positions[symbol] = 0

    
    def clear_long(self, index, prices, z_scores, valid_symbols, long_positions, end):
        # Sell long positions when they have risen suffiently
        for symbol in long_positions:
            if symbol in valid_symbols and (z_scores[symbol] > -1.5 or index == end - 1):
                if long_positions[symbol] > 0:
                    self.balance += prices[symbol] * long_positions[symbol]
                    long_positions[symbol] = 0


    def sell_short(self, prices, z_scores, valid_symbols, short_positions):
        short_equity = 0
        i = -1
        short_symbol = valid_symbols[i]
        if z_scores[short_symbol] >= 3:
            while short_equity + prices[short_symbol] <= self.balance * 0.3:
                if short_symbol not in short_positions:
                    short_positions[short_symbol] = 1
                else:
                    short_positions[short_symbol] += 1

                short_equity += prices[short_symbol]
                short_symbol = valid_symbols[(i := -1) if z_scores[short_symbol] < 3 else (i := i - 1)]
        return short_equity
    
    
    def buy_long(self, prices, z_scores, valid_symbols, long_positions):
        long_equity = 0
        i = 0
        long_symbol = valid_symbols[i]
        if z_scores[long_symbol] <= -3:
            while long_equity + prices[long_symbol] <= self.balance:
                if long_symbol not in long_positions:
                    long_positions[long_symbol] = 1
                else:
                    long_positions[long_symbol] += 1
                
                long_equity += prices[long_symbol]
                long_symbol = valid_symbols[(i := 0) if z_scores[long_symbol] > -3 else (i := i + 1)]
        return long_equity


    def run(self, tickers, queue):
        data = yf.download(tickers, start=self.start, end=self.end, interval='1m')['Adj Close']
        data['id'] = range(len(data))
        data.set_index('id', inplace=True)

        short_positions = {}
        long_positions = {}
        
        # iterate over df to simulate collecting data and to ensure that all prices are valid (not NaN)
        start = time.perf_counter()
        for index, prices in data.iterrows():
            valid_symbols, z_scores = self.process_prices(prices)

            if valid_symbols:
                    
                valid_symbols.sort(key=lambda symbol: z_scores[symbol])

                self.clear_short(index, prices, z_scores, valid_symbols, short_positions, len(data))
                self.clear_long(index, prices, z_scores, valid_symbols, long_positions, len(data))

                # if it is not the last row of data, buy more
                if index < len(data) - 1:
                    short_equity = self.sell_short(prices, z_scores, valid_symbols, short_positions)
                    self.balance += short_equity

                    long_equity = self.buy_long(prices, z_scores, valid_symbols, long_positions)
                    self.balance -= long_equity

                
        # print(f'[balance]: {self.balance}')
        # print(f'time to complete: {time.perf_counter() - start}')
        print(f'[balance]: {self.balance}, {self.start}')
        queue.put(self.balance)
        return self.balance
    

def download_sp(start, end, queue):
    series = yf.download(['SPY'], start=start, end=end, interval='1m')['Adj Close']
    queue.put([series[0], series[1]])
    return [series[0], series[1]]
        

if __name__ == '__main__':
        tickers = list(pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'])
        dates = [['2023-07-17','2023-07-18'], ['2023-07-18','2023-07-19'], ['2023-07-19','2023-07-20'], ['2023-07-20','2023-07-21'], ['2023-07-21','2023-07-22']]

        start_time = time.perf_counter()

        ls_objs = [LongShort(date[0], date[1]) for date in dates]

        sp_q = multiprocessing.Queue()
        strat_q = multiprocessing.Queue()

        sp_processes = [multiprocessing.Process(target=download_sp, args=(date[0], date[1], sp_q)) for date in dates]
        strat_processes = [multiprocessing.Process(target=ls_objs[i].run, args=(tickers, strat_q)) for i, _ in enumerate(dates)]

        processes = sp_processes + strat_processes
        for process in processes:
            process.start()
        
        for process in processes:
            process.join()

        sp_q.put(None)
        strat_q.put(None)

        sp_data = np.array(list(iter(lambda : sp_q.get(timeout=0.00001), None)))
        strat_data = np.array(list(iter(lambda : strat_q.get(timeout=0.00001), None)))

        # Testing with no multiprocessing (80.1291152080521 sec vs 21.57460333313793 sec)
        # sp_data = np.array([download_sp(date[0], date[1], sp_q) for date in dates], dtype=float)
        # strat_data = np.array([ls_objs[i].run(tickers, strat_q) for i, _ in enumerate(dates)], dtype=float)

        sp_return = np.mean(np.apply_along_axis(lambda a: a[1] / a[0], 1, sp_data))
        strat_return = np.mean(strat_data / 1000)

        print(f'[time elapsed]: {time.perf_counter() - start_time}')
        print(f'[sp return]: {sp_return}')
        print(f'[strat return]: {strat_return}')
