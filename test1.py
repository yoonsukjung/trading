import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Strategy:
    def __init__(self, data):
        self.data = data
        self.signals = pd.DataFrame(index=data['asset1'].index)
        self.position = 0

    def generate_signals(self):
        raise NotImplementedError("Should implement generate_signals method")

    def execute_trade(self, index, signal):
        if signal == "long":
            self.position = 1
            self.signals.loc[index, 'trade'] = "long"
            logger.info(f"Executed long trade at index {index}")
        elif signal == "short":
            self.position = -1
            self.signals.loc[index, 'trade'] = "short"
            logger.info(f"Executed short trade at index {index}")
        elif signal == "close":
            self.position = 0
            self.signals.loc[index, 'trade'] = "close"
            logger.info(f"Closed position at index {index}")


class CointegrationStrategy(Strategy):
    def __init__(self, data, spread_mean, spread_std, slope, crypto1, crypto2, entry_z_score=1.2, exit_z_score=0.2,
                 stop_z_score=3,
                 fee=0.001, slippage=0.001):
        super().__init__(data)
        self.data1 = data['asset1'].copy()
        self.data2 = data['asset2'].copy()
        self.spread_mean = spread_mean
        self.spread_std = spread_std
        self.slope = slope
        self.crypto1 = crypto1
        self.crypto2 = crypto2
        self.entry_z_score = entry_z_score
        self.exit_z_score = exit_z_score
        self.stop_z_score = stop_z_score
        self.fee = fee
        self.slippage = slippage

    def calculate_z_score(self):
        logger.info("Calculating z-score")
        self.data1['log_spread'] = np.log(self.data1['close']) - self.slope * np.log(self.data2['close'])

        # Debugging output for log_spread, spread_mean, spread_std
        logger.info(f"Spread mean: {self.spread_mean}")
        logger.info(f"Spread std: {self.spread_std}")
        logger.info(f"Slope: {self.slope}")
        logger.info(f"Sample log_spread values: {self.data1['log_spread'].head()}")

        self.data1['z_score'] = (self.data1['log_spread'] - self.spread_mean) / self.spread_std
        self.signals['z_score'] = self.data1['z_score']

        # Debugging output for z_score
        logger.info(f"Sample z_score values: {self.data1['z_score'].head()}")

    def generate_trade_signals(self):
        logger.info("Generating trade signals")
        self.signals['trade'] = np.nan
        self.signals['price1'] = np.nan
        self.signals['price2'] = np.nan

        entry_short = self.signals['z_score'] > self.entry_z_score
        entry_long = self.signals['z_score'] < -self.entry_z_score
        exit_position = self.signals['z_score'].abs() < self.exit_z_score
        stop_loss = self.signals['z_score'].abs() > self.stop_z_score

        stop_triggered = False  # 추가된 플래그

        for index in self.signals.index:
            if self.position == 0:
                if stop_triggered:
                    if exit_position[index]:
                        stop_triggered = False
                else:
                    if entry_short[index]:
                        self.signals.at[index, 'price1'] = self.data1.at[index, 'close'] * (1 + self.slippage)
                        self.signals.at[index, 'price2'] = self.data2.at[index, 'close'] * (1 - self.slippage)
                        self.execute_trade(index, 'short')
                    elif entry_long[index]:
                        self.signals.at[index, 'price1'] = self.data1.at[index, 'close'] * (1 - self.slippage)
                        self.signals.at[index, 'price2'] = self.data2.at[index, 'close'] * (1 + self.slippage)
                        self.execute_trade(index, 'long')
            elif self.position != 0:
                if exit_position[index] or stop_loss[index]:
                    self.signals.at[index, 'price1'] = self.data1.at[index, 'close']
                    self.signals.at[index, 'price2'] = self.data2.at[index, 'close']
                    self.execute_trade(index, 'close')
                    if stop_loss[index]:
                        stop_triggered = True

        self.signals['position'] = self.signals['trade'].ffill().shift().fillna(0).replace(
            {'long': 1, 'short': -1, 'close': 0})

    def generate_signals(self):
        logger.info("Generating signals")
        try:
            self.calculate_z_score()
            self.generate_trade_signals()

            logger.info("Signals generated")
            print(self.signals[['z_score', 'trade', 'price1', 'price2']])  # 신호 출력
        except Exception as e:
            logger.error(f"Error in generating signals: {e}")


class Backtester:
    def __init__(self, strategy, start_date=None, end_date=None, fee=0.001, slippage=0.001, result_path=None):
        self.strategy = strategy
        self.start_date = start_date
        self.end_date = end_date
        self.fee = fee
        self.slippage = slippage
        self.result_path = result_path
        self.trade_log = []

    def run_backtest(self):
        logger.info("Running backtest")
        try:
            self.prepare_data()
            self.strategy.generate_signals()
            self.calculate_performance()
            self.save_trade_log()
        except Exception as e:
            logger.error(f"Error in running backtest: {e}")

    def prepare_data(self):
        logger.info("Preparing data")
        if self.start_date and self.end_date:
            self.strategy.data1 = self.strategy.data1.loc[
                (self.strategy.data1.index >= self.start_date) &
                (self.strategy.data1.index <= self.end_date)
                ]
            self.strategy.data2 = self.strategy.data2.loc[
                (self.strategy.data2.index >= self.start_date) &
                (self.strategy.data2.index <= self.end_date)
                ]

    def calculate_performance(self):
        logger.info("Calculating performance")
        try:
            returns = self.calculate_trade_returns()
            self.add_returns_to_signals(returns)
            self.strategy.signals['equity'] = (self.strategy.signals['returns'] + 1).cumprod()
            self.print_report(len(returns))
            self.plot_performance()
        except Exception as e:
            logger.error(f"Error in calculating performance: {e}")

    def calculate_trade_returns(self):
        logger.info("Calculating trade returns")
        returns = []
        trades = self.strategy.signals[self.strategy.signals['trade'].notna()]
        print(trades)  # 거래 내역 출력
        entry_price1 = 0
        entry_price2 = 0
        entry_index = None
        entry_type = None

        for index, trade in trades.iterrows():
            if trade['trade'] in ['long', 'short']:
                entry_price1 = self.strategy.data1.loc[index, 'close']
                entry_price2 = self.strategy.data2.loc[index, 'close']
                entry_index = index
                entry_type = trade['trade']
            elif trade['trade'] == 'close' and entry_price1 != 0 and entry_price2 != 0:
                exit_price1 = self.strategy.data1.loc[index, 'close']
                exit_price2 = self.strategy.data2.loc[index, 'close']
                trade_return = self.calculate_single_trade_return(entry_price1, exit_price1, entry_price2, exit_price2,
                                                                  entry_type)
                returns.append((index, trade_return))
                self.trade_log.append({
                    'index': index,
                    'trade': trade['trade'],
                    'entry_price1': entry_price1,
                    'exit_price1': exit_price1,
                    'entry_price2': entry_price2,
                    'exit_price2': exit_price2,
                    'return': trade_return,
                    'z_score': self.strategy.signals.loc[index, 'z_score'],
                    'position': self.strategy.signals.loc[index, 'position']
                })
                entry_price1 = 0
                entry_price2 = 0
                entry_index = None
                entry_type = None

        return returns

    def calculate_single_trade_return(self, entry_price1, exit_price1, entry_price2, exit_price2, entry_type):
        logger.info("Calculating single trade return")
        if entry_type == 'long':
            trade_return = (exit_price1 - entry_price1) / entry_price1 - \
                           (exit_price2 - entry_price2) / entry_price2
        elif entry_type == 'short':
            trade_return = (entry_price1 - exit_price1) / entry_price1 - \
                           (entry_price2 - exit_price2) / entry_price2
        trade_return -= self.fee * 2  # 진입 및 청산 시 수수료
        trade_return -= self.slippage * 2  # 진입 및 청산 시 슬리피지
        return trade_return

    def add_returns_to_signals(self, returns):
        logger.info("Adding returns to signals")
        if returns:
            returns_df = pd.DataFrame(returns, columns=['index', 'returns']).set_index('index')
            self.strategy.signals = self.strategy.signals.join(returns_df, how='left')
            self.strategy.signals['returns'].fillna(0, inplace=True)
        else:
            self.strategy.signals['returns'] = 0

    def save_trade_log(self):
        if self.result_path:
            trade_log_df = pd.DataFrame(self.trade_log)
            plot_title = f"{self.strategy.crypto1} and {self.strategy.crypto2} Performance"
            result_file = os.path.join(self.result_path, f"{plot_title.replace(' ', '_')}.csv")
            trade_log_df.to_csv(result_file, index=False)
            logger.info(f"Trade log saved to {result_file}")

    def print_report(self, total_trades):
        logger.info("Printing report")
        try:
            total_returns = self.strategy.signals['equity'].iloc[-1] - 1
            if pd.isna(total_returns):
                total_returns = 0

            annualized_returns = (1 + total_returns) ** (365 / len(self.strategy.signals)) - 1
            volatility = self.strategy.signals['returns'].std() * np.sqrt(252)
            sharpe_ratio = annualized_returns / volatility
            max_drawdown = self.calculate_max_drawdown(self.strategy.signals['equity'])
            winning_trades = (self.strategy.signals['returns'] > 0).sum()
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            avg_trade_return = self.strategy.signals[self.strategy.signals['returns'] != 0]['returns'].mean()

            print(f"Total Returns: {total_returns:.2%}")
            print(f"Annualized Returns: {annualized_returns:.2%}")
            print(f"Volatility: {volatility:.2%}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Max Drawdown: {max_drawdown:.2%}")
            print(f"Win Rate: {win_rate:.2%}")
            print(f"Average Trade Return: {avg_trade_return:.2%}")
            print(f"Total Number of Trades: {total_trades}")
        except Exception as e:
            logger.error(f"Error in printing report: {e}")

    def calculate_max_drawdown(self, equity_curve):
        logger.info("Calculating max drawdown")
        try:
            drawdown = equity_curve / equity_curve.cummax() - 1
            return drawdown.min()
        except Exception as e:
            logger.error(f"Error in calculating max drawdown: {e}")
            return 0

    def plot_performance(self):
        logger.info("Plotting performance")
        try:
            plt.figure(figsize=(14, 10))

            plot_title = f"{self.strategy.crypto1} and {self.strategy.crypto2} Performance"

            # 첫 번째 그래프: 자산 가격 및 트레이딩 신호
            plt.subplot(3, 1, 1)
            plt.plot(self.strategy.data1.index, self.strategy.data1['close'], label='Asset 1 Close Price')
            plt.plot(self.strategy.data2.index, self.strategy.data2['close'], label='Asset 2 Close Price')
            if 'log_spread' in self.strategy.data1.columns:
                plt.plot(self.strategy.data1.index, self.strategy.data1['log_spread'], label='Log Spread')

            long_signals = self.strategy.signals[self.strategy.signals['trade'] == 'long']
            short_signals = self.strategy.signals[self.strategy.signals['trade'] == 'short']
            close_signals = self.strategy.signals[self.strategy.signals['trade'] == 'close']

            plt.scatter(long_signals.index, self.strategy.data1.loc[long_signals.index, 'close'], marker='^', color='g',
                        label='Long Signal', alpha=1)
            plt.scatter(short_signals.index, self.strategy.data1.loc[short_signals.index, 'close'], marker='v',
                        color='r',
                        label='Short Signal', alpha=1)
            plt.scatter(close_signals.index, self.strategy.data1.loc[close_signals.index, 'close'], marker='o',
                        color='b',
                        label='Close Signal', alpha=1)

            plt.legend()
            plt.title('Asset Prices and Trading Signals')
            plt.xlim(self.strategy.data1.index.min(), self.strategy.data1.index.max())

            # 두 번째 그래프: Equity Curve
            plt.subplot(3, 1, 2)
            plt.plot(self.strategy.signals.index, self.strategy.signals['equity'], label='Equity Curve', color='purple')
            plt.legend()
            plt.title('Equity Curve')
            plt.xlim(self.strategy.data1.index.min(), self.strategy.data1.index.max())

            # 세 번째 그래프: Z-score 및 트레이딩 신호
            plt.subplot(3, 1, 3)
            plt.plot(self.strategy.signals.index, self.strategy.signals['z_score'], label='Z-score')
            plt.axhline(y=self.strategy.entry_z_score, color='r', linestyle='--', label='Entry Z-score')
            plt.axhline(y=-self.strategy.entry_z_score, color='r', linestyle='--')
            plt.axhline(y=self.strategy.exit_z_score, color='g', linestyle='--', label='Exit Z-score')
            plt.axhline(y=-self.strategy.exit_z_score, color='g', linestyle='--')

            plt.scatter(long_signals.index, self.strategy.signals.loc[long_signals.index, 'z_score'], marker='^',
                        color='g', label='Long Signal', alpha=1)
            plt.scatter(short_signals.index, self.strategy.signals.loc[short_signals.index, 'z_score'], marker='v',
                        color='r', label='Short Signal', alpha=1)
            plt.scatter(close_signals.index, self.strategy.signals.loc[close_signals.index, 'z_score'], marker='o',
                        color='b', label='Close Signal', alpha=1)

            plt.legend()
            plt.title('Z-score and Trading Signals')
            plt.xlim(self.strategy.data1.index.min(), self.strategy.data1.index.max())

            plt.suptitle(plot_title)
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            # Save the plot as an image file in the result_path
            if self.result_path:
                image_filename = os.path.join(self.result_path, f"{plot_title.replace(' ', '_')}.png")
            else:
                image_filename = f"{plot_title.replace(' ', '_')}.png"
            plt.savefig(image_filename)
            plt.show()
        except Exception as e:
            logger.error(f"Error in plotting performance: {e}")


def load_data(file1_path, base_path):
    logger.info("Loading data from file1")
    try:
        file1 = pd.read_csv(file1_path)
        first_row = file1.iloc[0]
        crypto1 = first_row['crypto1']
        crypto2 = first_row['crypto2']
        spread_mean = first_row['spread_mean']
        spread_std = first_row['spread_std']
        slope = first_row['HR']

        data1_path = os.path.join(base_path, f"{crypto1}_USDT_1h.csv")
        data2_path = os.path.join(base_path, f"{crypto2}_USDT_1h.csv")

        logger.info(f"Loading data1 from {data1_path}")
        if not os.path.exists(data1_path):
            raise FileNotFoundError(f"{data1_path} not found")
        data1 = pd.read_csv(data1_path)
        data1['timestamp'] = pd.to_datetime(data1['timestamp'])

        logger.info(f"Loading data2 from {data2_path}")
        if not os.path.exists(data2_path):
            raise FileNotFoundError(f"{data2_path} not found")
        data2 = pd.read_csv(data2_path)
        data2['timestamp'] = pd.to_datetime(data2['timestamp'])

        if 'close' not in data1.columns or 'close' not in data2.columns:
            raise ValueError("Both data1 and data2 must contain 'close' column")

        data = {'asset1': data1.set_index('timestamp'), 'asset2': data2.set_index('timestamp')}
        return data, spread_mean, spread_std, slope, crypto1, crypto2
    except Exception as e:
        logger.error(f"Error in loading data: {e}")
        raise


# 사용 예제
def run_backtest_for_row(row_index):
    try:
        base_path = "/Users/yoonsukjung/PycharmProjects/Trading/data/1h_2024-07-01"
        result_path = "/Users/yoonsukjung/PycharmProjects/Trading/Statistical_Arbitrage/Results"
        file1_path = os.path.join(base_path, "coint_pairs.csv")

        # coint_pairs.csv 파일 로드
        file1 = pd.read_csv(file1_path)

        # 원하는 행의 데이터 로드 및 백테스트 실행
        row = file1.iloc[row_index]
        crypto1 = row['crypto1']
        crypto2 = row['crypto2']
        spread_mean = row['spread_mean']
        spread_std = row['spread_std']
        slope = row['HR']

        data1_path = os.path.join(base_path, f"{crypto1}_USDT_1h.csv")
        data2_path = os.path.join(base_path, f"{crypto2}_USDT_1h.csv")

        data1 = pd.read_csv(data1_path)
        data1['timestamp'] = pd.to_datetime(data1['timestamp'])

        data2 = pd.read_csv(data2_path)
        data2['timestamp'] = pd.to_datetime(data2['timestamp'])

        data = {'asset1': data1.set_index('timestamp'), 'asset2': data2.set_index('timestamp')}

        start_date = '2024-01-01'
        end_date = '2024-06-30'

        strategy = CointegrationStrategy(data, spread_mean, spread_std, slope, crypto1, crypto2)
        backtester = Backtester(strategy, start_date=start_date, end_date=end_date, fee=0.001, slippage=0.001, result_path=result_path)
        backtester.run_backtest()

    except Exception as e:
        logger.error(f"Error in the main execution: {e}")

# 예제 실행 (예: 0행과 1행에 대해 실행)
run_backtest_for_row(1)
