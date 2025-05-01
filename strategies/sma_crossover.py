import backtrader as bt

class SmaCrossOverStrategy(bt.Strategy):
    """Simple Moving Average Crossover Strategy."""
    params = (
        ('pfast', 10),  # Period for the fast moving average
        ('pslow', 30),  # Period for the slow moving average
        ('printlog', True),      # Whether to print transaction logs
    )

    def log(self, txt, dt=None, doprint=False):
        """Logging function for this strategy."""
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')

    def __init__(self):
        """Initialize the strategy."""
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add SimpleMovingAverage indicators
        self.sma_fast = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.pfast)
        self.sma_slow = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.pslow)

        # Add a CrossOver indicator
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)

        self.log(f'Strategy Initialized: Fast MA={self.params.pfast}, Slow MA={self.params.pslow}')

    def notify_order(self, order):
        """Notify of order status changes."""
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}'
                )
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}')

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        """Notify of trade completions."""
        if not trade.isclosed:
            return

        self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')

    def next(self):
        """Execute the strategy logic on each bar."""
        # Simply log the closing price of the series from the reference
        # self.log(f'Close, {self.dataclose[0]:.2f}')

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            # We are not in the market, look for a signal to enter
            # Fast MA crosses above Slow MA
            if self.crossover > 0:
                self.log(f'BUY CREATE, {self.dataclose[0]:.2f}')
                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()

        else:
            # We are in the market, look for a signal to exit
            # Fast MA crosses below Slow MA
            if self.crossover < 0:
                self.log(f'SELL CREATE, {self.dataclose[0]:.2f}')
                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()

    def stop(self):
        """Called when strategy termination is requested."""
        self.log(f'(Fast MA Period {self.params.pfast:2d}, Slow MA Period {self.params.pslow:2d}) Ending Value {self.broker.getvalue():.2f}', doprint=True)
