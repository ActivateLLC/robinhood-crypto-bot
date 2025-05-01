import os

# OpenAI API Configuration
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'your_openai_api_key_here')

# Robinhood API Configuration
ROBINHOOD_CLIENT_ID = os.environ.get('ROBINHOOD_CLIENT_ID', '')
ROBINHOOD_CLIENT_SECRET = os.environ.get('ROBINHOOD_CLIENT_SECRET', '')

# Trading Configuration
DEFAULT_SYMBOL = 'BTC-USD'
DEFAULT_DATA_PERIOD = '1y'
DEFAULT_DATA_INTERVAL = '1h'

# Risk Management Parameters
RISK_FREE_RATE = 0.02  # Approximate annual risk-free rate
MAX_PORTFOLIO_RISK = 0.05  # 5% maximum portfolio risk
TRANSACTION_COST_RATE = 0.001  # 0.1% transaction cost
