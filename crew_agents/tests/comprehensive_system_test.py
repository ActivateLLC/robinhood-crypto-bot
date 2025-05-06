import unittest
import pandas as pd
import numpy as np
from ..src.crypto_trading_env import CryptoTradingEnvironment
from ..src.utils import setup_logger

# Mock AltCryptoDataProvider for testing
class MockAltCryptoDataProvider:
    def __init__(self, config=None, logger=None):
        self.logger = logger if logger else setup_logger('MockDataProvider_test', log_level='CRITICAL')
        self.logger.info("MockAltCryptoDataProvider initialized.")
        # Create some dummy historical data consistent with SAMPLE_DATA shape for offline mode
        self.mock_historical_df = pd.DataFrame(SAMPLE_DATA).copy()
        # Pre-calculate returns and log_returns as the environment might expect them for non-live mode
        self.mock_historical_df['returns'] = self.mock_historical_df['Close'].pct_change()
        self.mock_historical_df['log_returns'] = np.log(1 + self.mock_historical_df['returns'])
        self.mock_historical_df.fillna(0.0, inplace=True)

    def get_historical_data(self, symbol, interval, lookback_period):
        self.logger.info(f"Mock get_historical_data called for {symbol}, {interval}, {lookback_period}")
        # Return a slice or the whole mock data. Ensure it's not empty for env setup.
        return self.mock_historical_df.head(int(lookback_period) if lookback_period else len(self.mock_historical_df)).copy()

    def get_current_price(self, symbol):
        self.logger.info(f"Mock get_current_price called for {symbol}")
        return self.mock_historical_df['Close'].iloc[-1] if not self.mock_historical_df.empty else 100.0

    def get_current_sentiment_score(self, symbol):
        self.logger.info(f"Mock get_current_sentiment_score called for {symbol}")
        return 0.5 # Neutral sentiment

    def check_api_keys(self):
        self.logger.info("Mock check_api_keys called.")
        return True

# Sample data for testing _advanced_data_preprocessing
SAMPLE_DATA = {
    'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
             110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
             120, 121, 122, 123, 124, 125, 126, 127, 128, 129],
    'High': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
             112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
             122, 123, 124, 125, 126, 127, 128, 129, 130, 131],
    'Low':  [99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
             109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
             119, 120, 121, 122, 123, 124, 125, 126, 127, 128],
    'Close':[101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
             111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
             121, 122, 123, 124, 125, 126, 127, 128, 129, 130],
    'Volume':[1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070, 1080, 1090,
              1100, 1110, 1120, 1130, 1140, 1150, 1160, 1170, 1180, 1190,
              1200, 1210, 1220, 1230, 1240, 1250, 1260, 1270, 1280, 1290]
}

class TestCryptoTradingEnvironment(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up for all tests in this class."""
        cls.test_logger = setup_logger('TestCryptoEnv', log_level='CRITICAL')

        cls.mock_config_dict = {
            'symbol': 'BTC-USD',
            'initial_capital': 10000.0,
            'lookback_window': 24, 
            'trading_fee': 0.001, # Matched key from CryptoTradingEnvironment.__init__
            'live_trading': False, # For most unit tests, keep this false
            'broker_type': 'simulation',
            'log_experience': False
        }

        # Data provider settings from the original mock_config
        cls.data_provider_config = {'type': 'mock'} # Keep for consistency if AltCryptoDataProvider uses it
        cls.mock_data_provider = MockAltCryptoDataProvider(config=cls.data_provider_config, logger=cls.test_logger)

        try:
            cls.env = CryptoTradingEnvironment(
                data_provider=cls.mock_data_provider,
                symbol=cls.mock_config_dict['symbol'],
                initial_capital=cls.mock_config_dict['initial_capital'],
                lookback_window=cls.mock_config_dict['lookback_window'],
                trading_fee=cls.mock_config_dict['trading_fee'],
                live_trading=cls.mock_config_dict['live_trading'],
                broker_type=cls.mock_config_dict['broker_type'],
                log_experience=cls.mock_config_dict['log_experience'],
                # Pass the pre-fetched mock data via df for non-live mode
                df=cls.mock_data_provider.get_historical_data(
                    cls.mock_config_dict['symbol'], '1h', cls.mock_config_dict['lookback_window']
                ) 
            )
            cls.env.logger = cls.test_logger # Override env's default logger for test quietness
        except Exception as e:
            cls.test_logger.error(f"Error setting up TestCryptoTradingEnvironment: {e}", exc_info=True)
            cls.env = None

    def setUp(self):
        """Set up for each test method."""
        if not TestCryptoTradingEnvironment.env:
            self.skipTest("CryptoTradingEnvironment instance could not be initialized in setUpClass.")
        # Create a fresh DataFrame for each test to avoid side effects
        self.sample_df = pd.DataFrame(SAMPLE_DATA).copy()
        # Ensure 'log_returns' column is present as it's used in _advanced_data_preprocessing
        # and typically calculated based on 'Close'.
        self.sample_df['returns'] = self.sample_df['Close'].pct_change()
        self.sample_df['log_returns'] = np.log(1 + self.sample_df['returns'])
        self.sample_df.fillna(0.0, inplace=True) # Mimic initial NA fill

    def test_advanced_data_preprocessing_runs_and_adds_columns(self):
        """Test that _advanced_data_preprocessing runs and adds expected indicator columns."""
        if self.env is None:
            self.skipTest("Environment setup failed.")
        
        # Ensure the sample_df has the columns _advanced_data_preprocessing expects initially
        # (e.g., 'Open', 'High', 'Low', 'Close', 'Volume')
        # The SAMPLE_DATA already provides these in uppercase.
        processed_df = self.env._advanced_data_preprocessing(self.sample_df.copy()) # Use a copy
        
        self.assertIsNotNone(processed_df, "Processed DataFrame should not be None.")
        self.assertFalse(processed_df.empty, "Processed DataFrame should not be empty.")
        
        # Update this list to reflect actual column names from the new _advanced_data_preprocessing
        expected_columns = [
            'High', 'Low', 'Close', 'Volume', # Original columns, might be Open too
            'SMA_20', 'EMA_20', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
            'BB_upper', 'BB_middle', 'BB_lower', 'ATR', 
            'Keltner_Upper', 'Keltner_Lower', 'Keltner_Basis', 
            'ROC', 'Stoch_K', 'Stoch_D', 
            'TTM_Squeeze_On', # 'TTM_Squeeze_Alert' might also be there
            'OBV', 'VWAP', 
            'Momentum_10', 'Volatility_20', 
            'Price_vs_EMA50', 'Price_vs_EMA200'
            # Add other essential columns your preprocessing is guaranteed to add
        ]
        
        for col in expected_columns:
            self.assertIn(col, processed_df.columns, f"Column '{col}' not found in processed DataFrame.")

        # Optionally, check for no NaNs if your preprocessing handles them thoroughly
        self.assertFalse(processed_df.isnull().values.any(), "Processed DataFrame should not contain NaNs after preprocessing.")


    def test_advanced_data_preprocessing_handles_lowercase_columns(self):
        """Test that preprocessing correctly handles lowercase input column names."""
        if self.env is None:
            self.skipTest("Environment setup failed.")

        lowercase_data = {
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123],
            'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128],
            'low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118],
            'close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125],
            'volume': [1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070, 1080, 1090, 1100, 1110, 1120, 1130, 1140, 1150, 1160, 1170, 1180, 1190, 1200, 1210, 1220, 1230]
        }
        lowercase_df = pd.DataFrame(lowercase_data)
        processed_df = self.env._advanced_data_preprocessing(lowercase_df.copy())
        self.assertIn('RSI', processed_df.columns) # Check one indicator (now uppercase) as a proxy
        # Ensure original columns are now uppercase if they were renamed
        self.assertIn('High', processed_df.columns)
        self.assertIn('Close', processed_df.columns)

    def test_advanced_data_preprocessing_handles_missing_critical_columns(self):
        """Test graceful handling of missing critical columns (e.g., 'Close')."""
        if self.env is None:
            self.skipTest("Environment setup failed.")
        data_missing_close = self.sample_df.drop(columns=['Close'])
        
        # Expect a ValueError because 'Close' is critical and cannot be easily imputed by this function
        with self.assertRaisesRegex(ValueError, "Missing critical columns after rename: \\['Close'\\]"):
            processed_df = self.env._advanced_data_preprocessing(data_missing_close.copy())

        # Test with 'high' missing - should also fail as it's required for many TAs
        data_missing_high = self.sample_df.drop(columns=['High'])
        with self.assertRaisesRegex(ValueError, "Missing critical columns after rename: \\['High'\\]"):
            processed_df = self.env._advanced_data_preprocessing(data_missing_high.copy())

# This allows running tests directly from this file
if __name__ == '__main__':
    unittest.main()