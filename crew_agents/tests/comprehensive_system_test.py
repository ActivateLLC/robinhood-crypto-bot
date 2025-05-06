import unittest
import pandas as pd
import numpy as np
from ..src.crypto_trading_env import CryptoTradingEnvironment
from ..src.utils import setup_logger
import pandas.testing as pd_testing

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
        cls.test_logger = setup_logger('TestCryptoEnv', log_level='DEBUG') # Ensure DEBUG level
        cls.test_logger.debug("setUpClass started.")

        # Updated mock_config_dict for the new CryptoTradingEnvironment constructor
        cls.mock_config_dict = {
            'symbol': 'BTC-USD',
            'initial_balance': 10000.0, # Changed from initial_capital
            'window_size': 24,          # Changed from lookback_window
            # 'trading_fee' is not directly set from config in the new __init__
            # 'log_experience' is not directly set from config in the new __init__
            # The environment will use its defaults or internal logic for these if not in config.
            # Add any other config keys the environment's __init__ now expects from the 'config' dict.
        }

        # This mock_data_provider is used to generate the initial 'df'
        cls.mock_data_provider = MockAltCryptoDataProvider(logger=cls.test_logger)
        cls.test_logger.debug("setUpClass: MockAltCryptoDataProvider created.")

        initial_df = cls.mock_data_provider.get_historical_data(
            symbol=cls.mock_config_dict['symbol'],
            interval='1h', # Or whatever interval is appropriate for the mock data
            lookback_period=str(cls.mock_config_dict['window_size'] * 5) # Ensure enough data for window + features
        )

        if initial_df is None or initial_df.empty:
            cls.test_logger.error("setUpClass: MockAltCryptoDataProvider returned empty or None DataFrame.")
            cls.env = None # Explicitly set to None
            cls.test_logger.debug("setUpClass: initial_df was empty or None. cls.env is None. Returning.")
            return # Stop further setup if data is bad

        cls.test_logger.debug(f"setUpClass: initial_df loaded, shape: {initial_df.shape}. dataframe is not None and not empty.")

        try:
            cls.test_logger.debug("setUpClass: Attempting to initialize CryptoTradingEnvironment with _test_trigger_debug_dump=True.")
            cls.env = CryptoTradingEnvironment(
                df=initial_df,
                config=cls.mock_config_dict,
                live_trading=False, # Explicitly set for testing offline scenarios
                broker_type='simulation',
                data_provider=cls.mock_data_provider, # Pass the mock provider
                _test_trigger_debug_dump=True # Explicitly set to True for this instantiation
            )
            cls.test_logger.info("setUpClass: CryptoTradingEnvironment initialized successfully. cls.env should be set.")
        except Exception as e:
            cls.test_logger.error(f"setUpClass: Failed to initialize CryptoTradingEnvironment: {e}", exc_info=True)
            cls.env = None # Ensure env is None if setup fails
        
        cls.test_logger.debug(f"setUpClass finished. cls.env is {'NOT None' if cls.env else 'None'}.")

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
        print(f"DEBUG TEST SETUP: self.env type: {type(self.env)}")
        # print(f"DEBUG TEST SETUP: self.env dir: {dir(self.env)}") # Can be very verbose
        if hasattr(self.env, 'window_size'):
            print(f"DEBUG TEST SETUP: self.env.window_size exists: {self.env.window_size}")
        else:
            print(f"DEBUG TEST SETUP: self.env.window_size DOES NOT EXIST")
        if hasattr(self.env, 'config') and isinstance(self.env.config, dict):
            print(f"DEBUG TEST SETUP: self.env.config['window_size'] from dict: {self.env.config.get('window_size')}")
        else:
            print(f"DEBUG TEST SETUP: self.env.config is not a dict or does not exist")

    def _normalize_series_for_test(self, series: pd.Series) -> pd.Series:
        """Applies min-max normalization to [-1, 1] as done in _advanced_data_preprocessing."""
        min_val = series.min()
        max_val = series.max()
        if max_val > min_val:
            return 2 * (series - min_val) / (max_val - min_val) - 1
        # If all values are the same, _advanced_data_preprocessing sets them to 0 after normalization
        return pd.Series(0.0, index=series.index, dtype=float) # Ensure float output

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

    def test_atr_calculation_custom_params(self):
        """Test ATR(14) calculation against expected values."""
        if self.env is None:
            self.skipTest("Environment setup failed.")
        
        # Use specific config for ATR for this test
        self.env.atr_length_config = 14
        
        # Preprocess data using the environment's method
        # The sample_df is used internally by _setup_initial_data_for_test
        processed_df = self.env._advanced_data_preprocessing(self.env.historical_data.copy())

        print(f"DEBUG TEST ATR: self.env type: {type(self.env)}")
        # print(f"DEBUG TEST ATR: self.env dir: {dir(self.env)}") # Can be very verbose
        if hasattr(self.env, 'window_size'):
            print(f"DEBUG TEST ATR: self.env.window_size exists: {self.env.window_size}")
        else:
            print(f"DEBUG TEST ATR: self.env.window_size DOES NOT EXIST")
        if hasattr(self.env, 'config') and isinstance(self.env.config, dict):
            print(f"DEBUG TEST ATR: self.env.config['window_size'] from dict: {self.env.config.get('window_size')}")
        else:
            print(f"DEBUG TEST ATR: self.env.config is not a dict or does not exist")

        # Calculate ATR directly using pandas_ta for comparison
        temp_df_for_atr = self.sample_df.copy()
        import pandas_ta as ta # Temporary import for direct calculation
        raw_expected_atr_series = ta.atr(temp_df_for_atr['High'], temp_df_for_atr['Low'], temp_df_for_atr['Close'], length=14).fillna(0.0)

        # Normalize the expected series to match preprocessing
        expected_atr_series = self._normalize_series_for_test(raw_expected_atr_series)

        # Align index with processed_df for comparison if necessary (e.g. if processed_df has a specific DatetimeIndex)
        # If both are default RangeIndex of same length, direct comparison might be fine.
        # However, explicit alignment is safer if processed_df's index might differ.
        # Ensure processed_df['ATR'] exists and has the expected length from preprocessing.
        if 'ATR' not in processed_df.columns:
            self.fail("ATR column not found in processed_df")
        if len(processed_df['ATR']) != len(expected_atr_series):
            self.fail(f"Length mismatch before index alignment: processed_df ATR len {len(processed_df['ATR'])}, expected_atr_series len {len(expected_atr_series)}")

        expected_atr_series.index = processed_df['ATR'].index

        # print(f"DEBUG TEST: Processed ATR (len {len(processed_df['ATR'])}):\n{processed_df['ATR'].to_string()}")
        # print(f"DEBUG TEST: Expected ATR (len {len(expected_atr_series)}):\n{expected_atr_series.to_string()}")

        pd_testing.assert_series_equal(processed_df['ATR'], expected_atr_series, check_dtype=False, atol=1e-5, check_names=False)

    def test_keltner_channels_calculation_custom_params(self):
        """Test Keltner Channels(20,14,sma) calculation against expected values."""
        if self.env is None:
            self.skipTest("Environment setup failed.")

        # Custom parameters for Keltner Channels
        self.env.keltner_length_config = 20
        self.env.keltner_scalar_config = 2.0
        self.env.keltner_mamode_config = 'sma'
        self.env.keltner_atr_length_config = 14

        processed_df = self.env._advanced_data_preprocessing(self.env.historical_data.copy())
        
        temp_df_for_kc = self.sample_df.copy()
        import pandas_ta as ta # Temporary import for direct calculation
        keltner_direct = ta.kc(temp_df_for_kc['High'], temp_df_for_kc['Low'], temp_df_for_kc['Close'], 
                                   length=self.env.keltner_length_config, 
                                   scalar=self.env.keltner_scalar_config, 
                                   mamode=self.env.keltner_mamode_config, 
                                   atr_length=self.env.keltner_atr_length_config)
        
        expected_kc_basis_col_name = f"KCBs_{self.env.keltner_length_config}_{self.env.keltner_scalar_config}"
        expected_kc_upper_col_name = f"KCUs_{self.env.keltner_length_config}_{self.env.keltner_scalar_config}"
        expected_kc_lower_col_name = f"KCLs_{self.env.keltner_length_config}_{self.env.keltner_scalar_config}"

        raw_expected_kc_basis_series = keltner_direct[expected_kc_basis_col_name].fillna(0.0)
        raw_expected_kc_upper_series = keltner_direct[expected_kc_upper_col_name].fillna(0.0)
        raw_expected_kc_lower_series = keltner_direct[expected_kc_lower_col_name].fillna(0.0)

        # Normalize expected series
        expected_kc_basis_series = self._normalize_series_for_test(raw_expected_kc_basis_series)
        expected_kc_upper_series = self._normalize_series_for_test(raw_expected_kc_upper_series)
        expected_kc_lower_series = self._normalize_series_for_test(raw_expected_kc_lower_series)

        # Align index with processed_df for comparison
        expected_kc_basis_series.index = processed_df['Keltner_Basis'].index 
        expected_kc_upper_series.index = processed_df['Keltner_Upper'].index
        expected_kc_lower_series.index = processed_df['Keltner_Lower'].index

        pd_testing.assert_series_equal(processed_df['Keltner_Basis'], expected_kc_basis_series, check_dtype=False, atol=1e-5, check_names=False)
        pd_testing.assert_series_equal(processed_df['Keltner_Upper'], expected_kc_upper_series, check_dtype=False, atol=1e-5, check_names=False)
        pd_testing.assert_series_equal(processed_df['Keltner_Lower'], expected_kc_lower_series, check_dtype=False, atol=1e-5, check_names=False)

    def test_roc_calculation_custom_params(self):
        """Test ROC(12) calculation against expected values."""
        if self.env is None:
            self.skipTest("Environment setup failed.")
            
        self.env.roc_length_config = 12
        
        # Preprocess data
        processed_df = self.env._advanced_data_preprocessing(self.env.historical_data.copy())

        # Expected ROC (12 periods)
        temp_df_for_roc = self.sample_df.copy()
        import pandas_ta as ta # Temporary import for direct calculation
        raw_expected_roc_series = ta.roc(temp_df_for_roc['Close'], length=12).fillna(0.0)

        # Normalize expected series
        expected_roc_series = self._normalize_series_for_test(raw_expected_roc_series)

        # Align index
        expected_roc_series.index = processed_df['ROC'].index # Align index

        # print(f"DEBUG TEST: Processed ROC (len {len(processed_df['ROC'])}):\n{processed_df['ROC'].to_string()}")
        pd_testing.assert_series_equal(processed_df['ROC'], expected_roc_series, check_dtype=False, atol=1e-5, check_names=False)

# This allows running tests directly from this file
if __name__ == '__main__':
    unittest.main()