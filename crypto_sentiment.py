import os
import requests
import json
import nltk
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

# Download necessary NLTK resources
try:
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

class CryptoSentimentAnalyzer:
    def __init__(self, symbols: List[str] = ['BTC', 'ETH'], cache_file: str = 'sentiment_cache.json'):
        """
        Multi-source cryptocurrency sentiment analyzer with SentiCrypt integration
        
        Args:
            symbols (List[str]): Cryptocurrency symbols to track
            cache_file (str): File to cache sentiment data
        """
        self.symbols = symbols
        self.sia = SentimentIntensityAnalyzer()
        self.cache_file = cache_file
        
        # SentiCrypt API endpoints
        self.senticrypt_endpoints = {
            'all_data': 'https://api.senticrypt.com/v2/all.json',
            'index': 'https://api.senticrypt.com/v2/index.json'
        }
        
        # Load or initialize symbol-specific cache (for processed latest sentiment per symbol)
        self.sentiment_cache = self._load_cache()

        # Cache for the raw 'all.json' SentiCrypt response
        self._raw_all_senticrypt_data = None
        self._raw_all_senticrypt_data_last_fetched = None
        self.senticrypt_cache_duration = timedelta(hours=1) # Cache 'all.json' for 1 hour

    def _load_cache(self) -> Dict:
        """
        Load sentiment data cache
        
        Returns:
            Dict: Cached sentiment data
        """
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Cache loading error: {e}")
        return {}
    
    def _save_cache(self):
        """
        Save sentiment data to cache
        """
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.sentiment_cache, f)
        except Exception as e:
            print(f"Cache saving error: {e}")
    
    def fetch_senticrypt_sentiment(self, symbol: str = 'BTC') -> Optional[Dict]:
        """
        Fetch cryptocurrency sentiment from SentiCrypt API, using a timed cache for the 'all_data' endpoint.
        
        Args:
            symbol (str): Cryptocurrency symbol
        
        Returns:
            Optional[Dict]: Sentiment data or None
        """
        now = datetime.now()
        needs_refresh = True

        if self._raw_all_senticrypt_data is not None and self._raw_all_senticrypt_data_last_fetched is not None:
            if now - self._raw_all_senticrypt_data_last_fetched < self.senticrypt_cache_duration:
                needs_refresh = False
        
        if needs_refresh:
            try:
                print(f"Fetching fresh SentiCrypt 'all_data' at {now.isoformat()}") # Logging for debug
                response = requests.get(self.senticrypt_endpoints['all_data'], timeout=10) # Added timeout
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                self._raw_all_senticrypt_data = response.json()
                self._raw_all_senticrypt_data_last_fetched = now
            except requests.exceptions.RequestException as e:
                print(f"SentiCrypt API error during refresh: {e}")
                # If refresh fails, and we don't have any raw data yet, return cached per-symbol data if available
                if self._raw_all_senticrypt_data is None:
                    return self.sentiment_cache.get(symbol) # Fallback to potentially older symbol-specific cache
                # Otherwise, we'll proceed with the (potentially stale) _raw_all_senticrypt_data if it exists

        # Use the (potentially newly fetched or cached) raw 'all_data' response
        data_to_process = self._raw_all_senticrypt_data
        
        if data_to_process and isinstance(data_to_process, list) and len(data_to_process) > 0:
            try:
                latest_sentiment = data_to_process[0] # Assumes data[0] is the desired latest global sentiment
                # Ensure latest_sentiment is a dictionary before accessing keys
                if isinstance(latest_sentiment, dict):
                    self.sentiment_cache[symbol] = {
                        'date': latest_sentiment.get('date'), # Use .get for safety
                        'score1': latest_sentiment.get('score1'),
                        'score2': latest_sentiment.get('score2'),
                        'score3': latest_sentiment.get('score3'),
                        'mean': latest_sentiment.get('mean')
                    }
                    self._save_cache() # This saves the symbol-specific processed cache
                    return self.sentiment_cache[symbol]
                else:
                    print(f"SentiCrypt 'all_data'[0] is not a dictionary: {latest_sentiment}")
            except KeyError as e:
                print(f"Error processing SentiCrypt data (KeyError: {e}): {latest_sentiment}")
            except Exception as e:
                print(f"Unexpected error processing SentiCrypt data: {e}")
        elif data_to_process is not None:
             print(f"SentiCrypt 'all_data' is not a non-empty list or is malformed: {type(data_to_process)}")

        # Fallback to the symbol-specific cache if processing raw data fails or raw data is empty/invalid
        return self.sentiment_cache.get(symbol)
    
    def analyze_text_sentiment(self, text: str) -> float:
        """
        Use NLTK VADER for text sentiment analysis
        
        Args:
            text (str): Input text to analyze
        
        Returns:
            float: Sentiment score between -1 and 1
        """
        if not text:
            return 0.0
        
        sentiment_scores = self.sia.polarity_scores(text)
        return sentiment_scores['compound']
    
    def aggregate_sentiment(self, symbol: str) -> Dict[str, float]:
        """
        Aggregate sentiment from multiple sources
        
        Args:
            symbol (str): Cryptocurrency symbol
        
        Returns:
            Dict[str, float]: Comprehensive sentiment scores
        """
        # Fetch SentiCrypt sentiment
        senticrypt_data = self.fetch_senticrypt_sentiment(symbol)
        
        # Combine sentiment sources
        sentiment_result = {
            'senticrypt_score1': senticrypt_data.get('score1', 0) if senticrypt_data else 0,
            'senticrypt_score2': senticrypt_data.get('score2', 0) if senticrypt_data else 0,
            'senticrypt_mean': senticrypt_data.get('mean', 0) if senticrypt_data else 0,
            'sentiment_score': senticrypt_data.get('score1', 0) if senticrypt_data else 0
        }
        
        return sentiment_result

# Example usage
if __name__ == '__main__':
    sentiment_analyzer = CryptoSentimentAnalyzer(['BTC', 'ETH'])
    btc_sentiment = sentiment_analyzer.aggregate_sentiment('BTC')
    print(json.dumps(btc_sentiment, indent=2))
