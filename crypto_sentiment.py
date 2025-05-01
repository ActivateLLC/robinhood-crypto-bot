import os
import requests
import json
import nltk
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from nltk.sentiment import SentimentIntensityAnalyzer

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
        
        # Load or initialize cache
        self.sentiment_cache = self._load_cache()
    
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
        Fetch cryptocurrency sentiment from SentiCrypt API
        
        Args:
            symbol (str): Cryptocurrency symbol
        
        Returns:
            Optional[Dict]: Sentiment data or None
        """
        try:
            response = requests.get(self.senticrypt_endpoints['all_data'])
            data = response.json()
            
            # Extract most recent sentiment data
            if data and len(data) > 0:
                latest_sentiment = data[0]
                self.sentiment_cache[symbol] = {
                    'date': latest_sentiment['date'],
                    'score1': latest_sentiment['score1'],
                    'score2': latest_sentiment['score2'],
                    'score3': latest_sentiment['score3'],
                    'mean': latest_sentiment['mean']
                }
                self._save_cache()
                return self.sentiment_cache[symbol]
        except Exception as e:
            print(f"SentiCrypt API error: {e}")
            # Fallback to cached data if available
            return self.sentiment_cache.get(symbol)
        
        return None
    
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
