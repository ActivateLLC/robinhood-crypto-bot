# Robinhood Crypto Trading Bot: AI-Powered Algorithmic Trading

**Author:** Aaron Marcello Aldana

This project is an advanced Python-based cryptocurrency trading bot designed to leverage Artificial Intelligence, including Reinforcement Learning (RL) and multi-agent systems (CrewAI), for making informed trading decisions via the official Robinhood Crypto API.

**Disclaimer:** Cryptocurrency trading involves significant risk of financial loss. This software is provided for educational and research purposes only. It is not financial advice. Users should understand the code, test thoroughly in simulated environments, and use at their own risk. The author is not responsible for any financial losses incurred through the use of this bot.

## Features

*   **Official Robinhood API Integration:** Securely connects to the Robinhood Crypto API for account information, real-time market data, and order execution.
*   **Multiple Trading Strategies:** 
    *   **Reinforcement Learning (RL):** Utilizes an RL agent (powered by `stable-baselines3`) trained to identify trading opportunities based on market data and portfolio status.
    *   **CrewAI-Driven Strategy:** Employs a team of specialized AI agents (Market Researcher, Technical Analyst, Risk Manager, Strategy Generator managed by CrewAI) to perform comprehensive market analysis and generate trading signals.
    *   **Traditional Strategies:** Includes a framework for implementing conventional indicator-based strategies (e.g., SMA crossover) as a baseline or alternative.
*   **Advanced Market Analysis:**
    *   Combines technical indicators (e.g., SMA, EMA, RSI, MACD) with AI-driven insights.
    *   Supports sentiment analysis (requires NEWS_API_KEY) to gauge market mood.
*   **Multi-Source Data Ingestion:** Fetches market data from `yfinance` and `AltCryptoDataProvider` (interfacing with CoinGecko) for a broader market view.
*   **Configurable & Extendable:** 
    *   Easily configure trading parameters (symbols, trade amounts, API keys, strategies) via a `.env` file.
    *   Modular design allows for the addition of new strategies, data sources, or AI agents.
*   **Automated & Scheduled Trading:** Executes trading cycles automatically based on a configurable interval.
*   **Comprehensive Logging:** Detailed JSON-formatted logs for operations, AI decisions, trades, and errors, aiding in monitoring and debugging.
*   **Optional Plotting:** Generates plots for price action, indicators, and (potentially) agent performance using `matplotlib` and `mplfinance`.

## Architecture Overview

The bot's architecture is designed for modularity and sophisticated decision-making:

1.  **`RobinhoodCryptoBot` (Main Class - `crypto_bot.py`):** The central orchestrator responsible for:
    *   Configuration management.
    *   Connecting to the data provider and broker.
    *   Managing the trading cycle (data fetching, signal generation, trade execution).
    *   Portfolio tracking and state management.
    *   Logging.

2.  **Trading Strategies:**
    *   **PPO Reinforcement Learning Agent:** A pre-trained or continuously learning model (using `stable-baselines3`) that takes market observations and portfolio state as input to produce buy/sell/hold actions.
    *   **`CryptoTradingAgents` (CrewAI - `crew_agents/src/crypto_agents.py`):** When selected, this component deploys a crew of AI agents:
        *   `Market Researcher`: Analyzes market trends, news, and macroeconomic factors.
        *   `Technical Analyst`: Performs detailed technical analysis on historical data.
        *   `Risk Manager`: Assesses potential risks and suggests mitigation strategies.
        *   `Strategy Generator`: Synthesizes information from other agents to produce a final trading recommendation (buy/sell/hold).

3.  **Data Providers (`data_providers/`):** Abstracted modules for fetching market data (e.g., `YahooFinanceProvider`, `AltCryptoDataProvider`).

4.  **Broker Interface (`brokers/robinhood_broker.py`):** Handles all communication with the Robinhood API for trading operations and account management.

5.  **Configuration (`config.py`, `.env`):** Manages all settings, API keys, and operational parameters.

## Key Technologies

*   **Programming Language:** Python 3.8+
*   **AI & Machine Learning:**
    *   `CrewAI` & `LangChain`: For orchestrating multi-agent AI systems and interacting with Large Language Models (LLMs).
    *   `stable-baselines3`: For developing and training Reinforcement Learning agents.
    *   `scikit-learn`, `numpy`, `pandas`, `scipy`: Core data science libraries.
*   **Data Providers & APIs:**
    *   Official Robinhood Crypto API
    *   `yfinance` (Yahoo Finance)
    *   `python-coingecko` (via `AltCryptoDataProvider`)
*   **Data Handling:** `pandas` for data manipulation.
*   **Scheduling:** `schedule` library for timed operations.
*   **Plotting:** `matplotlib`, `mplfinance`.
*   **Environment Management:** `python-dotenv`.

## Prerequisites

*   Python 3.8 or higher.
*   A Robinhood account with Crypto Trading enabled.
*   Robinhood Crypto API Key and Private Key (obtainable from Robinhood API Credentials Portal on a desktop browser).
*   `OPENAI_API_KEY` in your `.env` file for CrewAI agents.
*   Optionally, `NEWS_API_KEY` for sentiment analysis.

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <your_repository_url>
    cd robinhood-crypto-bot
    ```
2.  **Create and Activate a Virtual Environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure Environment Variables:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file with your specific credentials and preferences:
        *   `ROBINHOOD_API_KEY`, `ROBINHOOD_PRIVATE_KEY` (ensure it's the Base64 encoded private key as per Robinhood's documentation for the trading API, often named `BASE64_PRIVATE_KEY` or similar in their examples)
        *   `OPENAI_API_KEY`
        *   `NEWS_API_KEY` (optional)
        *   `SYMBOLS_TO_TRADE` (e.g., `BTC-USD,ETH-USD`)
        *   `TRADE_AMOUNT_USD`
        *   `TRADING_STRATEGY` (e.g., `RL`, `CREWAI`, `TRADITIONAL`)
        *   `ENABLE_TRADING` (`true` or `false`)
        *   Other relevant paths and parameters as defined in `.env.example` and `config.py`.

    **Security Warning:** Your `.env` file contains sensitive API keys. **NEVER commit this file to version control.** It is included in `.gitignore` by default to help prevent accidental exposure.

## Running the Bot

Execute the main bot script:
```bash
python crypto_bot.py
```
Monitor the console output and log files (`logs/crypto_bot.log` and `logs/live_experience.log`) for activity.

To stop the bot, press `Ctrl+C`. The bot includes a signal handler for graceful shutdown.

## Testing

Tests are managed using `pytest`.
```bash
pytest -vv
```
Ensure tests are regularly updated to cover new features and maintain code quality.

## Development

*   **Strategies:** Implement new trading strategies by adding logic to `crypto_bot.py` or creating new strategy modules.
*   **AI Agents:** Modify prompts, tools, or tasks for CrewAI agents in `crew_agents/src/crypto_agents.py`.
*   **Data Sources:** Integrate new data providers by creating classes that adhere to the `BaseDataProvider` interface.

## Important Considerations

*   **API Rate Limits:** Be mindful of API rate limits for Robinhood and other third-party services.
*   **Order Types:** The bot primarily uses market orders. Be aware of the implications (slippage) versus limit orders.
*   **Error Handling:** While efforts have been made for robust error handling, continuous improvement is encouraged.
*   **Backtesting & Simulation:** Thoroughly test all strategies and configurations with `ENABLE_TRADING=false` before engaging real funds.

## Contributing

Contributions, issues, and feature requests are welcome. Please feel free to fork the repository, make changes, and submit pull requests.

## License

(Consider adding a license, e.g., MIT, Apache 2.0. If so, create a `LICENSE` file and state it here.)

This project is licensed under the [NAME OF LICENSE] - see the LICENSE file for details.
