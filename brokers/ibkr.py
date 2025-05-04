import logging
from typing import Any, Dict, List, Optional
from decimal import Decimal

from ib_insync import IB, Contract, Order as IBOrder, Trade, util, OrderStatus, PortfolioItem, Ticker, AccountValue
import re
import datetime

from .base_broker import BrokerInterface, Order, Holding, MarketData

logger = logging.getLogger(__name__)

class IBKRBroker(BrokerInterface):
    """Interactive Brokers implementation using ib_insync."""

    def __init__(self, host: str = '127.0.0.1', port: int = 7497, client_id: int = 1):
        """
        Initialize the IBKRBroker.

        Args:
            host (str): Hostname for TWS or IB Gateway.
            port (int): Port for TWS (7496/7497) or IB Gateway (4001/4002).
            client_id (int): Unique client ID for the connection.
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()
        self.active_orders: Dict[str, Order] = {} # permId (str) -> Order object
        self.connection_status = 'disconnected' # Added connection status tracking
        # Configure logging for ib_insync
        # util.logToConsole(logging.INFO)

    @property
    def is_connected(self) -> bool:
        """Return True if currently connected to the broker, False otherwise."""
        return self.connection_status == 'connected' and self.ib.isConnected()

    def connect(self) -> bool:
        if self.ib.isConnected():
            logger.warning("Already connected.")
            self.connection_status = 'connected'
            return True
        try:
            self.connection_status = 'connecting'
            logger.info(f"Connecting to IBKR at {self.host}:{self.port} with client ID {self.client_id}...")
            self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=15) # Increased timeout
            if self.ib.isConnected():
                logger.info("IB connection established.")
                self.connection_status = 'connected'
                # Register event handlers
                self.ib.orderStatusEvent += self._on_order_status
                self.ib.execDetailsEvent += self._on_exec_details # Good to track executions too
                self.ib.errorEvent += self._on_error
                self.ib.disconnectedEvent += self._on_disconnected
                
                # Keep the event loop running in the background if using async features heavily
                # Or ensure ib.sleep() or ib.run() is called periodically elsewhere
                # For now, rely on calls like get_account_summary etc. to process events via ib.sleep()
                
                return True
            else:
                 logger.error("ib.connect returned without error, but not connected.")
                 self.connection_status = 'error'
                 return False
        except ConnectionRefusedError:
             logger.critical(f"Connection refused. Is TWS/Gateway running at {self.host}:{self.port} and API configured?")
             self.connection_status = 'error'
             return False
        except TimeoutError:
             logger.error(f"Connection timed out to {self.host}:{self.port}.")
             self.connection_status = 'error'
             return False
        except Exception as e:
            logger.exception(f"An unexpected error occurred during connection: {e}")
            self.connection_status = 'error'
            return False

    def disconnect(self):
        if self.ib.isConnected():
            logger.info("Disconnecting from IBKR...")
            # Unregister handlers?
            # self.ib.orderStatusEvent -= self._on_order_status ... (optional)
            self.ib.disconnect()
            self.connection_status = 'disconnected'
            logger.info("Disconnected.")
        else:
            logger.info("Already disconnected.")
            self.connection_status = 'disconnected'

    def get_account_summary(self) -> Dict[str, Any]:
        """Fetch account summary data using accountValues."""
        if not self.ib.isConnected():
            logger.warning("Not connected to IBKR. Cannot get account summary.")
            return {}

        try:
            # Ensure we wait for the account values to be populated
            self.ib.reqAccountUpdates(True, '') # Subscribe if not already
            self.ib.sleep(2) # Allow time for values to arrive initially
            account_values = self.ib.accountValues()
            self.ib.reqAccountUpdates(False, '') # Unsubscribe after getting values

            if not account_values:
                logger.warning("No account values received from IBKR.")
                # Might need to wait longer or check connection/subscription status
                # Re-attempting might be an option here, but keep it simple first
                return {}

            summary = {}
            # Extract key values. The 'account' field often specifies which account if running multi-account setups.
            # Filter for the main account or aggregate if necessary. Here, assume single account or first available.
            for av in account_values:
                # Common useful tags: NetLiquidation, TotalCashValue, AvailableFunds, BuyingPower, EquityWithLoanValue
                # The currency field is also important.
                # We'll store them as {tag}_{currency}: value
                key = f"{av.tag}_{av.currency}"
                summary[key] = av.value
                # Optionally store by account as well
                # key_account = f"{av.account}_{av.tag}_{av.currency}"
                # summary[key_account] = av.value
            
            logger.info(f"Fetched account summary with {len(summary)} values.")
            # Example: Filter for specific keys if needed
            # filtered_summary = {k: v for k, v in summary.items() if k.startswith('NetLiquidation') or k.startswith('AvailableFunds')}
            # return filtered_summary
            return summary

        except Exception as e:
            logger.exception(f"Error fetching account summary from IBKR: {e}")
            return {}

    def get_holdings(self, asset: Optional[str] = None) -> List[Holding]:
        """Fetch current portfolio holdings using ib.portfolio()."""
        if not self.ib.isConnected():
            logger.warning("Not connected to IBKR. Cannot get holdings.")
            return []

        holdings_list = []
        try:
            portfolio_items = self.ib.portfolio()
            if not portfolio_items:
                logger.info("No holdings found in the IBKR account.")
                return []

            logger.info(f"Received {len(portfolio_items)} portfolio items from IBKR.")

            for item in portfolio_items:
                try:
                    # TODO: Implement filtering based on the 'asset' parameter if needed.
                    # This would require parsing the 'asset' string and comparing it to the 'item.contract'.
                    if asset and self._contract_matches_symbol(item.contract, asset) is False:
                         continue

                    # Extract symbol representation from contract
                    # This might need refinement depending on how different asset types are represented
                    asset_symbol = self._contract_to_symbol(item.contract)
                    if not asset_symbol:
                         logger.warning(f"Could not determine symbol for contract: {item.contract}. Skipping holding.")
                         continue

                    holding = Holding(
                        asset=asset_symbol,
                        quantity=Decimal(str(item.position)), # Ensure position is Decimal
                        average_cost=Decimal(str(item.averageCost)), # Ensure averageCost is Decimal
                        market_value=Decimal(str(item.marketValue)) # Ensure marketValue is Decimal
                    )
                    holdings_list.append(holding)
                except Exception as e:
                    logger.warning(f"Could not parse portfolio item: Contract={item.contract}, Position={item.position}. Error: {e}")

            logger.info(f"Successfully parsed {len(holdings_list)} holdings.")
            return holdings_list

        except Exception as e:
            logger.exception(f"Error fetching or parsing holdings from IBKR: {e}")
            return []

    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Fetch market data (bid/ask/last) for a symbol using snapshot data."""
        if not self.ib.isConnected():
            logger.warning("Not connected to IBKR. Cannot get market data.")
            return None

        contract = self._symbol_to_contract(symbol)
        if not contract:
            logger.error(f"Could not create contract for symbol: {symbol}")
            return None
        
        # Qualify contract for uniqueness if needed, especially for stocks/options
        try:
            qual_contracts = self.ib.qualifyContracts(contract)
            if not qual_contracts:
                 logger.error(f"Could not qualify contract for symbol {symbol}")
                 return None
            contract = qual_contracts[0] # Use the first qualified contract
            logger.debug(f"Qualified contract for {symbol}: {contract}")
        except Exception as e:
             logger.exception(f"Error qualifying contract for {symbol}: {e}")
             return None

        try:
            # Request snapshot market data
            ticker = self.ib.reqMktData(contract, '', snapshot=True, regulatorySnapshot=False)
            # Wait briefly for the ticker data to populate
            self.ib.sleep(2) # Adjust sleep time if needed, might depend on connection speed

            if not ticker or ticker.contract != contract:
                 # Sometimes ticker might be stale or belong to a different contract request
                 logger.warning(f"Market data ticker not received or stale for {symbol}. Ticker: {ticker}")
                 # Try one more time after ensuring loop is running
                 self.ib.run(0.1) # Process messages
                 ticker = self.ib.ticker(contract)
                 if not ticker or ticker.contract != contract:
                      logger.error(f"Failed to get valid market data ticker for {symbol} after retry.")
                      return None

            # Check for valid price data (IB often uses -1 or NaN for missing data)
            bid = Decimal(str(ticker.bid)) if ticker.bid is not None and ticker.bid > 0 and not util.isNan(ticker.bid) else Decimal('0')
            ask = Decimal(str(ticker.ask)) if ticker.ask is not None and ticker.ask > 0 and not util.isNan(ticker.ask) else Decimal('0')
            # Use 'close' for previous day's close if 'last' is unavailable, or 'rtVolumeLast' etc.
            last = Decimal(str(ticker.last)) if ticker.last is not None and not util.isNan(ticker.last) else Decimal('0')
            if last == Decimal('0') and ticker.close is not None and not util.isNan(ticker.close):
                 last = Decimal(str(ticker.close)) # Fallback to close price

            logger.info(f"Market data for {symbol}: Bid={bid}, Ask={ask}, Last={last}")
            
            # Cancel market data subscription if snapshot=False was used (not needed here)
            # self.ib.cancelMktData(contract)

            return MarketData(
                symbol=symbol, # Return the original requested symbol
                bid=bid,
                ask=ask,
                last_price=last
            )

        except Exception as e:
            logger.exception(f"Error fetching market data for {symbol} from IBKR: {e}")
            # Attempt to cancel data request in case of error to prevent leaks
            try:
                self.ib.cancelMktData(contract)
            except Exception as cancel_e:
                logger.warning(f"Error trying to cancel market data for {symbol} after failure: {cancel_e}")
            return None

    def place_order(self, symbol: str, side: str, order_type: str, quantity: Decimal, price: Optional[Decimal] = None, time_in_force: str = 'GTC') -> Optional[Order]:
        """Place an order with Interactive Brokers."""
        if not self.ib.isConnected():
            logger.warning("Not connected to IBKR. Cannot place order.")
            return None

        contract = self._symbol_to_contract(symbol)
        if not contract:
            logger.error(f"Could not create contract for symbol: {symbol}")
            return None
        
        # Qualify contract
        try:
            qual_contracts = self.ib.qualifyContracts(contract)
            if not qual_contracts:
                 logger.error(f"Could not qualify contract for place_order: {symbol}")
                 return None
            contract = qual_contracts[0]
        except Exception as e:
             logger.exception(f"Error qualifying contract for place_order {symbol}: {e}")
             return None

        # Create IB Order object
        ib_order = IBOrder()
        ib_order.action = side.upper() # 'BUY' or 'SELL'
        ib_order.totalQuantity = float(quantity) # IB uses float for quantity
        ib_order.orderType = order_type.upper() # 'MKT', 'LMT', etc.
        ib_order.tif = time_in_force.upper() # 'GTC', 'DAY', 'IOC', etc.
        ib_order.transmit = True # Transmit the order immediately

        if ib_order.orderType == 'LMT':
            if price is None:
                logger.error(f"Limit price must be provided for LMT order (Symbol: {symbol})")
                raise ValueError("Price required for limit orders")
            ib_order.lmtPrice = float(price)
        elif ib_order.orderType == 'STP': # Example: Stop Loss
             if price is None:
                  logger.error(f"Stop price must be provided for STP order (Symbol: {symbol})")
                  raise ValueError("Stop price required for stop orders")
             ib_order.auxPrice = float(price)
        # Add other order types (STP LMT, TRAIL, etc.) as needed

        try:
            trade = self.ib.placeOrder(contract, ib_order)
            logger.info(f"Placed order for {symbol}. IB Trade object: {trade}")
            
            # Attempt to fetch and return the full Order object immediately
            # Use permId from the submitted order
            if not trade or not trade.order or not trade.orderStatus:
                logger.warning(f"Trade object, order, or orderStatus is missing after placing order for {symbol}. Cannot confirm status. Trade: {trade}")
                return None
            
            # Now we know trade, order, and orderStatus exist
            perm_id = trade.orderStatus.permId
            if perm_id and perm_id != 0: # 0 is sometimes returned initially before permId is assigned
                 logger.info(f"Order placed for {symbol}, IBKR PermId: {perm_id}")
                 # Store initial mapping (status might update later via events)
                 # self.ibkr_order_id_to_perm_id[str(trade.order.orderId)] = str(perm_id)
                 # self.perm_id_to_ibkr_order_id[str(perm_id)] = str(trade.order.orderId)
                 
                 # Brief pause to allow IB systems to register the order fully
                 self.ib.sleep(1) 
                 order = self.get_order_status(str(perm_id)) # This will now use the cache primarily
                 if order:
                      self.active_orders[order_id] = order # Ensure it's in cache
                 return order # Return the Order dataclass instance
            else:
                 logger.error(f"Placed order for {symbol} but did not receive a valid permId (received: {perm_id}) in the trade object.")
                 # We might still get the permId later via an event, but can't confirm status now
                 return None # Cannot fetch status without permId
             
        except Exception as e:
            logger.exception(f"Error placing order for {symbol} with IBKR: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by its permId."""
        if not self.ib.isConnected():
            logger.warning("Not connected to IBKR. Cannot cancel order.")
            return False
        
        try:
            perm_id = int(order_id)
        except ValueError:
            logger.error(f"Invalid order_id format for cancellation: '{order_id}'. Must be convertible to an integer permId.")
            return False

        try:
            # Find the order in open orders
            target_order: Optional[IBOrder] = None
            open_orders = self.ib.openOrders() # Fetch current open orders
            for order in open_orders:
                if order.permId == perm_id:
                    # Check if the order is actually cancellable (e.g., not already filled/cancelled)
                    if order.orderStatus.status in [OrderStatus.Submitted, OrderStatus.PendingSubmit, OrderStatus.PreSubmitted]:
                         target_order = order
                         break
                    else:
                         logger.warning(f"Order {perm_id} found but is not in a cancellable state ({order.orderStatus.status}).")
                         return False # Order exists but cannot be cancelled
            
            if not target_order:
                logger.warning(f"Open order with permId {perm_id} not found for cancellation.")
                # It might have already been filled or cancelled.
                # We could check ib.trades() again, but returning False seems appropriate if it's not open.
                return False

            # Cancel the found order
            logger.info(f"Attempting to cancel order with permId {perm_id}.")
            trade = self.ib.cancelOrder(target_order)
            
            # ib.cancelOrder returns the Trade object. Cancellation needs confirmation via orderStatus updates.
            # We'll poll briefly or rely on event updates in a more complex setup.
            self.ib.sleep(1) # Give time for status update
            
            # Re-check status (optional but good practice)
            updated_trade = next((t for t in self.ib.trades() if t.order.permId == perm_id), None)
            if updated_trade and updated_trade.orderStatus.status == OrderStatus.Cancelled:
                 logger.info(f"Order {perm_id} cancellation confirmed.")
                 return True
            elif updated_trade:
                 logger.warning(f"Cancellation requested for {perm_id}, but current status is {updated_trade.orderStatus.status}.")
                 return False # Cancellation likely failed or status not yet updated
            else:
                 # If trade is no longer found, it might mean cancellation was extremely fast or there's an issue.
                 # Check open orders again.
                 if not any(o.permId == perm_id for o in self.ib.openOrders()):
                      logger.info(f"Order {perm_id} no longer found in open orders after cancellation attempt, assuming success.")
                      return True
                 else:
                      logger.warning(f"Order {perm_id} cancellation status unclear after attempt.")
                      return False

        except Exception as e:
            logger.exception(f"Error cancelling order permId {perm_id} with IBKR: {e}")
            return False

    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get the status of a specific order by its permId. Checks cache first.
        Falls back to querying API if not in active cache.
        """
        # Check internal cache first
        if order_id in self.active_orders:
             cached_order = self.active_orders[order_id]
             logger.debug(f"Returning cached order status for {order_id}: {cached_order.status}")
             # Optional: Check if status is terminal, if so, maybe still query API once?
             return cached_order

        logger.debug(f"Order {order_id} not in active cache, querying API.")
        if not self.ib.isConnected():
            logger.warning("Not connected to IBKR. Cannot get order status.")
            return None
        
        try:
            perm_id = int(order_id) # Convert interface string ID to IB permId int
        except ValueError:
            logger.error(f"Invalid order_id format: '{order_id}'. Must be convertible to an integer permId.")
            return None

        try:
            # Fetch recent trades and filter by permId
            # Note: This might not find very old or inactive orders efficiently.
            # For real-time status of active orders, event handlers are better.
            self.ib.reqAllOpenOrders() # Ensure open orders are up-to-date
            self.ib.sleep(1)
            all_trades = self.ib.trades() 

            target_trade: Optional[Any] = None # ib_insync.Trade not easily importable for type hint
            for trade in all_trades:
                if trade.order.permId == perm_id:
                    target_trade = trade
                    break
            
            # If not found in recent trades, check open orders specifically
            if not target_trade:
                 open_orders = self.ib.openOrders()
                 for order in open_orders:
                      if order.permId == perm_id:
                           # Found in open orders, need to construct a partial 'Trade' like object or fetch details
                           # Simpler: Use reqOrders() if available, or return partial info
                           logger.info(f"Order {perm_id} found in open orders. Fetching full details might be needed.")
                           # Placeholder: return partial/converted open order - needs _ib_order_to_order adaptation
                           # return self._ib_order_to_order(???) # Need to adapt _ib_order_to_order
                           logger.warning(f"Order {perm_id} found in open orders, but returning full status from this state is not fully implemented.")
                           # Fall through to potentially find it in filled orders or return None
                           pass # Let's rely on ib.trades() for now which covers filled/active

            if target_trade:
                logger.info(f"Found trade details for order permId {perm_id}.")
                order = self._ib_order_to_order(target_trade) # Use helper to convert
                if order:
                     self.active_orders[order_id] = order # Update cache
                return order
            else:
                logger.warning(f"Order with permId {perm_id} not found in recent trades or open orders.")
                return None

        except Exception as e:
            logger.exception(f"Error fetching order status for permId {perm_id} from IBKR: {e}")
            return None

    # --- Helper methods specific to IBKR --- 

    def _contract_matches_symbol(self, contract: Contract, symbol: str) -> bool:
         """Check if an IBKR contract roughly matches a given symbol string."""
         # This is a simplified check and might need complex logic for options/futures
         target_contract = self._symbol_to_contract(symbol)
         if not target_contract:
              return False # Cannot create a comparable contract from the symbol
         
         # Compare key fields (add more as needed, e.g., expiry, strike for derivatives)
         return (
              contract.symbol == target_contract.symbol and
              contract.secType == target_contract.secType and
              contract.currency == target_contract.currency # Crucial differentiator
              # Add exchange, expiry, strike, right checks for derivatives
         )

    def _contract_to_symbol(self, contract: Contract) -> Optional[str]:
         """Convert an IBKR contract back into a string symbol representation."""
         # This needs to be robust enough to handle different secTypes
         if contract.secType == 'STK':
              return contract.symbol
         elif contract.secType == 'CRYPTO':
              # Reconstruct like 'BTC-USD' if possible
              return f"{contract.symbol}-{contract.currency}" # Assuming convention
         elif contract.secType == 'FUT':
              # Example: 'ESZ3' or 'MNQH4', CLV3
              return f"{contract.symbol} {contract.lastTradeDateOrContractMonth}" # Needs formatting
         elif contract.secType == 'OPT':
              # Example: 'AAPL 230915C00180000'
              date = contract.lastTradeDateOrContractMonth
              strike = str(int(contract.strike * 1000)).zfill(8) # OCC format
              right = contract.right
              return f"{contract.symbol.ljust(6)}{date[2:]}{right}{strike}" # OCC Symbol
         else:
              logger.warning(f"Cannot convert contract type '{contract.secType}' to symbol string: {contract}")
              return None # Fallback or raise error

    def _symbol_to_contract(self, symbol: str) -> Optional[Contract]:
        """
        Convert a string symbol into an IBKR Contract object.
        Handles Stocks, Options (OCC), Futures, and Crypto.
        """
        if not symbol:
            logger.warning("Empty symbol provided to _symbol_to_contract.")
            return None

        symbol_upper_stripped = symbol.upper().strip()
        contract = None

        # --- New Order of Checks --- 

        # 1. Future (Root + Month + Optional Space + Year pattern)
        # Regex allows optional space between month code and year digits
        fut_pattern = r'^([A-Z]{2,4})([FGHJKMNQUVXZ])\s?(\d{1,2})$'
        fut_match = re.fullmatch(fut_pattern, symbol_upper_stripped)
        if fut_match:
            logger.debug(f"_symbol_to_contract: Future pattern matched for '{symbol_upper_stripped}'. Match object: {fut_match}")
            # Pass the symbol *without* the space to the helper
            root, month_code, year_digits = fut_match.groups()
            symbol_for_helper = f"{root}{month_code}{year_digits}"
            logger.debug(f"_symbol_to_contract: Calling _parse_future_symbol with '{symbol_for_helper}'")
            contract = self._parse_future_symbol(symbol_for_helper)
            logger.debug(f"_symbol_to_contract: _parse_future_symbol returned: {contract}")
            if contract: return contract
        else:
            logger.debug(f"_symbol_to_contract: Future pattern did NOT match for '{symbol_upper_stripped}'")

        # 2. Option (Check only if it wasn't identified as a future)
        # OCC format requires a space
        if ' ' in symbol_upper_stripped:
             contract = self._parse_option_symbol(symbol_upper_stripped)
             if contract: return contract

        # 3. Crypto (Base-Quote format)
        if '-' in symbol_upper_stripped:
            contract = self._parse_crypto_symbol(symbol_upper_stripped)
            if contract: return contract

        # 4. Stock (Default/Fallback)
        contract = self._parse_stock_symbol(symbol_upper_stripped)
        if contract: return contract

        # --- End New Order --- 

        # If no pattern matches after trying all types
        logger.warning(f"Could not determine contract type for symbol: {symbol}")
        return None

    def _parse_stock_symbol(self, symbol_upper: str) -> Optional[Contract]:
        """Parse a standard stock symbol."""
        logger.debug(f"Attempting to parse as STK: {symbol_upper}")
        # Basic check: 1-5 uppercase letters usually works for US stocks
        if re.fullmatch(r'^[A-Z]{1,5}$', symbol_upper):
             # TODO: Add logic to handle potential ambiguities (e.g., warrants, preferred) if needed
             # TODO: Allow specifying exchange? Defaulting to SMART for now.
             return Contract(symbol=symbol_upper, secType='STK', currency='USD', exchange='SMART')
        return None

    def _parse_option_symbol(self, symbol_upper: str) -> Optional[Contract]:
        """Parse an OCC option symbol (e.g., 'AAPL 230915C00180000')."""
        logger.debug(f"Attempting to parse as OPT: {symbol_upper}")
        # Regex for OCC format: Underlying SPACE YYMMDD C/P Strike(8 digits)
        occ_match = re.fullmatch(r'^([A-Z]{1,5})\s+(\d{6})([CP])(\d{8})$', symbol_upper)
        if occ_match:
            underlying, expiry_str, right, strike_str = occ_match.groups()
            expiry = f"20{expiry_str}" # Assumes 21st century
            strike = float(strike_str) / 1000.0
            logger.debug(f"Parsed OCC Option: U={underlying}, Exp={expiry}, R={right}, K={strike}")
            # TODO: Add exchange logic if needed (SMART is often okay)
            # TODO: Consider multiplier for non-standard options?
            return Contract(
                symbol=underlying,
                secType='OPT',
                lastTradeDateOrContractMonth=expiry,
                strike=strike,
                right=right,
                exchange='SMART', # Or use a specific options exchange like CBOE
                currency='USD'
            )
        return None

    def _parse_future_symbol(self, symbol_upper: str) -> Optional[Contract]:
        """Parse a future symbol (e.g., 'ESZ3', 'CLG4', 'NGH24')."""
        logger.debug(f"Attempting to parse as FUT: {symbol_upper}")
        # Common format: Root (2-4 chars) + Month Code + Year Digit(s)
        # Month codes: FGHJKMNQUVXZ
        # Year: Z3 -> Dec 2023, H4 -> Mar 2024, H24 -> Mar 2024
        fut_match = re.fullmatch(r'^([A-Z]{2,4})([FGHJKMNQUVXZ])(\d{1,2})$', symbol_upper)
        if fut_match:
             root, month_code, year_digits = fut_match.groups()
             year_suffix = int(year_digits)
             
             # Determine full year (handle 1 vs 2 digit year - simple approach for now)
             current_year = datetime.datetime.now().year
             if len(year_digits) == 1:
                  # Calculate year assuming the current decade
                  base_year = (current_year // 10) * 10 + year_suffix
                  # If the calculated year is > 2 years in the past, assume it refers to the next decade
                  # (Adjust the threshold '2' if needed based on how far back contracts might be listed)
                  if current_year - base_year > 2:
                       year = base_year + 10
                  else:
                       year = base_year
             else: # Assume 2 digits like '24' means 2024
                  year = 2000 + year_suffix # Standard YY -> YYYY

             month_map = {'F': '01', 'G': '02', 'H': '03', 'J': '04', 'K': '05', 'M': '06',
                          'N': '07', 'Q': '08', 'U': '09', 'V': '10', 'X': '11', 'Z': '12'}
             month = month_map[month_code]
             contract_month = f"{year}{month}" # YYYYMM format

             logger.debug(f"Parsed Future: Root={root}, Month={contract_month}")
             # TODO: Need to map root symbol (e.g., ES, NQ, ZB, CL) to correct exchange and currency
             # This often requires a lookup table or more complex logic. Defaulting for now.
             exchange = 'CME' # Common default, but varies! (NYMEX, CBOT etc)
             currency = 'USD'
             if root in ['ES', 'NQ', 'RTY', 'YM', 'ZB', 'ZN', 'ZF', 'ZT', 'GE']:
                 exchange = 'CME' # Or GLOBEX
             elif root in ['CL', 'NG', 'HO', 'RB']:
                 exchange = 'NYMEX'
             elif root in ['ZC', 'ZS', 'ZW', 'KC', 'CT']:
                 exchange = 'CBOT' # Or ECBOT
             
             return Contract(
                 symbol=root,
                 secType='FUT',
                 lastTradeDateOrContractMonth=contract_month,
                 exchange=exchange,
                 currency=currency
             )
        return None

    def _parse_crypto_symbol(self, symbol_upper: str) -> Optional[Contract]:
        """Parse a crypto symbol (e.g., 'BTC-USD', 'ETH-EUR'). Needs PAXOS or ZEROCAP exchange."""
        logger.debug(f"Attempting to parse as CRYPTO: {symbol_upper}")
        crypto_match = re.fullmatch(r'^([A-Z]{2,5})-([A-Z]{3})$', symbol_upper)
        if crypto_match:
            base_asset, quote_currency = crypto_match.groups()
            logger.debug(f"Parsed Crypto: Base={base_asset}, Quote={quote_currency}")
            # IBKR uses specific exchanges for crypto, e.g., PAXOS for USD pairs
            exchange = 'PAXOS' # Default for USD pairs, might need adjustment for others (e.g., ZEROCAP for EUR?)
            if quote_currency != 'USD':
                 logger.warning(f"Non-USD crypto quote currency {quote_currency}, exchange might need adjustment from PAXOS.")
            
            # Note: IBKR often uses symbol like 'BTC' and currency 'USD'
            return Contract(symbol=base_asset, secType='CRYPTO', currency=quote_currency, exchange=exchange)
        return None

    # --- Event Handlers --- 

    def _on_order_status(self, trade: Trade):
        """Handle order status updates from IBKR."""
        # 'trade' here is an ib_insync.Trade object containing order, status, logs etc.
        if not trade or not hasattr(trade, 'orderStatus') or not hasattr(trade, 'order'):
            logger.warning(f"Received incomplete order status update: {trade}")
            return

        perm_id = trade.order.permId
        status = trade.orderStatus.status
        filled = trade.orderStatus.filled
        remaining = trade.orderStatus.remaining
        avg_fill_price = trade.orderStatus.avgFillPrice

        logger.info(f"Order Status Update - PermId: {perm_id}, Status: {status}, Filled: {filled}, Remaining: {remaining}, AvgFillPx: {avg_fill_price}")

        order = self._ib_order_to_order(trade)
        if order:
             order_id_str = str(order.id) # permId is the ID
             self.active_orders[order_id_str] = order
             logger.debug(f"Updated active_orders cache for {order_id_str}. New status: {order.status}")
             
             # Optional: Remove from active cache if terminal state
             if order.status in ['filled', 'cancelled', 'rejected', 'inactive']:
                  logger.info(f"Order {order_id_str} reached terminal state {order.status}. Removing from active cache.")
                  # del self.active_orders[order_id_str] # Or move to a completed_orders dict
        else:
             logger.warning(f"Could not convert order status update to Order object for PermId: {perm_id}")

    def _on_exec_details(self, trade: Trade, fill: Any): # fill is ib_insync.Fill
         """Handle execution details updates from IBKR."""
         if not trade or not trade.order or not fill or not fill.execution:
              logger.warning(f"Received incomplete execution detail: Trade={trade}, Fill={fill}")
              return

         perm_id = trade.order.permId
         order_id_str = str(perm_id)

         exec_id = fill.execution.execId
         exec_time_str = fill.time # ib_insync uses datetime
         # Safely convert exec_time_str to datetime if needed, though ib_insync might provide it
         exec_time = exec_time_str if isinstance(exec_time_str, datetime.datetime) else None
         
         # Note: fill.execution.shares contains the quantity of THIS specific fill
         # The trade.orderStatus contains the CUMULATIVE filled quantity and avg price
         # It's usually better to rely on the orderStatus update for cumulative state
         # But we can log the individual execution details here.
         exec_shares = fill.execution.shares
         exec_price = fill.execution.price
         cum_qty = fill.execution.cumQty # Cumulative quantity after this fill
         avg_price = fill.execution.avgPrice # Average price after this fill

         logger.info(f"Execution Detail - PermId: {perm_id}, ExecId: {exec_id}, Time: {exec_time}, Qty: {exec_shares}, Price: {exec_price}, CumQty: {cum_qty}, AvgPx: {avg_price}")

         # Update the cached order
         if order_id_str in self.active_orders:
              cached_order = self.active_orders[order_id_str]
              # Update based on the cumulative values from the execution message
              # These might be more timely than waiting for the next orderStatusEvent
              cached_order.filled_quantity = cum_qty 
              cached_order.average_fill_price = avg_price
              cached_order.remaining_quantity = trade.order.totalQuantity - cum_qty
              cached_order.updated_at = exec_time or datetime.datetime.now(datetime.timezone.utc)
              # Commission might be in fill.commissionReport - requires parsing
              if fill.commissionReport:
                   cached_order.commission = fill.commissionReport.commission

              logger.debug(f"Updated cached order {order_id_str} from execution details: Filled={cached_order.filled_quantity}, AvgPx={cached_order.average_fill_price}")
              # Note: The status itself is usually updated by _on_order_status
         else:
              logger.warning(f"Received execution detail for order {order_id_str} not found in active cache.")
         # TODO: Potentially reconcile fills if needed, e.g., if multiple fills come rapidly

    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract: Optional[Contract] = None):
        """Handle errors reported by IBKR TWS/Gateway."""
        # reqId: -1 means notification, otherwise it's the ID of the request that generated the error
        # Error Codes: https://interactivebrokers.github.io/tws-api/message_codes.html
        log_level = logging.WARNING # Default
        if errorCode in [2104, 2106, 2158]: # Market data connection messages (usually benign)
             log_level = logging.INFO
        elif errorCode == 2100: # "Connectivity between IB and Trader Workstation has been lost."
             log_level = logging.ERROR
        elif errorCode == 1100: # "Connectivity between IB and the TWS has been lost."
             log_level = logging.ERROR
        elif errorCode == 1102: # "Connectivity between IB and the TWS has been restored - data maintained."
             log_level = logging.INFO 
        elif errorCode == 202: # Order cancelled
             log_level = logging.INFO # Often expected
        elif errorCode in [502, 504]: # Couldn't connect to TWS, TWS not running?
             log_level = logging.CRITICAL
             self.connection_status = 'error' # Connection failed critically
         
        logger.log(log_level, f"IBKR Error - ReqId: {reqId}, Code: {errorCode}, Msg: {errorString}, Contract: {contract}")

        # If it's a connectivity loss error, update status
        if errorCode in [1100, 2100]:
             self.connection_status = 'disconnected'
        elif errorCode == 1102:
             self.connection_status = 'connected' # Connection restored
        # TODO: Implement specific actions based on error codes (e.g., reconnect logic for connectivity issues)

    def _on_disconnected(self):
        """Handle disconnection events."""
        logger.error("Disconnected from IBKR TWS/Gateway.")
        self.connection_status = 'disconnected'
        # TODO: Implement reconnection logic or notify application
        # Note: ib_insync might handle some reconnection automatically depending on setup
        