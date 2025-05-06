// CryptoNeon Dashboard JavaScript
// Handles chart rendering and interactive elements

document.addEventListener('DOMContentLoaded', function() {
    // Set chart.js defaults with cyberpunk theme
    Chart.defaults.color = 'rgba(255, 255, 255, 0.7)';
    Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.1)';
    Chart.defaults.font.family = "'Rajdhani', 'Roboto Mono', sans-serif";
    
    // Initialize dashboard data
    let portfolioData = {};
    let cryptoPrices = {};
    let tradingSignals = [];
    let botStatus = {};
    
    // Fetch initial data
    fetchDashboardData();
    
    // Market Overview Chart
    let marketChart;
    let selectedMarketTimeframe = 1; // Default to 1 day
    function initMarketChart(data) {
        const marketCtx = document.getElementById('marketOverviewChart').getContext('2d');
        
        // Format data for chart
        const labels = data.map(item => {
            const date = new Date(item.timestamp);
            return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        });
        
        const values = data.map(item => item.value);
        
        // Destroy existing chart if it exists
        if (marketChart) {
            marketChart.destroy();
        }
        
        marketChart = new Chart(marketCtx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Portfolio Value',
                        data: values,
                        borderColor: '#00ff66',
                        backgroundColor: 'rgba(0, 255, 102, 0.1)',
                        borderWidth: 2,
                        tension: 0.4,
                        pointBackgroundColor: '#00ff66',
                        pointBorderColor: '#00ff66',
                        pointRadius: 3,
                        pointHoverRadius: 5,
                        fill: true
                    }
                ]
            },
            options: {
                animation: false, // Disable animation for faster updates
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            boxWidth: 12,
                            usePointStyle: true,
                            pointStyle: 'circle'
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(18, 18, 18, 0.9)',
                        titleColor: '#00ff66',
                        bodyColor: '#ffffff',
                        borderColor: '#00ff66',
                        borderWidth: 1,
                        padding: 10,
                        displayColors: true,
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += new Intl.NumberFormat('en-US', { 
                                        style: 'currency', 
                                        currency: 'USD',
                                        minimumFractionDigits: 2
                                    }).format(context.parsed.y);
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            display: true,
                            color: 'rgba(255, 255, 255, 0.05)'
                        },
                        ticks: {
                            padding: 10
                        }
                    },
                    y: {
                        grid: {
                            display: true,
                            color: 'rgba(255, 255, 255, 0.05)'
                        },
                        ticks: {
                            padding: 10,
                            callback: function(value) {
                                return '$' + value.toLocaleString();
                            }
                        }
                    }
                }
            }
        });
    }
    
    // Meme Coins Chart
    let memeCoinsChart;
    function updateMemeCoinsChart(prices) {
        const memeCoins = ['SHIB', 'PEPE', 'BONK', 'WIF', 'POPCAT', 'PENGU', 'PNUT', 'TRUMP'];
        const memeCoinsCtx = document.getElementById('memeCoinsChart').getContext('2d');
        
        // Calculate 24h change (for demo we'll use random values)
        // In a real app, you'd compare current prices with 24h ago prices
        const changes = memeCoins.map(coin => {
            // Random change between -10% and +20%
            return (Math.random() * 30 - 10).toFixed(2);
        });
        
        const backgroundColors = changes.map(change => 
            parseFloat(change) >= 0 ? 'rgba(0, 255, 102, 0.7)' : 'rgba(255, 58, 94, 0.7)'
        );
        
        const borderColors = changes.map(change => 
            parseFloat(change) >= 0 ? '#00ff66' : '#ff3a5e'
        );
        
        if (memeCoinsChart) {
            memeCoinsChart.destroy();
        }
        
        memeCoinsChart = new Chart(memeCoinsCtx, {
            type: 'bar',
            data: {
                labels: memeCoins,
                datasets: [{
                    label: '24h Change %',
                    data: changes,
                    backgroundColor: backgroundColors,
                    borderColor: borderColors,
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(18, 18, 18, 0.9)',
                        titleColor: '#00ff66',
                        bodyColor: '#ffffff',
                        borderColor: '#00ff66',
                        borderWidth: 1,
                        padding: 10,
                        callbacks: {
                            label: function(context) {
                                let value = context.parsed.y;
                                let sign = value >= 0 ? '+' : '';
                                return sign + value + '%';
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            padding: 10
                        }
                    },
                    y: {
                        grid: {
                            display: true,
                            color: 'rgba(255, 255, 255, 0.05)'
                        },
                        ticks: {
                            padding: 10,
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                }
            }
        });
    }

    // Strategy Performance Chart
    let strategyChart;
    function updateStrategyChart(strategies) {
        const strategyCtx = document.getElementById('strategyPerformanceChart').getContext('2d');
        
        if (strategyChart) {
            strategyChart.destroy();
        }
        
        // Format data for chart
        const datasets = strategies.map((strategy, index) => {
            const colors = [
                { bg: 'rgba(0, 255, 102, 0.2)', border: '#00ff66' },
                { bg: 'rgba(177, 74, 237, 0.2)', border: '#b14aed' },
                { bg: 'rgba(12, 234, 255, 0.2)', border: '#0ceaff' },
                { bg: 'rgba(255, 193, 7, 0.2)', border: '#ffc107' }
            ];
            
            return {
                label: strategy.name,
                data: [
                    strategy.win_rate,
                    strategy.profit_factor * 20, // Scale to 0-100 range
                    strategy.avg_profit * 10,    // Scale to 0-100 range
                    100 - strategy.max_drawdown, // Invert so higher is better
                    strategy.sharpe_ratio * 20,  // Scale to 0-100 range
                    Math.min(strategy.win_rate * strategy.profit_factor / 2, 100) // Risk-adjusted return
                ],
                backgroundColor: colors[index % colors.length].bg,
                borderColor: colors[index % colors.length].border,
                borderWidth: 2,
                pointBackgroundColor: colors[index % colors.length].border,
                pointBorderColor: colors[index % colors.length].border,
                pointRadius: 3,
                pointHoverRadius: 5
            };
        });
        
        strategyChart = new Chart(strategyCtx, {
            type: 'radar',
            data: {
                labels: ['Win Rate', 'Profit Factor', 'Avg Profit', 'Recovery', 'Consistency', 'Risk Adj'],
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            boxWidth: 12,
                            usePointStyle: true,
                            pointStyle: 'circle'
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(18, 18, 18, 0.9)',
                        titleColor: '#00ff66',
                        bodyColor: '#ffffff',
                        borderColor: '#00ff66',
                        borderWidth: 1,
                        padding: 10
                    }
                },
                scales: {
                    r: {
                        angleLines: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        pointLabels: {
                            color: 'rgba(255, 255, 255, 0.7)',
                            font: {
                                size: 12
                            }
                        },
                        ticks: {
                            backdropColor: 'transparent',
                            color: 'rgba(255, 255, 255, 0.5)',
                            z: 1
                        },
                        suggestedMin: 0,
                        suggestedMax: 100
                    }
                }
            }
        });
    }
    
    // Fetch all dashboard data
    async function fetchDashboardData() {
        console.log("Fetching dashboard data...");
        try {
            const [statusRes, portfolioRes, pricesRes, signalsRes] = await Promise.all([
                fetch('/api/status'),
                fetch('/api/portfolio'),
                fetch('/api/crypto/prices'),
                fetch('/api/trading/signals')
            ]);

            if (!statusRes.ok || !portfolioRes.ok || !pricesRes.ok || !signalsRes.ok) {
                throw new Error(`HTTP error! Statuses: ${statusRes.status}, ${portfolioRes.status}, ${pricesRes.status}, ${signalsRes.status}`);
            }

            botStatus = await statusRes.json();
            portfolioData = await portfolioRes.json();
            cryptoPrices = await pricesRes.json(); // Assuming this returns an object like { 'BTC-USD': price, ... }
            tradingSignals = await signalsRes.json(); // Assuming this returns { last_action: 'BUY'/'SELL'/'HOLD', timestamp: ...}

            console.log("Data fetched:", { botStatus, portfolioData, cryptoPrices, tradingSignals });

            // Update UI elements
            updateBotStatus(botStatus);
            updatePortfolioStats(portfolioData);
            // Pass prices to assets table update if needed
            updateAssetsTable(portfolioData.holdings || {}, cryptoPrices);
            updateSignalsList(tradingSignals ? [tradingSignals] : []); // Wrap single signal in array if needed by updateSignalsList

            // Update charts (if necessary based on fetched data)
            // Example: Update market chart if portfolioData contains history
            if (portfolioData.history && marketChart) {
                // Assuming history is in the correct format { timestamp: ..., value: ...}
                initMarketChart(portfolioData.history); // Re-init or update based on chart logic
            } else if (marketChart) {
                // If no history, maybe clear or show default state
                // initMarketChart([]); // Example: Clear chart
                console.log("Portfolio history not available for chart.");
            }

            // Placeholder for other chart updates (Meme, Strategy)
            // updateMemeCoinsChart(cryptoPrices); // Requires specific price data
            // updateStrategyChart(portfolioData.strategy_performance); // Requires specific strategy data

        } catch (error) {
            console.error('Failed to fetch dashboard data:', error);
            // Optionally update UI to show error state
            // Example: updateBotStatus({ process_status: 'Error', agent_running: false, message: 'Failed to load data' });
        }
    }

    // Update portfolio stats
    function updatePortfolioStats(data) {
        console.log("Updating portfolio stats with data:", data);
        const portfolioValueEl = document.querySelector('.dashboard-grid .card:nth-child(1) .stat-value');
        const todaysProfitEl = document.querySelector('.dashboard-grid .card:nth-child(2) .stat-value');
        // const activeStrategyEl = document.querySelector('.dashboard-grid .card:nth-child(3) .stat-value');
        // const strategyBadgesEl = document.querySelector('.dashboard-grid .card:nth-child(3) .stat-change');
        const portfolioChangeEl = document.querySelector('.dashboard-grid .card:nth-child(1) .stat-change');
        const tradesTodayEl = document.querySelector('.dashboard-grid .card:nth-child(2) .stat-change');

        if (portfolioValueEl && data.portfolio_value !== undefined) {
            portfolioValueEl.textContent = `$${parseFloat(data.portfolio_value).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
        } else if (portfolioValueEl) {
            portfolioValueEl.textContent = '$ -.--';
        }

        // Set Today's Profit to N/A as it's not tracked
        if (todaysProfitEl) {
            todaysProfitEl.textContent = 'N/A';
            // Clear related change info if needed
            if (tradesTodayEl) {
                tradesTodayEl.textContent = '- trades';
            }
        }

        // Update portfolio change (Placeholder - requires historical data comparison)
        if (portfolioChangeEl) {
             portfolioChangeEl.innerHTML = `<i class="fas fa-minus"></i> -.--% today`; // Default or N/A state
             portfolioChangeEl.className = 'stat-change'; // Reset class
        }

        // Update Active Strategy (Placeholder - needs data from state)
        // if (activeStrategyEl && data.active_strategy) {
        //     activeStrategyEl.textContent = data.active_strategy;
        // }
        // if (strategyBadgesEl && data.strategy_components) {
        //     strategyBadgesEl.innerHTML = data.strategy_components.map(s => `<span class="badge badge-success">${s}</span>`).join(' ');
        // }
    }

    // Update assets table
    function updateAssetsTable(holdings, prices) {
        console.log("Updating assets table with holdings:", holdings, "and prices:", prices);
        const tableBody = document.querySelector('#cryptoAssetsTable tbody');
        if (!tableBody) {
            console.error("Asset table body not found!");
            return;
        }

        tableBody.innerHTML = ''; // Clear existing rows

        if (Object.keys(holdings).length === 0) {
            tableBody.innerHTML = '<tr><td colspan="8" style="text-align: center; padding: 20px; color: var(--text-muted);">No assets held.</td></tr>';
            return;
        }

        // Get the trading symbol from config (needed if state only has base asset)
        // This should ideally come from the state or config endpoint if possible
        const tradingSymbol = 'BTC-USD'; // ** Hardcoded: Needs dynamic source **
        const baseAsset = tradingSymbol.split('-')[0]; // e.g., BTC
        const quoteAsset = tradingSymbol.split('-')[1]; // e.g., USD

        // Add row for base asset holdings (e.g., BTC)
        if (holdings[baseAsset] && holdings[baseAsset] > 0) {
            const amount = holdings[baseAsset];
            const currentPrice = prices[tradingSymbol] || 0;
            const value = amount * currentPrice;
            const row = createAssetRow(baseAsset, tradingSymbol, currentPrice, 0, 0, amount, value); // Placeholders for price change, volume
            tableBody.appendChild(row);
        }

        // Add row for quote asset holdings (e.g., USD/Capital)
        if (holdings[quoteAsset] && holdings[quoteAsset] > 0) {
             const row = createAssetRow(quoteAsset, '-', '-', 0, 0, holdings[quoteAsset], holdings[quoteAsset], false); // No price/market data for cash
             tableBody.appendChild(row);
        }
    }

    // Helper to create a table row for an asset
    function createAssetRow(assetCode, symbol, price, changePercent, volume, holdingAmount, holdingValue, showTradeButton = true) {
        const tr = document.createElement('tr');

        const formatCurrency = (value) => value.toLocaleString('en-US', { style: 'currency', currency: 'USD' });
        const formatNumber = (value) => value.toLocaleString('en-US', { maximumFractionDigits: 8 }); // For crypto amounts

        // TODO: Get actual icons and implement price change styling
        const iconUrl = `https://cryptologos.cc/logos/${assetCode.toLowerCase()}-${assetCode.toLowerCase()}-logo.png`; // Placeholder icon logic
        const priceClass = changePercent >= 0 ? 'price-up' : 'price-down';
        const changePrefix = changePercent >= 0 ? '+' : '';

        tr.innerHTML = `
            <td>
                <div style="display: flex; align-items: center;">
                    <img src="${iconUrl}" alt="${assetCode}" class="crypto-icon" onerror="this.style.display='none'">
                    <span>${assetCode} ${symbol !== '-' ? '(' + symbol + ')' : ''}</span>
                </div>
            </td>
            <td>${price !== '-' ? formatCurrency(price) : '-'}</td>
            <td class="${priceClass}">${changePercent !== 0 ? changePrefix + changePercent.toFixed(2) + '%' : '-'}</td>
            <td>${volume !== 0 ? formatCurrency(volume) : '-'}</td>
            <td><span class="badge badge-secondary">HOLD</span></td> <!-- Placeholder signal -->
            <td>${formatNumber(holdingAmount)} ${assetCode}</td>
            <td>${formatCurrency(holdingValue)}</td>
            <td>
                ${showTradeButton ? '<button class="btn btn-primary btn-sm">Trade</button>' : '-'}
            </td>
        `;
        return tr;
    }


    // Update signals list
    function updateSignalsList(signals) {
        console.log("Updating signals list with:", signals);
        const signalsListDiv = document.querySelector('.signals-list');
        if (!signalsListDiv) {
             console.error("Signals list container not found!");
            return;
        }

        signalsListDiv.innerHTML = ''; // Clear existing signals

        if (!signals || signals.length === 0 || !signals[0] || !signals[0].last_action) {
            signalsListDiv.innerHTML = '<div class="signal-item" style="text-align: center; padding: 20px; color: var(--text-muted);">No recent trading signals.</div>';
            return;
        }

        // Display only the most recent signal from the state
        const latestSignal = signals[0];
        const action = latestSignal.last_action.toUpperCase(); // BUY, SELL, HOLD
        const timestamp = latestSignal.timestamp ? new Date(latestSignal.timestamp * 1000) : new Date(); // Use current time if no timestamp

        let actionClass = 'badge-secondary'; // Default for HOLD
        let iconClass = 'fa-minus-circle'; // Default for HOLD
        let priceClass = '';

        if (action === 'BUY') {
            actionClass = 'badge-success';
            iconClass = 'fa-arrow-up';
            priceClass = 'price-up';
        } else if (action === 'SELL') {
            actionClass = 'badge-danger';
            iconClass = 'fa-arrow-down';
            priceClass = 'price-down';
        }

        // Calculate time ago
        const now = new Date();
        const diffSeconds = Math.round((now - timestamp) / 1000);
        let timeAgo = '';
        if (diffSeconds < 60) timeAgo = `${diffSeconds} sec ago`;
        else if (diffSeconds < 3600) timeAgo = `${Math.floor(diffSeconds / 60)} min ago`;
        else if (diffSeconds < 86400) timeAgo = `${Math.floor(diffSeconds / 3600)} hr ago`;
        else timeAgo = `${Math.floor(diffSeconds / 86400)} days ago`;

        // Assuming a default symbol like BTC/USD for display
        // TODO: Get the actual trading symbol dynamically
        const displaySymbol = 'BTC/USD';

        const signalElement = document.createElement('div');
        signalElement.className = 'signal-item';
        signalElement.innerHTML = `
            <div class="signal-icon ${priceClass}">
                <i class="fas ${iconClass}"></i>
            </div>
            <div class="signal-details">
                <div class="signal-coin">${displaySymbol}</div>
                <div class="signal-price">${cryptoPrices[displaySymbol] ? '$' + parseFloat(cryptoPrices[displaySymbol]).toLocaleString() : 'N/A'}</div>
            </div>
            <div class="signal-action">
                <span class="badge ${actionClass}">${action}</span>
            </div>
            <div class="signal-time">${timeAgo}</div>
        `;

        signalsListDiv.appendChild(signalElement);
    }

    // Update bot status
    function updateBotStatus(status) {
        console.log("Updating bot status:", status);
        const statusValueEl = document.querySelector('.dashboard-grid .card:nth-child(4) .stat-value');
        const statusIconEl = document.querySelector('.dashboard-grid .card:nth-child(4) .stat-change i');
        const statusTextEl = document.querySelector('.dashboard-grid .card:nth-child(4) .stat-change');
        const startBtn = document.getElementById('start-bot-btn');
        const stopBtn = document.getElementById('stop-bot-btn');
        // const statusDiv = document.getElementById('bot-status-content'); // Defined later, maybe use header status?

        let displayStatus = 'Unknown';
        let statusColor = 'var(--text-muted)'; // Grey for Unknown/Stopped
        let nextCheckText = 'Bot is stopped.';

        if (status && status.process_status) {
            if (status.process_status === 'running' && status.agent_running) {
                displayStatus = 'Running';
                statusColor = 'var(--accent-green)';
                nextCheckText = 'Actively Trading'; // Or add next check time if available
            } else if (status.process_status === 'running' && !status.agent_running) {
                displayStatus = 'Idle'; // Process running, agent not (e.g., initializing)
                statusColor = 'var(--accent-yellow)';
                nextCheckText = 'Initializing...';
            } else if (status.process_status === 'stopped') {
                displayStatus = 'Stopped';
                statusColor = 'var(--text-muted)';
                nextCheckText = 'Bot is stopped.';
            } else if (status.process_status === 'error') {
                displayStatus = 'Error';
                statusColor = 'var(--accent-red)';
                nextCheckText = status.message || 'An error occurred.';
            }
        }

        if (statusValueEl) {
            statusValueEl.textContent = displayStatus;
            statusValueEl.style.color = statusColor;
        }
        if (statusIconEl) {
            statusIconEl.className = displayStatus === 'Running' ? 'fas fa-circle' : (displayStatus === 'Error' ? 'fas fa-exclamation-triangle' : 'far fa-circle');
            statusIconEl.style.color = statusColor;
        }
         if (statusTextEl) {
            // Keep the icon, replace the text content after it
            const iconHTML = statusIconEl ? statusIconEl.outerHTML : '';
            statusTextEl.innerHTML = `${iconHTML} ${nextCheckText}`;
         }

        // Update button states
        const isRunning = (status && status.process_status === 'running' && status.agent_running);
        if (startBtn) startBtn.disabled = isRunning;
        if (stopBtn) stopBtn.disabled = !isRunning;

        // Also update the simple status text potentially used by buttons
        const simpleStatusDiv = document.getElementById('bot-status-content');
        if (simpleStatusDiv) {
            simpleStatusDiv.textContent = `Bot Status: ${displayStatus}. ${nextCheckText}`;
            simpleStatusDiv.className = isRunning ? 'status-success' : (displayStatus === 'Error' ? 'status-error' : 'status-info');
        }
    }
    
    // Add interactivity to the dashboard
    
    // --- Timeframe Selectors ---
    const marketTimeframeSelectors = document.querySelectorAll('.card:has(#marketOverviewChart) .timeframe-selector button');
    
    marketTimeframeSelectors.forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all buttons in this group
            marketTimeframeSelectors.forEach(btn => btn.classList.remove('active'));
            // Add active class to the clicked button
            this.classList.add('active');
            
            selectedMarketTimeframe = parseInt(this.getAttribute('data-days'));
            console.log(`Market timeframe changed to: ${selectedMarketTimeframe} days`);
            
            // Fetch and update the market chart
            fetchMarketHistory(selectedMarketTimeframe);
        });
    });

    // Function to fetch market history for a specific timeframe
    async function fetchMarketHistory(days) {
        try {
            const response = await fetch(`/api/portfolio/history?days=${days}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const historyData = await response.json();
            // Update the market chart
            initMarketChart(historyData);
        } catch (error) {
            console.error('Error fetching market history:', error);
            // Optionally display an error message to the user
        }
    }

    // --- Bot start/stop toggle ---
    const botToggleBtn = document.querySelector('.user-menu .btn');
    let botRunning = botStatus.running || false;
    
    botToggleBtn.addEventListener('click', function() {
        const endpoint = botRunning ? '/api/bot/stop' : '/api/bot/start';
        
        fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            console.log('Bot status changed:', data);
            botRunning = !botRunning;
            
            if (botRunning) {
                this.innerHTML = '<i class="fas fa-power-off"></i> Stop Bot';
                this.style.backgroundColor = '#ff3a5e';
                document.querySelector('.stat-value[style*="color"]').textContent = 'Running';
                document.querySelector('.stat-value[style*="color"]').style.color = 'var(--accent-green)';
                document.querySelector('.stat-change i.fa-circle').style.color = 'var(--accent-green)';
            } else {
                this.innerHTML = '<i class="fas fa-power-off"></i> Start Bot';
                this.style.backgroundColor = 'var(--accent-green)';
                document.querySelector('.stat-value[style*="color"]').textContent = 'Stopped';
                document.querySelector('.stat-value[style*="color"]').style.color = '#ff3a5e';
                document.querySelector('.stat-change i.fa-circle').style.color = '#ff3a5e';
            }
        })
        .catch(error => console.error('Error changing bot status:', error));
    });
    
    // Initialize the button state
    if (botStatus.running) {
        botToggleBtn.innerHTML = '<i class="fas fa-power-off"></i> Stop Bot';
        botToggleBtn.style.backgroundColor = '#ff3a5e';
    } else {
        botToggleBtn.innerHTML = '<i class="fas fa-power-off"></i> Start Bot';
        botToggleBtn.style.backgroundColor = 'var(--accent-green)';
    }
    
    // Search functionality for assets table
    const searchInput = document.querySelector('.search-container input');
    
    searchInput.addEventListener('input', function() {
        const searchTerm = this.value.toLowerCase();
        const tableRows = document.querySelectorAll('tbody tr');
        
        tableRows.forEach(row => {
            const assetName = row.querySelector('td:first-child').textContent.toLowerCase();
            if (assetName.includes(searchTerm)) {
                row.style.display = '';
            } else {
                row.style.display = 'none';
            }
        });
    });
    
    // Add glitch effect animation
    const glitchElements = document.querySelectorAll('.glitch');
    
    glitchElements.forEach(element => {
        element.setAttribute('data-text', element.textContent);
    });
    
    // Refresh button functionality
    const refreshBtn = document.querySelector('.card-actions .btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function() {
            fetchDashboardData();
            this.innerHTML = '<i class="fas fa-sync-alt fa-spin"></i> Refreshing...';
            setTimeout(() => {
                this.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh';
            }, 1000);
        });
    }
    
    // Set up periodic data refresh
    setInterval(fetchDashboardData, 30000); // Refresh every 30 seconds
    
    // Add trade button functionality
    document.addEventListener('click', function(e) {
        if (e.target && e.target.classList.contains('btn-primary') && e.target.classList.contains('btn-sm')) {
            const row = e.target.closest('tr');
            const asset = row.querySelector('td:first-child span').textContent;
            const price = row.querySelector('td:nth-child(2)').textContent;
            
            alert(`Trading ${asset} at ${price}\nThis would open the trading modal in the full application.`);
        }
    });

    const startBtn = document.getElementById('start-bot-btn');
    const stopBtn = document.getElementById('stop-bot-btn');
    const headerRefreshBtn = document.getElementById('refresh-btn');
    const statusDiv = document.getElementById('bot-status-content'); 

    if (startBtn) {
        startBtn.addEventListener('click', () => {
            console.log("Start Bot button clicked.");
            startBtn.disabled = true;
            startBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';

            fetch('/start_bot', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    console.log("Start bot response:", data);
                    if (data.status === 'success') {
                        statusDiv.textContent = 'Bot started successfully.';
                        statusDiv.className = 'status-success';
                        stopBtn.disabled = false; 
                        startBtn.innerHTML = '<i class="fas fa-power-off"></i> Start Bot'; 
                        startBtn.disabled = false; 
                    } else {
                        statusDiv.textContent = `Error starting bot: ${data.message}`;
                        statusDiv.className = 'status-error';
                        startBtn.disabled = false; 
                        startBtn.innerHTML = '<i class="fas fa-power-off"></i> Start Bot';
                    }
                })
                .catch(error => {
                    console.error('Error sending start bot request:', error);
                    statusDiv.textContent = 'Failed to communicate with server to start bot.';
                    statusDiv.className = 'status-error';
                    startBtn.disabled = false; 
                    startBtn.innerHTML = '<i class="fas fa-power-off"></i> Start Bot';
                });
        });
    }

    if (stopBtn) {
        stopBtn.addEventListener('click', () => {
            console.log("Stop Bot button clicked.");
            stopBtn.disabled = true;
            stopBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Stopping...';

            fetch('/stop_bot', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    console.log("Stop bot response:", data);
                    if (data.status === 'success') {
                        statusDiv.textContent = 'Bot stopped successfully.';
                        statusDiv.className = 'status-success';
                        startBtn.disabled = false; 
                        stopBtn.innerHTML = '<i class="fas fa-stop-circle"></i> Stop Bot'; 
                        // stopBtn.disabled = false; // Keep disabled until started again?
                    } else {
                        statusDiv.textContent = `Error stopping bot: ${data.message}`;
                        statusDiv.className = 'status-error';
                        stopBtn.disabled = false; 
                        stopBtn.innerHTML = '<i class="fas fa-stop-circle"></i> Stop Bot';
                    }
                })
                .catch(error => {
                    console.error('Error sending stop bot request:', error);
                    statusDiv.textContent = 'Failed to communicate with server to stop bot.';
                    statusDiv.className = 'status-error';
                    stopBtn.disabled = false; 
                    stopBtn.innerHTML = '<i class="fas fa-stop-circle"></i> Stop Bot';
                });
        });
    }

    if (headerRefreshBtn) { 
        headerRefreshBtn.addEventListener('click', () => { 
            console.log("Header Refresh button clicked.");
            headerRefreshBtn.disabled = true; 
            headerRefreshBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Refreshing...'; 

            fetch('/refresh_bot', { method: 'POST' }) 
                .then(response => response.json())
                .then(data => {
                    console.log("Refresh bot status response:", data);
                    if (data.status === 'success') {
                        // Update statusDiv based on data.bot_status (e.g., running, stopped, error)
                        statusDiv.textContent = `Bot Status: ${data.bot_status || 'Unknown'}`;
                        statusDiv.className = data.bot_status === 'running' ? 'status-success' : (data.bot_status === 'error' ? 'status-error' : 'status-info');
                        // Update button states based on status
                        startBtn.disabled = (data.bot_status === 'running');
                        stopBtn.disabled = (data.bot_status !== 'running');
                    } else {
                        statusDiv.textContent = `Error refreshing bot status: ${data.message}`;
                        statusDiv.className = 'status-error';
                    }
                    headerRefreshBtn.disabled = false; 
                    headerRefreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh Status'; 
                })
                .catch(error => {
                    console.error('Error sending refresh bot status request:', error);
                    statusDiv.textContent = 'Failed to communicate with server to refresh bot status.';
                    statusDiv.className = 'status-error';
                    headerRefreshBtn.disabled = false; 
                    headerRefreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh Status'; 
                });
        });
    }

    if (statusDiv) {
         statusDiv.textContent = 'Bot status unknown. Click Refresh or Start.';
         statusDiv.className = 'status-info';
    }

});
