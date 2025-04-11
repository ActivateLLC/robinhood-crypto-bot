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
    function fetchDashboardData() {
        // Fetch portfolio data
        fetch('/api/portfolio')
            .then(response => response.json())
            .then(data => {
                portfolioData = data;
                updatePortfolioStats(data);
                updateAssetsTable(data.holdings);
            })
            .catch(error => console.error('Error fetching portfolio data:', error));
        
        // Fetch crypto prices
        fetch('/api/crypto/prices')
            .then(response => response.json())
            .then(data => {
                cryptoPrices = data;
            })
            .catch(error => console.error('Error fetching crypto prices:', error));
        
        // Fetch trading signals
        fetch('/api/trading/signals')
            .then(response => response.json())
            .then(data => {
                tradingSignals = data;
                updateSignalsList(data);
            })
            .catch(error => console.error('Error fetching trading signals:', error));
        
        // Fetch bot status
        fetch('/api/status') // Corrected endpoint
            .then(response => response.json())
            .then(data => {
                botStatus = data;
                updateBotStatus(data);
            })
            .catch(error => console.error('Error fetching bot status:', error));
        
        // Fetch portfolio history
        fetch('/api/portfolio/history')
            .then(response => response.json())
            .then(data => {
                if (data.length > 0) {
                    if (!marketChart) {
                        initMarketChart(data);
                    } else {
                        // Update existing chart
                        const labels = data.map(item => {
                            const date = new Date(item.timestamp);
                            return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                        });
                        
                        const values = data.map(item => item.value);
                        
                        marketChart.data.labels = labels;
                        marketChart.data.datasets[0].data = values;
                        marketChart.update();
                    }
                }
            })
            .catch(error => console.error('Error fetching portfolio history:', error));
        
        // Fetch strategy performance
        fetch('/api/strategy/performance')
            .then(response => response.json())
            .then(data => {
                updateStrategyChart(data.strategies);
            })
            .catch(error => console.error('Error fetching strategy performance:', error));
        
        // Update meme coins chart
        updateMemeCoinsChart(cryptoPrices);
    }
    
    // Update portfolio stats
    function updatePortfolioStats(data) {
        const portfolioValue = document.querySelector('.stat-value');
        if (portfolioValue) {
            portfolioValue.textContent = '$' + data.total_value.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
        }
        
        // Calculate daily change (for demo we'll use a random value)
        // In a real app, you'd compare with yesterday's closing value
        const changePercent = (Math.random() * 8 - 4).toFixed(2);
        const changeValue = data.total_value * (parseFloat(changePercent) / 100);
        
        const priceChange = document.querySelector('.stat-change');
        if (priceChange) {
            if (changePercent >= 0) {
                priceChange.className = 'stat-change price-up';
                priceChange.innerHTML = '<i class="fas fa-arrow-up"></i> ' + changePercent + '% today';
            } else {
                priceChange.className = 'stat-change price-down';
                priceChange.innerHTML = '<i class="fas fa-arrow-down"></i> ' + Math.abs(changePercent) + '% today';
            }
        }
        
        // Update today's profit
        const todayProfit = document.querySelectorAll('.stat-value')[1];
        if (todayProfit) {
            todayProfit.textContent = '$' + Math.abs(changeValue).toFixed(2);
        }
    }
    
    // Update assets table
    function updateAssetsTable(holdings) {
        const tableBody = document.querySelector('tbody');
        if (!tableBody) return;
        
        // Clear existing rows
        tableBody.innerHTML = '';
        
        // Add rows for each holding
        holdings.forEach(holding => {
            const row = document.createElement('tr');
            
            // Calculate 24h change (random for demo)
            const change = (Math.random() * 16 - 8).toFixed(2);
            const changeClass = parseFloat(change) >= 0 ? 'price-up' : 'price-down';
            const changePrefix = parseFloat(change) >= 0 ? '+' : '';
            
            // Get logo URL (in a real app, you'd have a mapping of symbols to logo URLs)
            const logoUrl = `https://cryptologos.cc/logos/${holding.symbol.toLowerCase()}-${holding.symbol.toLowerCase()}-logo.png`;
            
            // Format market cap (random for demo)
            const marketCap = (Math.random() * 500 + 100).toFixed(2);
            
            // Format signal (random for demo)
            const signals = ['BUY', 'SELL', 'HOLD'];
            const signal = signals[Math.floor(Math.random() * signals.length)];
            const signalClass = signal === 'BUY' ? 'badge-success' : (signal === 'SELL' ? 'badge-danger' : 'badge-warning');
            
            row.innerHTML = `
                <td>
                    <div style="display: flex; align-items: center;">
                        <img src="${logoUrl}" alt="${holding.symbol}" class="crypto-icon" onerror="this.src='https://via.placeholder.com/32'">
                        <span>${holding.symbol}</span>
                    </div>
                </td>
                <td>$${holding.price.toFixed(2)}</td>
                <td class="${changeClass}">${changePrefix}${change}%</td>
                <td>$${marketCap}B</td>
                <td><span class="badge ${signalClass}">${signal}</span></td>
                <td>${holding.quantity.toFixed(6)}</td>
                <td>$${holding.value.toFixed(2)}</td>
                <td>
                    <button class="btn btn-primary btn-sm">Trade</button>
                </td>
            `;
            
            tableBody.appendChild(row);
        });
    }
    
    // Update signals list
    function updateSignalsList(signals) {
        const signalsList = document.querySelector('.signals-list');
        if (!signalsList || signals.length === 0) return;
        
        // Clear existing signals
        signalsList.innerHTML = '';
        
        // Add new signals (limit to 4)
        signals.slice(0, 4).forEach(signal => {
            const signalItem = document.createElement('div');
            signalItem.className = 'signal-item';
            
            // Format timestamp
            const timestamp = new Date(signal.timestamp);
            const now = new Date();
            const diffMs = now - timestamp;
            const diffMins = Math.round(diffMs / 60000);
            const timeAgo = diffMins < 60 
                ? `${diffMins} min ago` 
                : `${Math.round(diffMins / 60)} hr ago`;
            
            signalItem.innerHTML = `
                <div class="signal-icon ${signal.action === 'BUY' ? 'price-up' : 'price-down'}">
                    <i class="fas fa-arrow-${signal.action === 'BUY' ? 'up' : 'down'}"></i>
                </div>
                <div class="signal-details">
                    <div class="signal-coin">${signal.symbol}</div>
                    <div class="signal-strategy">${signal.strategy}</div>
                </div>
                <div class="signal-action">
                    <span class="badge ${signal.action === 'BUY' ? 'badge-success' : 'badge-danger'}">${signal.action}</span>
                </div>
                <div class="signal-time">${timeAgo}</div>
            `;
            
            signalsList.appendChild(signalItem);
        });
    }
    
    // Update bot status
    function updateBotStatus(status) {
        const botStatusValue = document.querySelectorAll('.stat-value')[3];
        const botStatusIndicator = document.querySelector('.stat-change i.fa-circle');
        const nextCheckText = document.querySelectorAll('.stat-change')[3];
        
        if (botStatusValue) {
            botStatusValue.textContent = status.running ? 'Running' : 'Stopped';
            botStatusValue.style.color = status.running ? 'var(--accent-green)' : '#ff3a5e';
        }
        
        if (botStatusIndicator) {
            botStatusIndicator.style.color = status.running ? 'var(--accent-green)' : '#ff3a5e';
        }
        
        if (nextCheckText && status.time_until_next_check) {
            const minutes = Math.floor(status.time_until_next_check / 60);
            const seconds = status.time_until_next_check % 60;
            nextCheckText.innerHTML = `<i class="fas fa-circle" style="color: ${status.running ? 'var(--accent-green)' : '#ff3a5e'};"></i> Next check in ${minutes}m ${seconds}s`;
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
                        statusDiv.className = data.bot_status === 'running' ? 'status-success' : 'status-info';
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
