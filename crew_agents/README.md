# CrewAI Multi-Agent Crypto Intelligence

## Overview
A sophisticated multi-agent system for cryptocurrency trading intelligence, leveraging CrewAI to enhance reinforcement learning strategies.

## Key Components
- `trading_intelligence.py`: Multi-agent system for market analysis
- `rl_intelligence_bridge.py`: Bridges CrewAI insights with RL environment
- Specialized agents:
  1. Market Researcher
  2. Technical Analyst
  3. Risk Manager
  4. Strategy Optimizer
  5. Reward Function Engineer

## Features
- Advanced market trend analysis
- Dynamic strategy generation
- Intelligent reward function design
- Risk management insights
- Adaptive trading strategy recommendations

## Setup
1. Install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export OPENAI_API_KEY='your_openai_api_key'
```

## Usage
```python
from rl_intelligence_bridge import RLIntelligenceBridge

# Create enhanced RL environment for BTC-USD
intelligence_bridge = RLIntelligenceBridge(symbol='BTC-USD')
enhanced_env = intelligence_bridge.generate_enhanced_environment()

# Export intelligence report
report_path = intelligence_bridge.export_intelligence_report(enhanced_env)
```

## Workflow
1. Gather comprehensive market data
2. Generate multi-agent insights
3. Design adaptive reward function
4. Create optimized trading strategy
5. Integrate insights into RL environment

## Future Enhancements
- Real-time market data integration
- More sophisticated agent interactions
- Advanced machine learning model integration

## Performance Tracking
- Detailed logging of agent decisions
- Comprehensive performance metrics
- Continuous strategy refinement
