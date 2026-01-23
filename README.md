# Retail Data Assistant

An intelligent data assistant for retail analytics, powered by LangChain AI agents. This application provides natural language querying capabilities over retail transaction data, with advanced analytics for customer lifetime value (CLV) prediction, churn risk scoring, and customer segmentation.

## Features

### 🤖 AI-Powered Natural Language Queries
- Ask questions in plain English about your retail data
- Multi-step reasoning with conversation memory
- Automatic tool selection (SQL queries vs. predictive analytics)
- Powered by LangChain and OpenAI GPT models

### 📊 Advanced Analytics

#### Customer Lifetime Value (CLV)
- **BG/NBD Model**: Predicts purchase frequency
- **Gamma-Gamma Model**: Predicts average order value
- Calibrated predictions with scaling factors
- Horizon-based predictions (default: 90 days)

#### Survival Analysis & Churn Prediction
- **Kaplan-Meier**: Overall survival curves
- **Cox Proportional Hazards**: Risk-based churn prediction
- **Churn Probability**: Conditional probability of churn in next X days
- **Expected Remaining Lifetime**: Restricted expected lifetime in days
- **Customer Segmentation**: 9 segments combining risk and lifetime with action recommendations

### 🗄️ Data Management
- SQLite database with retail transaction data
- ETL pipeline for data loading and cleaning
- Read-only SQL query interface with security validation
- Database schema introspection

### 🎨 User Interface
- Streamlit-based web interface
- Chat-style interaction
- Conversation memory management
- Real-time analytics execution

## Architecture

```
┌─────────────────┐
│  Streamlit UI   │
│   (ui/app.py)   │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│  FastAPI Backend │
│  (app/main.py)  │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────────────┐
│ SQLite │ │ LangChain Agent │
│  DB    │ │ (llm_langchain) │
└────────┘ └────────┬─────────┘
                    │
         ┌──────────┼──────────┐
         ▼          ▼          ▼
    ┌────────┐ ┌──────────┐ ┌──────────┐
    │  CLV    │ │ Survival │ │   SQL    │
    │ Analytics│ │ Analytics│ │  Queries │
    └────────┘ └──────────┘ └──────────┘
```

## Setup

### Prerequisites
- Python 3.12+
- OpenAI API key (for LangChain agent)

### Installation

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd data-assistant
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4o-mini  # Optional, defaults to gpt-4o-mini
   API_URL=http://127.0.0.1:8000  # Optional, for UI
   ```

5. **Load the data**:
   ```bash
   python etl/load_online_retail.py
   ```
   This will:
   - Load data from `data/raw/online_retail.csv`
   - Clean and validate transactions
   - Create SQLite database at `data/retail.sqlite`

## Usage

### Start the Backend API

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at `http://127.0.0.1:8000`

### Start the Streamlit UI

In a separate terminal:

```bash
streamlit run ui/app.py
```

The UI will be available at `http://localhost:8501`

### Using the UI

1. Open the Streamlit app in your browser
2. Type questions in natural language, such as:
   - "What is the total revenue by country?"
   - "Show me the top 10 customers by sales"
   - "What is the customer lifetime value for customer 12345?"
   - "Which customers are at high risk of churning?"
   - "What is the probability customer 12345 will churn in 90 days?"
   - "Show me customer segments with recommended actions"

3. The AI agent will automatically:
   - Select the appropriate tool (SQL query or analytics function)
   - Execute the analysis
   - Return results in natural language

### Using the API Directly

#### Natural Language Queries
```bash
curl -X POST "http://127.0.0.1:8000/ask-langchain" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the total revenue by country?",
    "use_memory": true,
    "thread_id": "default"
  }'
```

#### Direct Analytics Endpoints

**Customer Lifetime Value**:
```bash
curl -X POST "http://127.0.0.1:8000/clv" \
  -H "Content-Type: application/json" \
  -d '{
    "cutoff_date": "2011-09-30",
    "horizon_days": 180,
    "limit_customers": 100
  }'
```

**Churn Risk Scoring**:
```bash
curl -X POST "http://127.0.0.1:8000/survival/score?inactivity_days=90&cutoff_date=2011-12-09"
```

**Churn Probability**:
```bash
curl -X POST "http://127.0.0.1:8000/survival/churn-probability?X_days=90&inactivity_days=90&cutoff_date=2011-12-09"
```

**Expected Remaining Lifetime**:
```bash
curl -X POST "http://127.0.0.1:8000/survival/expected-lifetime?H_days=365&inactivity_days=90&cutoff_date=2011-12-09"
```

**Customer Segmentation**:
```bash
curl -X POST "http://127.0.0.1:8000/survival/segmentation?H_days=365&inactivity_days=90&cutoff_date=2011-12-09"
```

#### SQL Queries
```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "sql": "SELECT country, SUM(revenue) as total_revenue FROM transactions GROUP BY country ORDER BY total_revenue DESC LIMIT 10",
    "limit": 100
  }'
```

#### Database Schema
```bash
curl "http://127.0.0.1:8000/schema"
```

## API Endpoints

### Core Endpoints

- `GET /health` - Health check
- `GET /schema` - Get database schema
- `POST /query` - Execute read-only SQL query
- `POST /ask-langchain` - Natural language question (AI agent)
- `POST /ask-langchain/clear-memory` - Clear conversation memory

### Analytics Endpoints

- `POST /clv` - Customer Lifetime Value prediction
- `POST /survival/km` - Kaplan-Meier survival curve
- `POST /survival/score` - Churn risk scoring
- `POST /survival/churn-probability` - Churn probability prediction
- `POST /survival/expected-lifetime` - Expected remaining lifetime
- `POST /survival/segmentation` - Customer segmentation

See the FastAPI docs at `http://127.0.0.1:8000/docs` for detailed API documentation.

## Analytics Capabilities

### Customer Lifetime Value (CLV)

Uses the **BG/NBD** (Beta-Geometric/Negative Binomial Distribution) and **Gamma-Gamma** models:

- **BG/NBD**: Models purchase frequency (how often customers buy)
- **Gamma-Gamma**: Models monetary value (how much customers spend per order)
- **Calibration**: Models are calibrated using train/test validation
- **Scaling**: Predictions are scaled to match historical patterns

**Output**: Predicted purchases, average order value, and total CLV for each customer.

### Survival Analysis

Uses **Cox Proportional Hazards** model for churn prediction:

- **Covariates**: 
  - `n_orders`: Total number of orders
  - `log_monetary_value`: Log-transformed mean order value
  - `product_diversity`: Number of unique products purchased

- **Churn Definition**: Customer inactive for ≥90 days (configurable)

- **Capabilities**:
  1. **Risk Scoring**: Relative churn risk (High/Medium/Low buckets)
  2. **Churn Probability**: Conditional probability of churn in next X days
  3. **Expected Lifetime**: Restricted expected remaining lifetime in days
  4. **Segmentation**: 9 segments combining risk and lifetime with action recommendations

**Segments**:
- High-Long: Priority Save
- High-Medium: Save
- High-Short: Let Churn
- Medium-Long: Growth Retain
- Medium-Medium: Nurture
- Medium-Short: Monitor
- Low-Long: VIP
- Low-Medium: Maintain
- Low-Short: Sunset

## Project Structure

```
data-assistant/
├── analytics/           # Analytics modules
│   ├── clv.py          # Customer Lifetime Value (BG/NBD + Gamma-Gamma)
│   └── survival.py      # Survival analysis (Cox model, churn prediction)
├── app/                 # FastAPI backend
│   ├── main.py         # API endpoints
│   ├── db.py           # Database utilities
│   └── llm_langchain.py # LangChain agent and tools
├── data/                # Data files
│   ├── raw/            # Raw data files
│   └── retail.sqlite   # SQLite database (generated)
├── etl/                 # ETL pipeline
│   └── load_online_retail.py
├── test/                # Test files
│   ├── test-clv.py
│   ├── test_cox_*.py
│   └── ...
├── ui/                  # Streamlit UI
│   └── app.py
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Testing

Run tests to validate analytics models:

```bash
# CLV calibration tests
python test/test-clv.py

# Survival analysis tests
python test/test_cox_baseline.py
python test/test_cox_combinations.py
python test/test_expected_lifetime.py
python test/test_score_customers.py
python test/test_survival_covariates.py
python test/test_survival_prob.py
```

## Configuration

### Model Parameters

**CLV Models** (in `analytics/clv.py`):
- `PURCHASE_SCALE`: Calibration scale for purchase predictions (default: 1.1)
- `REVENUE_SCALE`: Calibration scale for revenue predictions (default: 1.7)

**Survival Models** (in `analytics/survival.py`):
- `CUTOFF_DATE`: Default cutoff date for analysis (default: "2011-12-09")
- `INACTIVITY_DAYS`: Days of inactivity to define churn (default: 90)

**LangChain Agent** (in `app/llm_langchain.py`):
- `FIXED_CUTOFF_DATE`: Fixed cutoff for analytics functions (default: "2011-12-09")
- `FIXED_INACTIVITY_DAYS`: Fixed inactivity threshold (default: 90)
- `CACHE_TTL`: Model cache time-to-live in seconds (default: 3600)

## Security

- **SQL Injection Protection**: All SQL queries are validated and restricted to read-only operations
- **Input Validation**: Comprehensive validation for SQL queries and API parameters
- **Rate Limiting**: Consider implementing rate limiting for production use
- **API Keys**: Store OpenAI API keys securely in `.env` file (not in version control)

## Dependencies

Key dependencies:
- `fastapi` - Web framework
- `streamlit` - UI framework
- `langchain` - AI agent framework
- `langchain-openai` - OpenAI integration
- `langgraph` - Agent orchestration
- `lifetimes` - CLV models (BG/NBD, Gamma-Gamma)
- `lifelines` - Survival analysis (Cox, Kaplan-Meier)
- `pandas` - Data manipulation
- `sqlite3` - Database (built-in)

See `requirements.txt` for complete list.

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Support

[Add support information here]

