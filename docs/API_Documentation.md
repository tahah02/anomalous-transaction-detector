# Banking Fraud Detection API - Complete Documentation

## Overview

The `api.py` file implements a FastAPI-based REST API that exposes the existing Streamlit fraud detection system through HTTP endpoints. This API maintains the same triple-layer fraud detection logic while providing programmatic access for integration with banking systems.

## File Structure Analysis

### Imports and Dependencies
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional
```
- **FastAPI**: Modern, high-performance web framework for building APIs
- **Pydantic**: Data validation using Python type annotations
- **datetime**: Timestamp handling for transactions and velocity tracking
- **typing**: Type hints for better code clarity and validation

### Data Models (Pydantic Classes)

#### TransactionRequest
```python
class TransactionRequest(BaseModel):
    customer_id: str
    from_account_no: str
    to_account_no: str
    transaction_amount: float = Field(gt=0)
    transfer_type: str = Field(pattern="^[SILQO]$")
    datetime: datetime
    bank_country: Optional[str] = "UAE"
```
**Purpose**: Validates incoming transaction data
- `customer_id`: Unique customer identifier
- `from_account_no`: Source account number
- `to_account_no`: Destination account number
- `transaction_amount`: Must be positive (gt=0)
- `transfer_type`: Must be one of S/I/L/Q/O (regex validation)
- `datetime`: Transaction timestamp
- `bank_country`: Optional, defaults to "UAE"

#### TransactionResponse
```python
class TransactionResponse(BaseModel):
    decision: str
    risk_score: float
    confidence_level: float
    reasons: List[str]
    individual_scores: dict
    transaction_id: str
    processing_time_ms: int
```
**Purpose**: Structured response format
- `decision`: "APPROVED" or "REQUIRES_USER_APPROVAL"
- `risk_score`: Numerical risk assessment (0.0 to 1.0)
- `confidence_level`: System confidence in decision
- `reasons`: Array of human-readable explanations
- `individual_scores`: Breakdown by detection layer
- `transaction_id`: Unique identifier for tracking
- `processing_time_ms`: Performance metric

#### BatchRequest & BatchResponse
```python
class BatchRequest(BaseModel):
    transactions: List[TransactionRequest] = Field(max_items=100)

class BatchResponse(BaseModel):
    results: List[TransactionResponse]
    summary: dict
```
**Purpose**: Handle multiple transactions in single request
- Limited to 100 transactions per batch
- Returns individual results plus summary statistics

### UserStatsManager Class

#### Purpose
Replaces Streamlit's session state with persistent file-based storage for user statistics and velocity tracking.

#### Key Methods

**`__init__(self)`**
```python
def __init__(self):
    self.stats_file = "data/user_stats.json"
    self.velocity_file = "data/velocity_counters.json"
    self.stats = self.load_stats()
    self.velocity = self.load_velocity()
```
- Initializes file paths for persistent storage
- Loads existing data on startup

**`load_stats(self)` & `load_velocity(self)`**
```python
def load_stats(self):
    if os.path.exists(self.stats_file):
        with open(self.stats_file, 'r') as f:
            return json.load(f)
    return {}
```
- Loads user statistics from JSON files
- Returns empty dict if files don't exist
- Handles file I/O errors gracefully

**`get_user_stats(self, customer_id: str, account_no: str)`**
```python
def get_user_stats(self, customer_id: str, account_no: str):
    key = f"{customer_id}_{account_no}"
    if key not in self.stats:
        self.stats[key] = {
            "user_avg_amount": 5000.0,
            "user_std_amount": 2000.0,
            "user_max_amount": 15000.0,
            "user_txn_frequency": 0,
            "user_international_ratio": 0.0,
            "current_month_spending": 0.0
        }
    return self.stats[key]
```
- Creates composite key from customer_id and account_no
- Initializes default statistics for new customers
- Returns existing statistics for known customers

**`get_velocity_metrics(self, customer_id: str, account_no: str)`**
```python
def get_velocity_metrics(self, customer_id: str, account_no: str):
    key = f"{customer_id}_{account_no}"
    now = datetime.now()
    
    if key not in self.velocity:
        self.velocity[key] = []
    
    history = [datetime.fromisoformat(t) for t in self.velocity[key]]
    history = [t for t in history if (now - t).total_seconds() < 3600]
    
    count_10min = sum(1 for t in history if (now - t).total_seconds() < 600)
    count_1hour = len(history)
    time_since_last = (now - max(history)).total_seconds() if history else 3600
    
    return {
        'txn_count_10min': count_10min,
        'txn_count_1hour': count_1hour,
        'time_since_last_txn': time_since_last
    }
```
- Filters transaction history to last hour
- Calculates velocity metrics for 10-minute and 1-hour windows
- Computes time since last transaction
- Returns metrics used by fraud detection engine

**`record_transaction(self, customer_id: str, account_no: str, amount: float)`**
```python
def record_transaction(self, customer_id: str, account_no: str, amount: float):
    key = f"{customer_id}_{account_no}"
    now = datetime.now()
    
    if key not in self.velocity:
        self.velocity[key] = []
    self.velocity[key].append(now.isoformat())
    
    stats = self.get_user_stats(customer_id, account_no)
    stats["user_txn_frequency"] += 1
    stats["current_month_spending"] += amount
    
    self.save_stats()
```
- Records transaction timestamp for velocity tracking
- Updates user statistics (frequency, spending totals)
- Persists changes to disk immediately

### Global Initialization
```python
stats_manager = UserStatsManager()
model, features, scaler = load_model()
autoencoder = AutoencoderInference()
autoencoder.load()
```
- Creates single instance of UserStatsManager
- Loads pre-trained ML models (Isolation Forest)
- Initializes Autoencoder neural network
- Models are loaded once at startup for performance

### API Endpoints

#### Health Check Endpoint
```python
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "isolation_forest": "loaded" if model else "unavailable",
            "autoencoder": "loaded" if autoencoder else "unavailable"
        }
    }
```
**Purpose**: System monitoring and diagnostics
- Returns current system status
- Indicates model availability
- Provides timestamp for monitoring

#### Single Transaction Analysis
```python
@app.post("/api/analyze-transaction", response_model=TransactionResponse)
def analyze_transaction(request: TransactionRequest):
```
**Flow**:
1. **Timing**: Records start time for performance measurement
2. **Data Retrieval**: Gets user statistics and velocity metrics
3. **Transaction Preparation**: Formats data for fraud detection engine
4. **Fraud Analysis**: Calls existing `make_decision` function
5. **Decision Mapping**: Converts internal result to API response format
6. **State Update**: Records transaction if approved
7. **Response Generation**: Creates structured response with timing

**Key Logic**:
```python
txn = {
    'amount': request.transaction_amount,
    'transfer_type': request.transfer_type,
    'bank_country': request.bank_country,
    'txn_count_10min': velocity['txn_count_10min'] + 1,
    'txn_count_1hour': velocity['txn_count_1hour'] + 1,
    'time_since_last_txn': velocity['time_since_last_txn']
}

result = make_decision(txn, user_stats, model, features, autoencoder)
decision = "REQUIRES_USER_APPROVAL" if result['is_fraud'] else "APPROVED"
```
- Reuses existing Streamlit fraud detection logic
- Increments velocity counters for current transaction
- Maps boolean fraud flag to API decision format

#### Batch Transaction Analysis
```python
@app.post("/api/analyze-batch", response_model=BatchResponse)
def analyze_batch(request: BatchRequest):
```
**Flow**:
1. **Initialization**: Creates result arrays and counters
2. **Individual Processing**: Calls `analyze_transaction` for each item
3. **Error Handling**: Catches and handles individual transaction failures
4. **Summary Generation**: Aggregates results and statistics
5. **Response Assembly**: Returns individual results plus batch summary

**Error Handling**:
```python
try:
    result = analyze_transaction(txn)
    results.append(result)
except Exception as e:
    error_result = TransactionResponse(
        decision="REQUIRES_USER_APPROVAL",
        risk_score=1.0,
        confidence_level=0.0,
        reasons=[f"Processing error: {str(e)}"],
        individual_scores={},
        transaction_id=f"error_{i}",
        processing_time_ms=0
    )
    results.append(error_result)
```
- Individual transaction failures don't stop batch processing
- Failed transactions default to requiring user approval
- Error details are included in response

### Application Startup
```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
- Starts FastAPI server using Uvicorn ASGI server
- Listens on all interfaces (0.0.0.0) port 8000
- Enables direct execution with `python api.py`

## Key Design Decisions

### 1. **Minimal Code Changes**
- Reuses existing `make_decision` function from Streamlit app
- Maintains same fraud detection logic and business rules
- Only changes data input/output format

### 2. **File-Based Persistence**
- Replaces browser session state with JSON files
- Enables stateful behavior across API calls
- Simple deployment without database requirements

### 3. **Pydantic Validation**
- Automatic request validation and error responses
- Type safety and documentation generation
- Clear error messages for invalid inputs

### 4. **Performance Optimization**
- Models loaded once at startup
- In-memory statistics with periodic persistence
- Efficient velocity calculations with time filtering

### 5. **Error Resilience**
- Graceful handling of model failures
- Default to requiring approval when uncertain
- Individual transaction isolation in batch processing

## Usage Examples

### Starting the API
```bash
pip install -r requirements_api.txt
python api.py
```

### Testing with curl
```bash
# Health check
curl http://localhost:8000/api/health

# Single transaction
curl -X POST http://localhost:8000/api/analyze-transaction \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "12345",
    "from_account_no": "1001234567",
    "to_account_no": "2001234567",
    "transaction_amount": 5000.0,
    "transfer_type": "S",
    "datetime": "2024-01-15T14:30:00Z"
  }'
```

### Integration with Banking Systems
```python
import requests

response = requests.post(
    "http://localhost:8000/api/analyze-transaction",
    json={
        "customer_id": "12345",
        "from_account_no": "1001234567",
        "to_account_no": "2001234567",
        "transaction_amount": 5000.0,
        "transfer_type": "S",
        "datetime": "2024-01-15T14:30:00Z"
    }
)

result = response.json()
if result["decision"] == "APPROVED":
    # Process transaction automatically
    process_transaction(transaction_data)
else:
    # Route for manual review
    queue_for_approval(transaction_data, result["reasons"])
```

This API provides a clean, efficient interface to the existing fraud detection system while maintaining all business logic and detection capabilities.