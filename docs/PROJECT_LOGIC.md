# Banking Fraud Detection System - Complete Logic Documentation (Updated)

## Overview
Yeh system banking transactions ko analyze karta hai aur fraud/anomalies detect karta hai using:
1. **Machine Learning (Isolation Forest)** - Unusual patterns detect karta hai
2. **Autoencoder Neural Network** - Behavioral anomalies detect karta hai  
3. **Rule Engine** - Dynamic thresholds check karta hai per transfer type

---

## File Structure (Updated)

```
├── app.py                    # Main Streamlit application
├── backend/
│   ├── utils.py              # Helper functions aur constants (42 features)
│   ├── rule_engine.py        # Dynamic threshold calculations
│   ├── feature_engineering.py # Data ko ML-ready features mein convert
│   ├── train_isolation_forest.py # Isolation Forest model training
│   ├── isolation_forest.py   # Isolation Forest inference
│   ├── train_autoencoder.py  # Autoencoder model training
│   ├── autoencoder.py        # Autoencoder inference
│   └── hybrid_decision.py    # ML + Rules combine karke decision
│   └── model/                # Trained models storage
│       ├── isolation_forest.pkl
│       ├── isolation_forest_scaler.pkl
│       ├── autoencoder.h5
│       ├── autoencoder_scaler.pkl
│       └── autoencoder_threshold.json
├── data/
│   ├── Clean.csv             # Original clean transaction data
│   └── feature_datasetv2.csv # Processed features wala data (42 features)
├── docs/                     # Documentation folder
│   ├── BRD.md               # Business Requirements Document
│   ├── PROJECT_LOGIC.md     # This documentation file
│   ├── projectarchitecture.md # System architecture
│   └── projectflow.md       # Project workflow
└── tests/                    # Testing files
```

---

## 1. utils.py - Helper Functions (Updated)

### Purpose:
Basic utility functions aur constants define karta hai, including centralized feature list.

### Key Components:

```python
# File paths
get_clean_csv_path()           # data/Clean.csv path
get_feature_engineered_path()  # data/feature_datasetv2.csv path  
get_model_path()               # backend/model/isolation_forest.pkl path

# Transfer Type Mappings
TRANSFER_TYPE_MAPPING = {
    'S': 'Overseas',    # International transfer - HIGH RISK
    'I': 'Ajman',       # Local Ajman transfer - LOW RISK
    'L': 'UAE',         # Within UAE transfer - LOW RISK
    'Q': 'Quick',       # Quick remittance - MEDIUM RISK
    'O': 'Own'          # Own account transfer - LOWEST RISK
}

# Encoded values for ML model
TRANSFER_TYPE_ENCODED = {'S': 4, 'I': 1, 'L': 2, 'Q': 3, 'O': 0}

# Risk scores (0 to 1, higher = more risky)
TRANSFER_TYPE_RISK = {'S': 0.9, 'I': 0.1, 'L': 0.2, 'Q': 0.5, 'O': 0.0}

# Centralized Features List (42 features)
MODEL_FEATURES = [
    'transaction_amount','flag_amount','transfer_type_encoded','transfer_type_risk',
    'channel_encoded','deviation_from_avg','amount_to_max_ratio','rolling_std',
    'transaction_velocity','weekly_total','weekly_txn_count','weekly_avg_amount',
    'weekly_deviation','amount_vs_weekly_avg','current_month_spending','monthly_txn_count',
    'monthly_avg_amount','monthly_deviation','amount_vs_monthly_avg','hourly_total',
    'hourly_count','daily_total','daily_count','hour','day_of_week','is_weekend',
    'is_night','time_since_last','recent_burst','txn_count_30s','txn_count_10min',
    'txn_count_1hour','user_avg_amount','user_std_amount','user_max_amount',
    'user_txn_frequency','intl_ratio','user_high_risk_txn_ratio',
    'user_multiple_accounts_flag','cross_account_transfer_ratio',
    'geo_anomaly_flag','is_new_beneficiary','beneficiary_txn_count_30d'
]
```

---

## 2. Model Architecture (Updated)

### Training vs Inference Separation:

#### **Training Files:**
- `train_isolation_forest.py` - IsolationForestTrainer class
- `train_autoencoder.py` - AutoencoderTrainer class

#### **Inference Files:**
- `isolation_forest.py` - IsolationForestInference class
- `autoencoder.py` - AutoencoderInference class

### Benefits:
- **Clean Architecture**: Training aur inference separate
- **Production Ready**: Fast inference without training overhead
- **Memory Efficient**: Sirf inference models load karte hain production mein
- **Consistent Features**: Sab models same 42 features use karte hain

---

## 3. Enhanced Feature Engineering (42 Features)

### Purpose:
Raw transaction data ko ML model ke liye 42 useful features mein convert karta hai.

### New Features Added:

#### Advanced Spending Analytics:
| Feature | Description |
|---------|-------------|
| `weekly_total` | Weekly spending totals |
| `weekly_txn_count` | Weekly transaction counts |
| `weekly_avg_amount` | Weekly average amounts |
| `weekly_deviation` | Weekly spending deviations |
| `amount_vs_weekly_avg` | Current vs weekly average |
| `current_month_spending` | Current month expenditure |
| `monthly_txn_count` | Monthly transaction counts |
| `monthly_avg_amount` | Monthly average amounts |
| `monthly_deviation` | Monthly spending deviations |
| `amount_vs_monthly_avg` | Current vs monthly average |

#### Enhanced Behavioral Features:
| Feature | Description |
|---------|-------------|
| `user_high_risk_txn_ratio` | Ratio of high-risk transactions |
| `user_multiple_accounts_flag` | Multiple account usage indicator |
| `cross_account_transfer_ratio` | Cross-account transfer patterns |
| `geo_anomaly_flag` | Geographic anomaly detection |
| `is_new_beneficiary` | New beneficiary indicator |
| `beneficiary_txn_count_30d` | Beneficiary transaction history |

---

## 4. Autoencoder Neural Network (New)

### Purpose:
Behavioral pattern analysis aur subtle anomaly detection.

### Architecture:
```python
Input Layer (42 features)
    ↓
Hidden Layer (64 neurons) + BatchNorm + ReLU
    ↓  
Hidden Layer (32 neurons) + BatchNorm + ReLU
    ↓
Bottleneck (Encoding Dim) + ReLU
    ↓
Hidden Layer (32 neurons) + BatchNorm + ReLU
    ↓
Hidden Layer (64 neurons) + BatchNorm + ReLU
    ↓
Output Layer (42 features)
```

### How it Works:
```
Normal transactions = Low reconstruction error
Anomalous transactions = High reconstruction error

Threshold = mean_error + (3 × std_error)
```

### Training Process:
1. **Data Loading**: feature_datasetv2.csv (42 features)
2. **Scaling**: StandardScaler for normalization
3. **Training**: 100 epochs with early stopping
4. **Threshold Calculation**: Statistical threshold from training errors
5. **Validation**: Model performance verification

---

## 5. Enhanced Decision Flow (Updated)

### Triple-Layer Architecture:

```
                    Transaction Input
                          ↓
              ┌───────────────────────┐
              │   Prepare Features    │
              │    (42 features)      │
              └───────────┬───────────┘
                          ↓
         ┌────────────────┼────────────────┐
         ↓                ↓                ↓
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   Rule Engine   │ │ Isolation Forest│ │   Autoencoder   │
│ (Business Rules)│ │  (ML Anomaly)   │ │  (Behavioral)   │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         ↓                   ↓                   ↓
    Hard Block?         Anomaly Score?      Reconstruction
                                           Error High?
         ↓                   ↓                   ↓
         └────────────────┬──┴───────────────────┘
                          ↓
                   Final Decision
                   (Combined Result)
```

### Decision Priority:
1. **Rule Engine**: Hard blocks (velocity, amount limits)
2. **Isolation Forest**: Statistical anomaly detection
3. **Autoencoder**: Behavioral pattern analysis
4. **Combined**: Aggregate all results with detailed reasons

---

## 6. Model Storage Organization (Updated)

### New Structure:
```
backend/model/
├── isolation_forest.pkl      # Trained IF model
├── isolation_forest_scaler.pkl # IF feature scaler
├── autoencoder.h5           # Trained AE model
├── autoencoder_scaler.pkl   # AE feature scaler
└── autoencoder_threshold.json # AE threshold config
```

### Benefits:
- **Organized**: Models grouped under backend
- **Consistent**: Same path structure for all models
- **Maintainable**: Easy to backup and deploy
- **Scalable**: Easy to add new models

---

## 7. Enhanced App Features (Updated)

### New Capabilities:
1. **Dual ML Models**: Both Isolation Forest aur Autoencoder
2. **Detailed Scoring**: Individual model scores aur combined result
3. **Enhanced Features**: 42 features instead of 26
4. **Better Architecture**: Clean separation of training/inference
5. **Improved UI**: More detailed results aur explanations

### Session Tracking (Enhanced):
```python
def enhanced_analysis(transaction):
    # Rule engine check
    rule_result = check_business_rules(transaction)
    
    # Isolation Forest analysis  
    if_result = isolation_forest_inference.score_transaction(features)
    
    # Autoencoder analysis
    ae_result = autoencoder_inference.score_transaction(features)
    
    # Combined decision
    final_decision = combine_results(rule_result, if_result, ae_result)
    
    return final_decision
```

---

## Data Flow Summary (Updated)

```
1. User Login
      ↓
2. Select Account + Enter Transaction Details
      ↓
3. Click "Analyze Transaction"
      ↓
4. Feature Engineering (42 features from utils.MODEL_FEATURES)
      ↓
5. Rule Engine Check (velocity + amount limits)
      ↓
6. Isolation Forest Analysis (anomaly scoring)
      ↓
7. Autoencoder Analysis (behavioral patterns)
      ↓
8. Combined Decision (aggregate all results)
      ↓
9. Display Detailed Results
   - Rule violations (if any)
   - ML anomaly scores
   - Behavioral analysis
   - Final recommendation
      ↓
10. Record Transaction (if approved)
    - Update session statistics
    - Log decision details
```

---

## Why These Enhancements?

### Why 4 Features?
- **More Comprehensive**: Better capture of user behavior
- **Temporal Patterns**: Weekly/monthly spending analysis
- **Enhanced Detection**: More sophisticated anomaly detection
- **Behavioral Insights**: Cross-account and beneficiary patterns

### Why Autoencoder?
- **Behavioral Analysis**: Detects subtle pattern changes
- **Complementary**: Different approach than Isolation Forest
- **Deep Learning**: Neural networks capture complex relationships
- **Unsupervised**: No labeled data required

### Why Separate Training/Inference?
- **Production Efficiency**: Faster inference without training overhead
- **Clean Architecture**: Clear separation of concerns
- **Memory Management**: Load only what's needed
- **Scalability**: Easy to deploy and maintain

### Why Centralized Features?
- **DRY Principle**: Single source of truth
- **Consistency**: All models use same features
- **Maintainability**: Easy to add/remove features
- **Error Prevention**: No feature mismatches

---

## Enhanced Transfer Type Risk Analysis

| Type | Risk | Multiplier | Min Floor | ML Features | AE Sensitivity |
|------|------|------------|-----------|-------------|----------------|
| S | HIGH | 2.0x | 5000 | High anomaly detection | Very sensitive |
| Q | MEDIUM | 2.5x | 3000 | Medium sensitivity | Moderate |
| L | LOW | 3.0x | 2000 | Standard detection | Normal |
| I | LOW | 3.5x | 1500 | Relaxed thresholds | Less sensitive |
| O | LOWEST | 4.0x | 1000 | Minimal detection | Least sensitive |

---

## Conclusion (Updated)

Enhanced system ab provide karta hai:

1. **Triple-Layer Protection** - Rules + Isolation Forest + Autoencoder
2. **42 Advanced Features** - Comprehensive behavioral analysis
3. **Clean Architecture** - Separate training/inference, centralized config
4. **Production Ready** - Optimized for real-time processing
5. **Scalable Design** - Easy to add new models and features
6. **Detailed Analytics** - Individual model scores aur combined insights

Yeh system ab industry-standard fraud detection capabilities provide karta hai with state-of-the-art machine learning techniques.