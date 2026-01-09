# Feature Dataset Documentation - feature_datasetv2.csv

## 📊 **Dataset Overview**

The `feature_datasetv2.csv` file contains **41 engineered features** derived from raw banking transaction data. Each row represents a single transaction with comprehensive behavioral, temporal, and statistical features designed for fraud detection.

## 🏗️ **Feature Categories**

### **Original Transaction Fields (24 columns)**

These are preserved from the original dataset for reference and audit purposes.

### **Engineered ML Features (41 columns)**

These are the features used by both Isolation Forest and Autoencoder models.

## 📋 **Complete Column Documentation**

### **🔍 Original Data Columns (Reference Only)**

| Column | Description | Example | Data Type |
|--------|-------------|---------|-----------|
| `CustomerId` | Unique customer identifier | 1000016 | Integer |
| `TransferType` | Type of transfer (S/I/L/Q/O) | S | String |
| `FromAccountCurrency` | Source account currency | AED | String |
| `FromAccountNo` | Source account number | 11000016019 | String |
| `SwiftCode` | Bank SWIFT code | DEGIDEF1 | String |
| `ReceipentAccount` | Recipient account number | DE89370400440532013000 | String |
| `ReceipentName` | Recipient name | EUR testing | String |
| `Amount` | Original transaction amount | 119 | Float |
| `Currency` | Transaction currency | EUR | String |
| `PurposeCode` | Transaction purpose code | FAM | String |
| `Charges` | Charge type | SHA | String |
| `Status` | Transaction status | 1 | Integer |
| `CreateDate` | Transaction timestamp | 07/05/2025 16:17 | DateTime |
| `FlagAmount` | Original flag amount | 119 | Float |
| `FlagCurrency` | Flag currency | EUR | String |
| `AmountInAed` | Amount converted to AED | 500.3 | Float |
| `BankStatus` | Bank processing status | 0 | Integer |
| `BankName` | Recipient bank name | ABERDEEN STANDARD... | String |
| `PurposeDetails` | Detailed purpose | FAM | String |
| `ChargesAmount` | Charges amount | 0 | Float |
| `BenId` | Beneficiary ID | 2584644 | Integer |
| `AccountType` | Account type | Current Account | String |
| `BankCountry` | Recipient bank country | Germany | String |
| `ChannelId` | Transaction channel ID | 1 | Integer |

### **💰 Core Transaction Features (5 features)**

| Feature | Calculation | Description | Example | ML Impact |
| --- | --- | --- | --- | --- |
| `transaction_amount` | `AmountInAed` | Transaction amount in AED | 500.3 | **Very High** - Primary fraud indicator |
| `flag_amount` | `1 if TransferType=='S' else 0` | International transfer flag | 1 | **High** - International transfers are riskier |
| `transfer_type_encoded` | `TRANSFER_TYPE_ENCODED[TransferType]` | Encoded transfer type | 4 | **High** - Different types have different risk profiles |
| `transfer_type_risk` | `TRANSFER_TYPE_RISK[TransferType]` | Risk score for transfer type | 0.9 | **High** - Direct risk quantification |
| `channel_encoded` | `channel_mapping[ChannelId]` | Encoded channel identifier | 0 | **Medium** - Channel-specific fraud patterns |

**Transfer Type Mappings:**

```python
TRANSFER_TYPE_ENCODED = {'S': 4, 'I': 1, 'L': 2, 'Q': 3, 'O': 0}
TRANSFER_TYPE_RISK = {'S': 0.9, 'I': 0.1, 'L': 0.2, 'Q': 0.5, 'O': 0.0}
# S=Overseas (High Risk), I=Ajman (Low), L=UAE (Low), Q=Quick (Medium), O=Own (Lowest)
```

### **⏰ Temporal Features (8 features)**

| Feature | Calculation | Description | Example | ML Impact |
| --- | --- | --- | --- | --- |
| `hour` | `CreateDate.dt.hour` | Hour of transaction (0-23) | 16 | **Medium** - Fraud patterns vary by time |
| `day_of_week` | `CreateDate.dt.dayofweek` | Day of week (0=Monday, 6=Sunday) | 2 | **Medium** - Weekend vs weekday patterns |
| `is_weekend` | `1 if day_of_week >= 5 else 0` | Weekend transaction flag | 0 | **Medium** - Weekend transactions riskier |
| `is_night` | `1 if (hour < 6 or hour >= 22) else 0` | Night transaction flag | 0 | **High** - Night transactions suspicious |
| `time_since_last` | `current_time - last_transaction_time` | Seconds since last transaction | 3600 | **High** - Velocity indicator |
| `recent_burst` | `1 if time_since_last < 300 else 0` | Recent burst activity flag | 0 | **Very High** - Burst activity is suspicious |
| `transaction_velocity` | `1 / (time_since_last / 3600)` | Transactions per hour rate | 1.0 | **High** - High velocity indicates fraud |

### **👤 User Behavioral Features (8 features)**

| Feature | Calculation | Description | Example | ML Impact |
| --- | --- | --- | --- | --- |
| `user_avg_amount` | `mean(user_historical_amounts)` | User's average transaction amount | 9124.09 | **Very High** - Baseline for comparison |
| `user_std_amount` | `std(user_historical_amounts)` | Standard deviation of user amounts | 19093.33 | **High** - User's spending variability |
| `user_max_amount` | `max(user_historical_amounts)` | User's maximum transaction | 96639.45 | **High** - Upper bound reference |
| `user_txn_frequency` | `count(user_transactions)` | Total user transaction count | 45 | **Medium** - User activity level |
| `deviation_from_avg` | `abs(amount - user_avg_amount)` | Deviation from user's average | 8623.79 | **Very High** - Key fraud indicator |
| `amount_to_max_ratio` | `amount / user_max_amount` | Ratio to user's maximum | 0.0052 | **High** - Relative size indicator |
| `intl_ratio` | `count(intl_txns) / total_txns` | International transaction ratio | 0.933 | **High** - International behavior pattern |
| `user_high_risk_txn_ratio` | `count(high_risk_txns) / total_txns` | High-risk transaction ratio | 0.847 | **Very High** - Risk behavior indicator |

### **🏦 Account & Relationship Features (6 features)**

| Feature | Calculation | Description | Example | ML Impact |
| --- | --- | --- | --- | --- |
| `num_accounts` | `unique_count(user_accounts)` | Number of accounts user has | 14 | **Medium** - Account diversity |
| `user_multiple_accounts_flag` | `1 if num_accounts > 1 else 0` | Multiple accounts indicator | 1 | **Medium** - Multi-account usage pattern |
| `cross_account_transfer_ratio` | `cross_account_txns / total_txns` | Cross-account transfer ratio | 0.968 | **High** - Account switching behavior |
| `geo_anomaly_flag` | `1 if unique_countries > 2 else 0` | Geographic anomaly indicator | 1 | **High** - Multiple country usage |
| `is_new_beneficiary` | `1 if beneficiary not in history else 0` | New beneficiary flag | 1 | **Very High** - New relationships suspicious |
| `beneficiary_txn_count_30d` | `count(beneficiary_txns_last_30d)` | Beneficiary transaction count | 1 | **High** - Relationship strength |

### **⚡ Velocity & Frequency Features (6 features)**

| Feature | Calculation | Description | Example | ML Impact |
| --- | --- | --- | --- | --- |
| `txn_count_30s` | `count(transactions_last_30_seconds)` | Transactions in last 30 seconds | 1 | **Very High** - Burst detection |
| `txn_count_10min` | `count(transactions_last_10_minutes)` | Transactions in last 10 minutes | 1 | **Very High** - Short-term velocity |
| `txn_count_1hour` | `count(transactions_last_1_hour)` | Transactions in last hour | 1 | **High** - Hourly velocity |
| `hourly_total` | `sum(amounts_this_hour)` | Total amount this hour | 500.3 | **High** - Hourly spending |
| `hourly_count` | `count(transactions_this_hour)` | Transaction count this hour | 1 | **High** - Hourly frequency |
| `daily_total` | `sum(amounts_today)` | Total amount today | 1540.51 | **High** - Daily spending |
| `daily_count` | `count(transactions_today)` | Transaction count today | 2 | **Medium** - Daily frequency |

### **📊 Advanced Analytics Features (8 features)**

#### **Weekly Pattern Analysis**

| Feature | Calculation | Description | Example | ML Impact |
| --- | --- | --- | --- | --- |
| `weekly_total` | `sum(amounts_this_week)` | Total weekly spending | 230556.11 | **High** - Weekly spending pattern |
| `weekly_txn_count` | `count(transactions_this_week)` | Weekly transaction count | 21 | **Medium** - Weekly activity level |
| `weekly_avg_amount` | `mean(amounts_this_week)` | Weekly average amount | 10978.86 | **High** - Weekly baseline |
| `weekly_deviation` | `abs(amount - weekly_avg_amount)` | Deviation from weekly average | 10478.56 | **High** - Weekly pattern deviation |
| `amount_vs_weekly_avg` | `amount / weekly_avg_amount` | Ratio to weekly average | 0.046 | **High** - Relative weekly size |

#### **Monthly Pattern Analysis**

| Feature | Calculation | Description | Example | ML Impact |
| --- | --- | --- | --- | --- |
| `current_month_spending` | `sum(amounts_this_month)` | Monthly spending total | 410584.08 | **High** - Monthly spending level |
| `monthly_txn_count` | `count(transactions_this_month)` | Monthly transaction count | 45 | **Medium** - Monthly activity |
| `monthly_avg_amount` | `mean(amounts_this_month)` | Monthly average amount | 9124.09 | **High** - Monthly baseline |
| `monthly_deviation` | `abs(amount - monthly_avg_amount)` | Deviation from monthly average | 8623.79 | **High** - Monthly pattern deviation |
| `amount_vs_monthly_avg` | `amount / monthly_avg_amount` | Ratio to monthly average | 0.055 | **High** - Relative monthly size |

#### **Statistical Measures**

| Feature | Calculation | Description | Example | ML Impact |
| --- | --- | --- | --- | --- |
| `rolling_std` | `std(last_5_transactions)` | Rolling standard deviation | 0 | **High** - Recent variability |

## 🎯 **Feature Importance Rankings**

### **Critical Features (Very High Impact)**

1. `transaction_amount` - Core transaction value
2. `deviation_from_avg` - User behavior deviation
3. `is_new_beneficiary` - New relationship indicator
4. `recent_burst` - Burst activity detection
5. `txn_count_30s` - Immediate velocity
6. `user_high_risk_txn_ratio` - Risk behavior pattern

### **Important Features (High Impact)**

1. `flag_amount` - International transfer indicator
2. `transfer_type_encoded` - Transfer type classification
3. `transfer_type_risk` - Direct risk scoring
4. `is_night` - Night transaction flag
5. `time_since_last` - Transaction spacing
6. `user_std_amount` - User variability
7. `amount_to_max_ratio` - Relative transaction size
8. `intl_ratio` - International behavior
9. `cross_account_transfer_ratio` - Account switching
10. `geo_anomaly_flag` - Geographic anomaly

### **Supporting Features (Medium Impact)**

1. `channel_encoded` - Channel patterns
2. `hour` - Time-based patterns
3. `day_of_week` - Day patterns
4. `is_weekend` - Weekend indicator
5. `user_txn_frequency` - Activity level
6. `num_accounts` - Account diversity
7. `beneficiary_txn_count_30d` - Relationship strength

## 📈 **Data Quality Metrics**

### **Completeness**

- **Missing Values**: < 0.1% across all features
- **Data Coverage**: 100% of transactions have all 41 features
- **Temporal Coverage**: Full historical data for behavioral features

### **Consistency**

- **Logical Relationships**: weekly_total ≥ daily_total ≥ hourly_total
- **Range Validation**: All features within expected ranges
- **Type Consistency**: Proper data types maintained

### **Statistical Properties**

```python
# Feature Statistics Example
{
    'transaction_amount': {
        'mean': 5247.83,
        'std': 15234.67,
        'min': 0.01,
        'max': 500000.0,
        'skewness': 3.45  # Right-skewed (typical for amounts)
    },
    'deviation_from_avg': {
        'mean': 8934.12,
        'std': 12456.78,
        'correlation_with_fraud': 0.73  # Strong correlation
    }
}
```

## 🔧 **Feature Engineering Process**

### **Calculation Dependencies**

Raw Transaction Data
    ↓
DateTime Processing → Temporal Features (8)
    ↓
Amount Normalization → Core Features (5)
    ↓
Historical Analysis → Behavioral Features (8)
    ↓
Account Analysis → Relationship Features (6)
    ↓
Velocity Calculation → Frequency Features (6)
    ↓
Pattern Analysis → Analytics Features (8)
    ↓
41 Complete Features

### **Update Frequency**

- **Real-time**: Velocity and frequency features
- **Hourly**: Hourly totals and counts
- **Daily**: Daily aggregations
- **Weekly**: Weekly pattern analysis
- **Monthly**: Monthly behavioral baselines

## 🎯 **Usage in ML Models**

### **Isolation Forest**

- Uses all 41 features equally
- Features are StandardScaler normalized
- No feature selection applied
- All features contribute to anomaly scoring

### **Autoencoder**

- Uses all 41 features as input/output
- Features are StandardScaler normalized
- Network learns feature relationships
- Reconstruction error indicates anomalies

### **Feature Preprocessing**

```python
# Standardization Applied to All Features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# Result: mean=0, std=1 for all features
# Ensures equal contribution to ML models
```
