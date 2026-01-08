# Design Document

## Overview

This design transforms the Banking Anomaly Detection System from customer-level to account-level fraud detection. The system will track limits, velocity, and spending patterns per Customer ID + Account Number combination, providing more granular fraud detection capabilities.

## Architecture

### Current vs New Architecture

**Current Architecture:**
```
Customer ID → User Stats → Fraud Detection
```

**New Architecture:**
```
Customer ID + Account Number → Account Key → Account-Specific Stats → Fraud Detection
```

### Key Components

1. **Account Key Generator**: Creates unique identifiers from Customer ID + Account Number
2. **Account-Level Statistics Calculator**: Computes behavioral patterns per account
3. **Enhanced Feature Engineering**: Generates 42 features with account-level granularity
4. **Account-Aware Session Management**: Tracks velocity and spending per account
5. **Updated UI Components**: Displays account-specific information

## Components and Interfaces

### 1. Account Key Management

```python
class AccountKeyManager:
    @staticmethod
    def create_account_key(customer_id: str, account_number: str) -> str:
        """Create unique account key from customer ID and account number"""
        return f"{customer_id}_{account_number}"
    
    @staticmethod
    def parse_account_key(account_key: str) -> tuple:
        """Parse account key back to customer ID and account number"""
        return account_key.split('_', 1)
```

### 2. Enhanced Feature Engineering

```python
class AccountLevelFeatureEngineering:
    def __init__(self):
        self.features = [
            # Core transaction features
            'transaction_amount', 'flag_amount', 'transfer_type_encoded',
            'transfer_type_risk', 'channel_encoded',
            
            # User behavior features (per account)
            'user_avg_amount', 'user_std_amount', 'user_max_amount',
            'user_txn_frequency', 'deviation_from_avg', 'amount_to_max_ratio',
            'rolling_std', 'transaction_velocity',
            
            # Temporal aggregation features (per account)
            'weekly_total', 'weekly_txn_count', 'weekly_avg_amount',
            'weekly_deviation', 'amount_vs_weekly_avg',
            'current_month_spending', 'monthly_txn_count', 'monthly_avg_amount',
            'monthly_deviation', 'amount_vs_monthly_avg',
            'hourly_total', 'hourly_count', 'daily_total', 'daily_count',
            
            # Time-based features
            'hour', 'day_of_week', 'is_weekend', 'is_night',
            'time_since_last', 'recent_burst', 'last_txn_time',
            
            # Velocity features (per account)
            'txn_count_30s', 'txn_count_10min', 'txn_count_1hour',
            
            # Risk assessment features (per account)
            'intl_ratio', 'user_high_risk_txn_ratio', 'user_multiple_accounts_flag',
            'cross_account_transfer_ratio', 'geo_anomaly_flag',
            
            # Beneficiary features (per account)
            'is_new_beneficiary', 'beneficiary_txn_count_30d'
        ]
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all 42 features with account-level granularity"""
        # Use AmountInAED for all calculations
        # Group by Customer ID + Account Number
        # Calculate statistics per account
        pass
```

### 3. Account-Aware Rule Engine

```python
class AccountLevelRuleEngine:
    def calculate_threshold(self, account_avg: float, account_std: float, 
                          transfer_type: str = 'O') -> float:
        """Calculate spending threshold for specific account"""
        multiplier = TRANSFER_MULTIPLIERS.get(transfer_type, 3.0)
        floor = TRANSFER_MIN_FLOORS.get(transfer_type, 2000)
        return max(account_avg + multiplier * account_std, floor)
    
    def check_rule_violation(self, amount: float, account_key: str,
                           account_stats: dict, transfer_type: str,
                           velocity_data: dict) -> tuple:
        """Check rule violations for specific account"""
        # Use account-specific statistics
        # Check account-specific velocity limits
        # Return account-specific violations
        pass
```

### 4. Account-Level Session Management

```python
class AccountSessionManager:
    def __init__(self):
        self.account_velocity = {}  # {account_key: [timestamps]}
        self.account_session_count = {}  # {account_key: count}
        self.account_monthly_spending = {}  # {account_key: amount}
    
    def get_velocity(self, account_key: str) -> dict:
        """Get velocity data for specific account"""
        pass
    
    def record_transaction(self, account_key: str):
        """Record transaction for specific account"""
        pass
    
    def add_monthly_spending(self, account_key: str, amount: float):
        """Add to monthly spending for specific account"""
        pass
```

## Data Models

### Enhanced Transaction Data Model

```python
@dataclass
class AccountTransaction:
    customer_id: str
    account_number: str
    account_key: str  # Generated from customer_id + account_number
    amount_in_aed: float  # Primary amount field
    transfer_type: str
    timestamp: datetime
    
    # All 42 engineered features
    transaction_amount: float
    user_avg_amount: float  # Account-specific average
    user_std_amount: float  # Account-specific std dev
    # ... (all other features)
```

### Account Statistics Model

```python
@dataclass
class AccountStatistics:
    account_key: str
    customer_id: str
    account_number: str
    
    # Basic statistics (per account)
    avg_amount: float
    std_amount: float
    max_amount: float
    transaction_frequency: int
    
    # Temporal statistics (per account)
    weekly_avg: float
    monthly_avg: float
    daily_avg: float
    
    # Risk metrics (per account)
    international_ratio: float
    high_risk_ratio: float
    cross_account_ratio: float
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property Reflection

After analyzing all acceptance criteria, several properties can be consolidated:
- Properties 2.2 and 2.3 (velocity tracking for different time windows) can be combined into a single comprehensive velocity isolation property
- Properties 1.1 and 1.2 (account-level statistics) can be combined into one property about account isolation
- Properties 3.1 and 3.2 (monthly spending tracking) can be combined into one comprehensive spending tracking property

### Core Properties

**Property 1: Account Isolation for Statistics**
*For any* customer with multiple accounts, calculating statistics for one account should only use transactions from that specific account and should be independent of other accounts' transactions
**Validates: Requirements 1.1, 1.2**

**Property 2: Account-Level Velocity Isolation**
*For any* time window (10 minutes or 1 hour), transaction velocity counts for an account should only include transactions from that specific account, regardless of other accounts belonging to the same customer
**Validates: Requirements 2.1, 2.2, 2.3**

**Property 3: Account-Level Spending Tracking**
*For any* account, monthly spending calculations should only include transactions from that specific account and should be tracked independently from other accounts
**Validates: Requirements 3.1, 3.2**

**Property 4: Account Ownership Validation**
*For any* account selection attempt, the system should only allow access to accounts that belong to the currently logged-in customer
**Validates: Requirements 4.4**

**Property 5: Session State Account Isolation**
*For any* session tracking operation, the system should use account keys to ensure that session data (velocity, counts, spending) is isolated per account
**Validates: Requirements 5.1**

**Property 6: Feature Engineering Account Grouping**
*For any* feature calculation, the system should group data by account key and use AmountInAED column consistently for all amount-based computations
**Validates: Requirements 6.1, 6.3**

**Property 7: Transaction Count Accuracy**
*For any* transaction processing, the system should accurately increment the transaction count for the specific account without affecting other accounts' counts
**Validates: Requirements 7.6**

**Property 8: Backward Compatibility Fallback**
*For any* historical data processing where account numbers are missing, the system should gracefully handle the missing data using default account identifiers
**Validates: Requirements 8.1**

**Property 9: Fallback Statistics Calculation**
*For any* account with insufficient transaction history, the system should fall back to customer-level statistics while maintaining account-level tracking for future transactions
**Validates: Requirements 1.3**

<function_calls>
<invoke name="prework">
<parameter name="featureName">account-level-fraud-detection

## Error Handling

### Account Key Generation Errors
- **Missing Customer ID**: Use "UNKNOWN_CUSTOMER" as fallback
- **Missing Account Number**: Use "DEFAULT_ACCOUNT" as fallback
- **Invalid Characters**: Sanitize input by removing special characters

### Feature Engineering Errors
- **Missing AmountInAED**: Fall back to Amount column, then to 0
- **Insufficient Account History**: Use customer-level statistics as fallback
- **Missing Temporal Data**: Use default values (hour=12, day_of_week=0)

### Session Management Errors
- **Account Key Not Found**: Initialize new account tracking
- **Velocity Calculation Errors**: Return safe defaults (0 transactions)
- **Monthly Spending Errors**: Return 0 and log warning

### UI Display Errors
- **No Accounts Available**: Show "Default Account" option
- **Account Selection Errors**: Revert to first available account
- **Statistics Display Errors**: Show "N/A" for unavailable data

## Testing Strategy

### Dual Testing Approach
The system will use both unit testing and property-based testing for comprehensive coverage:

**Unit Tests:**
- Test specific account key generation scenarios
- Test feature engineering with known data sets
- Test UI component rendering with various account configurations
- Test error handling with edge cases

**Property-Based Tests:**
- Test account isolation properties across randomly generated customer/account combinations
- Test velocity tracking with random transaction patterns
- Test spending calculations with various account configurations
- Test backward compatibility with randomly missing data

### Property-Based Testing Configuration
- **Testing Framework**: Hypothesis (Python)
- **Minimum Iterations**: 100 per property test
- **Test Tags**: Each property test will be tagged with format:
  - **Feature: account-level-fraud-detection, Property 1: Account Isolation for Statistics**
  - **Feature: account-level-fraud-detection, Property 2: Account-Level Velocity Isolation**
  - etc.

### Test Data Generation
- Generate customers with 1-5 accounts each
- Generate transaction histories with varying patterns
- Generate edge cases (missing data, insufficient history)
- Generate temporal patterns (different time zones, months, years)

This comprehensive testing approach ensures that the account-level fraud detection system maintains correctness while providing enhanced granularity and accuracy in fraud detection capabilities.