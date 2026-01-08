# Requirements Document

## Introduction

The Banking Anomaly Detection System currently uses Customer ID only for calculating limits, velocity tracking, and monthly spending. This enhancement will change the system to use Customer ID + Account Number combination for more granular fraud detection at the account level.

## Glossary

- **Customer_ID**: Unique identifier for a bank customer
- **Account_Number**: Unique account number within a customer's portfolio
- **Account_Key**: Combined identifier (Customer_ID + Account_Number) for account-specific tracking
- **Fraud_Detection_System**: The existing triple-layer fraud detection system
- **Velocity_Tracker**: Component that monitors transaction frequency
- **Limit_Calculator**: Component that calculates spending thresholds
- **Monthly_Spending_Tracker**: Component that tracks monthly expenditure

## Requirements

### Requirement 1: Account-Level Limit Calculation

**User Story:** As a fraud analyst, I want spending limits calculated per account rather than per customer, so that each account has its own risk profile and spending patterns.

#### Acceptance Criteria

1. WHEN calculating spending limits, THE Limit_Calculator SHALL use Customer_ID + Account_Number combination to determine user statistics
2. WHEN a customer has multiple accounts, THE Limit_Calculator SHALL calculate separate average, standard deviation, and maximum amounts for each account
3. WHEN an account has insufficient transaction history, THE Limit_Calculator SHALL use customer-level statistics as fallback
4. THE Limit_Calculator SHALL maintain the existing transfer type multipliers (S=2.0, Q=2.5, L=3.0, I=3.5, O=4.0)
5. THE Limit_Calculator SHALL maintain the existing minimum floor amounts per transfer type

### Requirement 2: Account-Level Velocity Tracking

**User Story:** As a fraud analyst, I want transaction velocity tracked per account, so that rapid transactions from one account don't affect limits on other accounts.

#### Acceptance Criteria

1. WHEN tracking transaction velocity, THE Velocity_Tracker SHALL use Account_Key (Customer_ID + Account_Number) as the tracking identifier
2. WHEN counting transactions in 10-minute windows, THE Velocity_Tracker SHALL only count transactions from the same account
3. WHEN counting transactions in 1-hour windows, THE Velocity_Tracker SHALL only count transactions from the same account
4. THE Velocity_Tracker SHALL maintain existing velocity limits (5 transactions per 10 minutes, 15 transactions per hour)
5. WHEN recording transaction timestamps, THE Velocity_Tracker SHALL store them with Account_Key association

### Requirement 3: Account-Level Monthly Spending

**User Story:** As a fraud analyst, I want monthly spending tracked per account, so that spending limits are enforced individually for each account.

#### Acceptance Criteria

1. WHEN tracking monthly spending, THE Monthly_Spending_Tracker SHALL use Account_Key as the tracking identifier
2. WHEN calculating current month spending from historical data, THE Monthly_Spending_Tracker SHALL filter by both Customer_ID and Account_Number
3. WHEN adding session spending, THE Monthly_Spending_Tracker SHALL associate amounts with the specific Account_Key
4. WHEN checking spending limits, THE Monthly_Spending_Tracker SHALL compare account-specific spending against account-specific limits
5. THE Monthly_Spending_Tracker SHALL reset monthly totals per account at month boundaries

### Requirement 4: Account Selection Integration

**User Story:** As a bank customer, I want to select which account to use for transactions, so that the system applies the correct account-specific fraud rules.

#### Acceptance Criteria

1. WHEN a customer logs in, THE Fraud_Detection_System SHALL display all available accounts for that customer
2. WHEN processing a transaction, THE Fraud_Detection_System SHALL use the selected account number in all fraud detection calculations
3. WHEN no account is selected, THE Fraud_Detection_System SHALL require account selection before processing
4. THE Fraud_Detection_System SHALL validate that the selected account belongs to the logged-in customer
5. WHEN displaying limits in the sidebar, THE Fraud_Detection_System SHALL show limits specific to the selected account

### Requirement 5: Data Model Updates

**User Story:** As a system architect, I want the data model updated to support account-level tracking, so that all components can access account-specific information.

#### Acceptance Criteria

1. WHEN storing session state, THE Fraud_Detection_System SHALL use Account_Key for all tracking dictionaries
2. WHEN saving transaction history, THE Fraud_Detection_System SHALL include both Customer_ID and Account_Number
3. WHEN loading historical data, THE Fraud_Detection_System SHALL group statistics by Account_Key
4. THE Fraud_Detection_System SHALL maintain backward compatibility with existing transaction history data
5. WHEN creating user statistics, THE Fraud_Detection_System SHALL calculate statistics per Account_Key

### Requirement 6: Enhanced Feature Engineering

**User Story:** As a data scientist, I want comprehensive feature engineering with account-level statistics and additional behavioral features, so that ML models receive rich account-specific patterns.

#### Acceptance Criteria

1. WHEN processing transaction data, THE Feature_Engineering_System SHALL use AmountInAED column for all amount calculations
2. WHEN creating features, THE Feature_Engineering_System SHALL generate all 42 specified features per Account_Key
3. WHEN calculating user behavioral features, THE Feature_Engineering_System SHALL group data by Account_Key (Customer_ID + Account_Number)
4. WHEN computing temporal features, THE Feature_Engineering_System SHALL calculate weekly, monthly, hourly, and daily statistics per account
5. WHEN generating velocity features, THE Feature_Engineering_System SHALL track transaction counts in 30-second, 10-minute, and 1-hour windows per account
6. WHEN calculating risk features, THE Feature_Engineering_System SHALL compute high-risk transaction ratios and cross-account transfer patterns
7. WHEN processing beneficiary data, THE Feature_Engineering_System SHALL track new beneficiaries and beneficiary transaction counts
8. THE Feature_Engineering_System SHALL save the enhanced feature dataset as feature_datasetv2.csv in the data folder

### Requirement 7: UI Display Updates

**User Story:** As a fraud analyst, I want the UI to display account-specific information with accurate transaction counts, so that I can monitor each account's fraud detection status separately.

#### Acceptance Criteria

1. WHEN displaying customer statistics in the sidebar, THE Fraud_Detection_System SHALL show statistics for the selected account
2. WHEN showing velocity information, THE Fraud_Detection_System SHALL display account-specific transaction counts
3. WHEN displaying monthly spending, THE Fraud_Detection_System SHALL show spending for the selected account only
4. WHEN showing transfer type limits, THE Fraud_Detection_System SHALL display limits calculated for the selected account
5. THE Fraud_Detection_System SHALL clearly indicate which account is currently selected
6. WHEN displaying total transaction count, THE Fraud_Detection_System SHALL accurately increment the count for the specific account
7. WHEN showing session transaction metrics, THE Fraud_Detection_System SHALL track and display counts per account accurately

### Requirement 9: Complete Feature Set Implementation

**User Story:** As a data scientist, I want all 42 specified features implemented with account-level granularity, so that the fraud detection system has comprehensive behavioral analysis.

#### Acceptance Criteria

1. THE Feature_Engineering_System SHALL implement these core transaction features:
   - transaction_amount (using AmountInAED)
   - flag_amount, transfer_type_encoded, transfer_type_risk, channel_encoded

2. THE Feature_Engineering_System SHALL implement these user behavior features per account:
   - user_avg_amount, user_std_amount, user_max_amount, user_txn_frequency
   - deviation_from_avg, amount_to_max_ratio, rolling_std, transaction_velocity

3. THE Feature_Engineering_System SHALL implement these temporal aggregation features per account:
   - weekly_total, weekly_txn_count, weekly_avg_amount, weekly_deviation, amount_vs_weekly_avg
   - current_month_spending, monthly_txn_count, monthly_avg_amount, monthly_deviation, amount_vs_monthly_avg
   - hourly_total, hourly_count, daily_total, daily_count

4. THE Feature_Engineering_System SHALL implement these time-based features:
   - hour, day_of_week, is_weekend, is_night
   - time_since_last, recent_burst, last_txn_time

5. THE Feature_Engineering_System SHALL implement these velocity features per account:
   - txn_count_30s, txn_count_10min, txn_count_1hour

6. THE Feature_Engineering_System SHALL implement these risk assessment features per account:
   - intl_ratio, user_high_risk_txn_ratio, user_multiple_accounts_flag
   - cross_account_transfer_ratio, geo_anomaly_flag

7. THE Feature_Engineering_System SHALL implement these beneficiary features per account:
   - is_new_beneficiary, beneficiary_txn_count_30d

### Requirement 8: Backward Compatibility

**User Story:** As a system administrator, I want the system to handle existing data gracefully, so that historical transactions remain valid during the transition.

#### Acceptance Criteria

1. WHEN processing historical data without account numbers, THE Fraud_Detection_System SHALL use a default account identifier
2. WHEN encountering missing account information, THE Fraud_Detection_System SHALL fall back to customer-level calculations
3. THE Fraud_Detection_System SHALL continue to function with existing CSV data formats
4. WHEN migrating session state, THE Fraud_Detection_System SHALL preserve existing customer-level tracking data
5. THE Fraud_Detection_System SHALL provide clear error messages when account information is missing