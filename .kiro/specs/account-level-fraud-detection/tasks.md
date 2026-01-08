# Implementation Plan: Account-Level Fraud Detection

## Overview

Transform the fraud detection system from customer-level to account-level tracking by implementing Customer ID + Account Number combination logic for limits, velocity, and spending calculations. Generate 42 enhanced features using AmountInAED column and save to feature_datasetv2.csv.

## Tasks

- [ ] 1. Create account key management utilities
  - Implement account key generation and parsing functions
  - Add account key validation logic
  - _Requirements: 5.1, 8.1_

- [ ]* 1.1 Write property test for account key generation
  - **Property 1: Account Isolation for Statistics**
  - **Validates: Requirements 1.1, 1.2**

- [-] 2. Update feature engineering for account-level processing
  - [x] 2.1 Modify feature_engineering.py to use AmountInAED column
    - Replace all amount calculations to use AmountInAED
    - Group all statistics by Customer ID + Account Number
    - _Requirements: 6.1, 6.3_

  - [x] 2.2 Implement all 42 features with account-level granularity
    - Add core transaction features (transaction_amount, flag_amount, etc.)
    - Add user behavior features per account (user_avg_amount, user_std_amount, etc.)
    - Add temporal aggregation features per account (weekly_total, monthly_avg_amount, etc.)
    - Add time-based features (hour, day_of_week, is_weekend, etc.)
    - Add velocity features per account (txn_count_30s, txn_count_10min, etc.)
    - Add risk assessment features per account (intl_ratio, user_high_risk_txn_ratio, etc.)
    - Add beneficiary features per account (is_new_beneficiary, beneficiary_txn_count_30d)
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7_

  - [x] 2.3 Save enhanced dataset as feature_datasetv2.csv
    - Output all 42 features to data/feature_datasetv2.csv
    - _Requirements: 6.8_

- [ ]* 2.4 Write property test for feature engineering account grouping
  - **Property 6: Feature Engineering Account Grouping**
  - **Validates: Requirements 6.1, 6.3**

- [ ] 3. Update rule engine for account-level calculations
  - [ ] 3.1 Modify rule_engine.py to use account-specific statistics
    - Update calculate_threshold to use account-level averages and std dev
    - Update check_rule_violation to use account-specific data
    - _Requirements: 1.1, 1.2, 1.3_

- [ ]* 3.2 Write property test for account-level velocity isolation
  - **Property 2: Account-Level Velocity Isolation**
  - **Validates: Requirements 2.1, 2.2, 2.3**

- [-] 4. Update session management for account-level tracking
  - [ ] 4.1 Modify app.py session state to use account keys
    - Update txn_history, session_count, monthly_spending to use account keys
    - Update get_velocity function to track per account
    - Update record_transaction to use account keys
    - _Requirements: 2.1, 2.2, 2.3, 3.1, 3.2_

  - [ ] 4.2 Update monthly spending calculations
    - Modify get_monthly_spending_from_csv to filter by account
    - Update add_monthly_spending to use account keys
    - _Requirements: 3.1, 3.2_

- [ ]* 4.3 Write property test for account-level spending tracking
  - **Property 3: Account-Level Spending Tracking**
  - **Validates: Requirements 3.1, 3.2**

- [ ] 5. Update hybrid decision engine for account-level processing
  - [ ] 5.1 Modify hybrid_decision.py to pass account-specific data
    - Update make_decision to use account-specific user_stats
    - Ensure all feature calculations use account-level data
    - _Requirements: 1.1, 1.2, 6.3_

- [ ]* 5.2 Write property test for session state account isolation
  - **Property 5: Session State Account Isolation**
  - **Validates: Requirements 5.1**

- [ ] 6. Update UI for account-level display and selection
  - [ ] 6.1 Modify dashboard function for account selection
    - Update account selection logic to be mandatory
    - Ensure selected account is used throughout processing
    - _Requirements: 4.1, 4.4_

  - [ ] 6.2 Update sidebar statistics display
    - Show account-specific statistics instead of customer-level
    - Update velocity display to show account-specific counts
    - Update monthly spending to show account-specific amounts
    - Update limits display to show account-specific thresholds
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [ ] 6.3 Fix transaction count accuracy
    - Ensure transaction counts increment correctly per account
    - Update session metrics to track per account
    - _Requirements: 7.6, 7.7_

- [ ]* 6.4 Write property test for transaction count accuracy
  - **Property 7: Transaction Count Accuracy**
  - **Validates: Requirements 7.6**

- [ ] 7. Implement backward compatibility and error handling
  - [ ] 7.1 Add fallback logic for missing account data
    - Handle missing account numbers with default values
    - Implement customer-level fallback for insufficient account history
    - _Requirements: 8.1, 8.2, 8.3, 1.3_

- [ ]* 7.2 Write property test for backward compatibility fallback
  - **Property 8: Backward Compatibility Fallback**
  - **Validates: Requirements 8.1**

- [ ]* 7.3 Write property test for fallback statistics calculation
  - **Property 9: Fallback Statistics Calculation**
  - **Validates: Requirements 1.3**

- [ ] 8. Update data loading and model integration
  - [ ] 8.1 Update load_data function to use feature_datasetv2.csv
    - Modify get_feature_engineered_path to point to new file
    - Ensure all 42 features are loaded correctly
    - _Requirements: 6.8_

  - [ ] 8.2 Verify ML model compatibility with new features
    - Test isolation forest with account-level features
    - Test autoencoder with account-level features
    - _Requirements: 6.3_

- [ ]* 8.3 Write property test for account ownership validation
  - **Property 4: Account Ownership Validation**
  - **Validates: Requirements 4.4**

- [ ] 9. Integration and testing
  - [ ] 9.1 Test complete account-level workflow
    - Test login with multiple accounts
    - Test transaction processing with account selection
    - Test fraud detection with account-specific limits
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [ ] 9.2 Verify feature_datasetv2.csv generation
    - Run feature engineering and verify output file
    - Validate all 42 features are present and correctly calculated
    - _Requirements: 6.8, 9.1-9.7_

- [ ] 10. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Code should be clean, simple, and without excessive comments
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases