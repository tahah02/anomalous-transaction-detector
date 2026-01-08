import pandas as pd
import numpy as np
from backend.utils import ensure_data_dir, get_clean_csv_path, TRANSFER_TYPE_ENCODED, TRANSFER_TYPE_RISK

def get_feature_datasetv2_path():
    return 'data/feature_datasetv2.csv'

def engineer_features():
    ensure_data_dir()
    df = pd.read_csv(get_clean_csv_path())
    
    if 'CreateDate' in df.columns:
        df['CreateDate'] = pd.to_datetime(df['CreateDate'], errors='coerce')

    df['transaction_amount'] = pd.to_numeric(df.get('AmountInAed', 0), errors='coerce').fillna(0)
    
    if 'TransferType' in df.columns:
        df['flag_amount'] = df['TransferType'].apply(lambda x: 1 if str(x).upper() == 'S' else 0)
        df['transfer_type_encoded'] = df['TransferType'].apply(lambda x: TRANSFER_TYPE_ENCODED.get(str(x).upper(), 0))
        df['transfer_type_risk'] = df['TransferType'].apply(lambda x: TRANSFER_TYPE_RISK.get(str(x).upper(), 0.5))
    else:
        df['flag_amount'], df['transfer_type_encoded'], df['transfer_type_risk'] = 0, 0, 0.5
    
    df['channel_encoded'] = 0
    if 'ChannelId' in df.columns:
        mapping = {v: i for i, v in enumerate(df['ChannelId'].dropna().unique())}
        df['channel_encoded'] = df['ChannelId'].map(mapping).fillna(0).astype(int)
    
    if 'CreateDate' in df.columns and df['CreateDate'].notna().any():
        df['hour'] = df['CreateDate'].dt.hour.fillna(12).astype(int)
        df['day_of_week'] = df['CreateDate'].dt.dayofweek.fillna(0).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] < 6) | (df['hour'] >= 22)).astype(int)
    else:
        df['hour'], df['day_of_week'], df['is_weekend'], df['is_night'] = 12, 0, 0, 0

    if 'CustomerId' in df.columns and 'FromAccountNo' in df.columns:
        account_key = ['CustomerId', 'FromAccountNo']
        
        stats = df.groupby(account_key)['transaction_amount'].agg(['mean', 'std', 'max', 'count'])
        stats.columns = ['user_avg_amount', 'user_std_amount', 'user_max_amount', 'user_txn_frequency']
        stats['user_std_amount'] = stats['user_std_amount'].fillna(0)
        df = df.merge(stats.reset_index(), on=account_key, how='left')
        
        df['deviation_from_avg'] = abs(df['transaction_amount'] - df['user_avg_amount'])
        df['amount_to_max_ratio'] = df['transaction_amount'] / df['user_max_amount'].replace(0, 1)

        if 'TransferType' in df.columns:
            df['intl_ratio'] = df.groupby(account_key)['flag_amount'].transform('mean')
        else:
            df['intl_ratio'] = 0

        df['user_high_risk_txn_ratio'] = df.groupby(account_key)['transfer_type_risk'].transform('mean')
        
        account_counts = df.groupby('CustomerId')['FromAccountNo'].nunique().reset_index()
        account_counts.columns = ['CustomerId', 'num_accounts']
        df = df.merge(account_counts, on='CustomerId', how='left')
        df['user_multiple_accounts_flag'] = (df['num_accounts'] > 1).astype(int)
        
        cross_account = df.groupby('CustomerId').apply(
            lambda x: (x['FromAccountNo'] != x['FromAccountNo'].iloc[0]).mean() if len(x) > 1 else 0
        ).reset_index()
        cross_account.columns = ['CustomerId', 'cross_account_transfer_ratio']
        df = df.merge(cross_account, on='CustomerId', how='left')
        
        df['geo_anomaly_flag'] = 0
        if 'BankCountry' in df.columns:
            country_counts = df.groupby(account_key)['BankCountry'].nunique()
            df = df.merge(country_counts.reset_index().rename(columns={'BankCountry': 'country_count'}), 
                         on=account_key, how='left')
            df['geo_anomaly_flag'] = (df['country_count'] > 2).astype(int)
            df.drop(columns=['country_count'], inplace=True)
        
        df['is_new_beneficiary'] = 0
        df['beneficiary_txn_count_30d'] = 1
        if 'ReceipentAccount' in df.columns and 'CreateDate' in df.columns:
            df = df.sort_values(account_key + ['CreateDate']).reset_index(drop=True)
            df['is_new_beneficiary'] = df.groupby(account_key)['ReceipentAccount'].transform(
                lambda x: (~x.duplicated()).astype(int)
            )
            
            def count_beneficiary_txns(group):
                result = []
                for i, (date, beneficiary) in enumerate(zip(group['CreateDate'], group['ReceipentAccount'])):
                    if pd.isna(date):
                        result.append(1)
                        continue
                    window_start = date - pd.Timedelta(days=30)
                    mask = (group['CreateDate'][:i+1] >= window_start) & (group['ReceipentAccount'][:i+1] == beneficiary)
                    result.append(mask.sum())
                return pd.Series(result, index=group.index)
            
            df['beneficiary_txn_count_30d'] = df.groupby(account_key, group_keys=False).apply(count_beneficiary_txns)

        if 'CreateDate' in df.columns and df['CreateDate'].notna().any():
            df = df.sort_values(account_key + ['CreateDate']).reset_index(drop=True)
            df['time_since_last'] = df.groupby(account_key)['CreateDate'].diff().dt.total_seconds().fillna(3600)
            df['recent_burst'] = (df['time_since_last'] < 300).astype(int)
            df['last_txn_time'] = df.groupby(account_key)['CreateDate'].shift(1)
            
            def count_txns_in_window(group, seconds):
                counts = []
                timestamps = group['CreateDate'].values
                for i, ts in enumerate(timestamps):
                    if pd.isna(ts):
                        counts.append(1)
                        continue
                    window_start = ts - np.timedelta64(seconds, 's')
                    count = np.sum((timestamps[:i] >= window_start) & (timestamps[:i] <= ts)) + 1
                    counts.append(count)
                return pd.Series(counts, index=group.index)
            
            df['txn_count_30s'] = df.groupby(account_key, group_keys=False).apply(lambda g: count_txns_in_window(g, 30))
            df['txn_count_10min'] = df.groupby(account_key, group_keys=False).apply(lambda g: count_txns_in_window(g, 600))
            df['txn_count_1hour'] = df.groupby(account_key, group_keys=False).apply(lambda g: count_txns_in_window(g, 3600))

            df['hour_key'] = df['CreateDate'].dt.floor('H')
            hourly_stats = df.groupby(account_key + ['hour_key'])['transaction_amount'].agg(['sum', 'count'])
            hourly_stats.columns = ['hourly_total', 'hourly_count']
            df = df.merge(hourly_stats.reset_index(), on=account_key + ['hour_key'], how='left')
            df.drop(columns=['hour_key'], inplace=True)

            df['day_key'] = df['CreateDate'].dt.floor('D')
            daily_stats = df.groupby(account_key + ['day_key'])['transaction_amount'].agg(['sum', 'count'])
            daily_stats.columns = ['daily_total', 'daily_count']
            df = df.merge(daily_stats.reset_index(), on=account_key + ['day_key'], how='left')
            df.drop(columns=['day_key'], inplace=True)
            
            df['week_key'] = df['CreateDate'].dt.to_period('W')
            weekly_stats = df.groupby(account_key + ['week_key'])['transaction_amount'].agg(['sum', 'count', 'mean'])
            weekly_stats.columns = ['weekly_total', 'weekly_txn_count', 'weekly_avg_amount']
            df = df.merge(weekly_stats.reset_index(), on=account_key + ['week_key'], how='left')
            df['weekly_deviation'] = abs(df['transaction_amount'] - df['weekly_avg_amount'])
            df['amount_vs_weekly_avg'] = df['transaction_amount'] / df['weekly_avg_amount'].replace(0, 1)
            df.drop(columns=['week_key'], inplace=True)
            
            df['month_key'] = df['CreateDate'].dt.to_period('M')
            monthly_stats = df.groupby(account_key + ['month_key'])['transaction_amount'].agg(['sum', 'count', 'mean'])
            monthly_stats.columns = ['current_month_spending', 'monthly_txn_count', 'monthly_avg_amount']
            df = df.merge(monthly_stats.reset_index(), on=account_key + ['month_key'], how='left')
            df['monthly_deviation'] = abs(df['transaction_amount'] - df['monthly_avg_amount'])
            df['amount_vs_monthly_avg'] = df['transaction_amount'] / df['monthly_avg_amount'].replace(0, 1)
            df.drop(columns=['month_key'], inplace=True)
            
        else:
            df['time_since_last'], df['recent_burst'] = 3600, 0
            df['last_txn_time'] = None
            df['txn_count_30s'], df['txn_count_10min'], df['txn_count_1hour'] = 1, 1, 1
            df['hourly_total'], df['hourly_count'] = df['transaction_amount'], 1
            df['daily_total'], df['daily_count'] = df['transaction_amount'], 1
            df['weekly_total'], df['weekly_txn_count'], df['weekly_avg_amount'] = df['transaction_amount'], 1, df['transaction_amount']
            df['weekly_deviation'], df['amount_vs_weekly_avg'] = 0, 1
            df['current_month_spending'], df['monthly_txn_count'], df['monthly_avg_amount'] = df['transaction_amount'], 1, df['transaction_amount']
            df['monthly_deviation'], df['amount_vs_monthly_avg'] = 0, 1
            
        df['rolling_std'] = df.groupby(account_key)['transaction_amount'].transform(
            lambda x: x.rolling(window=min(5, len(x)), min_periods=1).std()
        ).fillna(0)
        
        df['transaction_velocity'] = df.groupby(account_key)['time_since_last'].transform(
            lambda x: 1 / (x.replace(0, 1) / 3600)
        ).fillna(0)
        
    else:
        df['user_avg_amount'] = df['transaction_amount'].mean()
        df['user_std_amount'] = df['transaction_amount'].std()
        df['user_max_amount'] = df['transaction_amount'].max()
        df['user_txn_frequency'] = len(df)
        df['deviation_from_avg'], df['amount_to_max_ratio'] = 0, 0
        df['intl_ratio'] = 0
        df['user_high_risk_txn_ratio'] = 0
        df['user_multiple_accounts_flag'] = 0
        df['cross_account_transfer_ratio'] = 0
        df['geo_anomaly_flag'] = 0
        df['is_new_beneficiary'] = 0
        df['beneficiary_txn_count_30d'] = 1
        df['time_since_last'], df['recent_burst'] = 3600, 0
        df['last_txn_time'] = None
        df['txn_count_30s'], df['txn_count_10min'], df['txn_count_1hour'] = 1, 1, 1
        df['hourly_total'], df['hourly_count'] = df['transaction_amount'], 1
        df['daily_total'], df['daily_count'] = df['transaction_amount'], 1
        df['weekly_total'], df['weekly_txn_count'], df['weekly_avg_amount'] = df['transaction_amount'], 1, df['transaction_amount']
        df['weekly_deviation'], df['amount_vs_weekly_avg'] = 0, 1
        df['current_month_spending'], df['monthly_txn_count'], df['monthly_avg_amount'] = df['transaction_amount'], 1, df['transaction_amount']
        df['monthly_deviation'], df['amount_vs_monthly_avg'] = 0, 1
        df['rolling_std'] = 0
        df['transaction_velocity'] = 0

    output_path = get_feature_datasetv2_path()
    df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

if __name__ == "__main__":
    engineer_features()
