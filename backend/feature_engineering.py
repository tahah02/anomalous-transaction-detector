import pandas as pd
import numpy as np
from backend.utils import (ensure_data_dir, get_clean_csv_path,
                          TRANSFER_TYPE_ENCODED, TRANSFER_TYPE_RISK)

OUTPUT_PATH = 'data/feature_datasetv2.csv'

def engineer_features():
    ensure_data_dir()
    df = pd.read_csv(get_clean_csv_path())
    
    df['CreateDate'] = pd.to_datetime(df.get('CreateDate'), errors='coerce')
    df['transaction_amount'] = pd.to_numeric(df.get('AmountInAed', 0), errors='coerce').fillna(0)
    
    if 'TransferType' in df:
        tt = df['TransferType'].astype(str).str.upper()
        df['flag_amount'] = (tt == 'S').astype(int)
        df['transfer_type_encoded'] = tt.map(TRANSFER_TYPE_ENCODED).fillna(0)
        df['transfer_type_risk'] = tt.map(TRANSFER_TYPE_RISK).fillna(0.5)
    else:
        df[['flag_amount','transfer_type_encoded','transfer_type_risk']] = [0,0,0.5]

    df['channel_encoded'] = 0
    if 'ChannelId' in df:
        df['channel_encoded'] = (df['ChannelId'].map({v:i for i,v in enumerate(df['ChannelId'].dropna().unique())})
                                 .fillna(0).astype(int))

    if df['CreateDate'].notna().any():
        df['hour'] = df['CreateDate'].dt.hour.fillna(12).astype(int)
        df['day_of_week'] = df['CreateDate'].dt.dayofweek.fillna(0).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] < 6) | (df['hour'] >= 22)).astype(int)
    else:
        df[['hour','day_of_week','is_weekend','is_night']] = [12,0,0,0]

    if {'CustomerId','FromAccountNo'}.issubset(df.columns):
        key = ['CustomerId','FromAccountNo']
        df = df.sort_values(key + ['CreateDate'])
        
        stats = df.groupby(key)['transaction_amount'].agg(['mean','std','max','count'])
        stats.columns = ['user_avg_amount','user_std_amount','user_max_amount','user_txn_frequency']
        df = df.merge(stats.fillna(0).reset_index(), on=key, how='left')
        
        df['deviation_from_avg'] = abs(df['transaction_amount'] - df['user_avg_amount'])
        df['amount_to_max_ratio'] = df['transaction_amount'] / df['user_max_amount'].replace(0,1)
        df['intl_ratio'] = df.groupby(key)['flag_amount'].transform('mean')
        df['user_high_risk_txn_ratio'] = df.groupby(key)['transfer_type_risk'].transform('mean')
        
        acc_cnt = df.groupby('CustomerId')['FromAccountNo'].nunique()
        df = df.merge(acc_cnt.rename('num_accounts'), on='CustomerId', how='left')
        df['user_multiple_accounts_flag'] = (df['num_accounts'] > 1).astype(int)
        
        cross = df.groupby('CustomerId').apply(lambda x: (x['FromAccountNo'] != x['FromAccountNo'].iloc[0]).mean())
        df = df.merge(cross.rename('cross_account_transfer_ratio'), on='CustomerId', how='left')

        df['geo_anomaly_flag'] = 0
        if 'BankCountry' in df:
            cc = df.groupby(key)['BankCountry'].nunique()
            df = df.merge(cc.rename('country_count'), on=key, how='left')
            df['geo_anomaly_flag'] = (df['country_count'] > 2).astype(int)
            df.drop(columns='country_count', inplace=True)

        df['is_new_beneficiary'] = 0
        df['beneficiary_txn_count_30d'] = 1
        if {'ReceipentAccount','CreateDate'}.issubset(df.columns):
            df['is_new_beneficiary'] = df.groupby(key)['ReceipentAccount'].transform(lambda x: (~x.duplicated()).astype(int))
      
        df['time_since_last'] = df.groupby(key)['CreateDate'].diff().dt.total_seconds().fillna(3600)
        df['recent_burst'] = (df['time_since_last'] < 300).astype(int)
        
        def rolling_count(g, sec):
            t = g['CreateDate'].values
            return pd.Series([np.sum((t[:i] >= ts - np.timedelta64(sec,'s')) & (t[:i] <= ts)) + 1
                             if not pd.isna(ts) else 1
                             for i, ts in enumerate(t)], index=g.index)
        
        for s, col in [(30,'txn_count_30s'),(600,'txn_count_10min'),(3600,'txn_count_1hour')]:
            df[col] = df.groupby(key, group_keys=False).apply(lambda g: rolling_count(g, s))
        for freq, cols in [('H', ('hourly_total','hourly_count')),
                          ('D', ('daily_total','daily_count'))]:
            k = df['CreateDate'].dt.floor(freq)
            agg = df.groupby(key + [k])['transaction_amount'].agg(['sum','count'])
            agg.columns = cols
            df = df.merge(agg.reset_index(), on=key + [k.name], how='left').drop(columns=k.name)
        
        df['week_key'] = df['CreateDate'].dt.to_period('W')
        wk = df.groupby(key+['week_key'])['transaction_amount'].agg(['sum','count','mean'])
        wk.columns = ['weekly_total','weekly_txn_count','weekly_avg_amount']
        df = df.merge(wk.reset_index(), on=key+['week_key'], how='left').drop(columns='week_key')
        df['weekly_deviation'] = abs(df['transaction_amount'] - df['weekly_avg_amount'])
        df['amount_vs_weekly_avg'] = df['transaction_amount'] / df['weekly_avg_amount'].replace(0,1)
        
        df['month_key'] = df['CreateDate'].dt.to_period('M')
        mo = df.groupby(key+['month_key'])['transaction_amount'].agg(['sum','count','mean'])
        mo.columns = ['current_month_spending','monthly_txn_count','monthly_avg_amount']
        df = df.merge(mo.reset_index(), on=key+['month_key'], how='left').drop(columns='month_key')
        df['monthly_deviation'] = abs(df['transaction_amount'] - df['monthly_avg_amount'])
        df['amount_vs_monthly_avg'] = df['transaction_amount'] / df['monthly_avg_amount'].replace(0,1)
        
        df['rolling_std'] = df.groupby(key)['transaction_amount'].transform(lambda x: x.rolling(min(5,len(x)),1).std()).fillna(0)
        df['transaction_velocity'] = (1 / (df['time_since_last'].replace(0,1) / 3600)).fillna(0)
    
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH} | {df.shape}")
    return df

if __name__ == "__main__":
    engineer_features()
