import numpy as np
from backend.rule_engine import check_rule_violation


def make_decision(txn, user_stats, model, features, autoencoder=None):
    result = {
        "is_fraud": False,
        "reasons": [],
        "risk_score": 0.0,
        "threshold": 0.0,
        "ml_flag": False,
        "ae_flag": False,
        "ae_reconstruction_error": None,
        "ae_threshold": None,
    }
    violated, rule_reasons, threshold = check_rule_violation(
        amount=txn["amount"],
        user_avg=user_stats["user_avg_amount"],
        user_std=user_stats["user_std_amount"],
        transfer_type=txn["transfer_type"],
        txn_count_10min=txn["txn_count_10min"],
        txn_count_1hour=txn["txn_count_1hour"],
        monthly_spending=user_stats["current_month_spending"],
    )

    result["threshold"] = threshold
    if violated:
        result["is_fraud"] = True
        result["reasons"].extend(rule_reasons)

    if model is not None:
        vec = np.array([[txn.get(f, 0) for f in features]])
        pred = model.predict(vec)[0]
        score = -model.decision_function(vec)[0]

        result["risk_score"] = score

        if pred == -1:
            result["ml_flag"] = True
            result["is_fraud"] = True
            result["reasons"].append(
                f"ML anomaly detected: abnormal behavior pattern (risk score {score:.4f})"
            )
    if autoencoder is not None:
        amount = txn.get('amount', 0)
        user_avg = user_stats.get('user_avg_amount', 5000)
        user_max = max(user_stats.get('user_max_amount', 1), 1)
        weekly_avg = user_stats.get('user_weekly_avg_amount', 0)
        monthly_avg = user_stats.get('monthly_avg_amount', user_avg)
        time_since_last = txn.get('time_since_last_txn', 3600)
        
        ae_features = { 
            'transaction_amount': amount,
            'flag_amount': 1 if txn.get('transfer_type') == 'S' else 0,
            'transfer_type_encoded': {'S': 4, 'I': 1, 'L': 2, 'Q': 3, 'O': 0}.get(txn.get('transfer_type', 'O'), 0),
            'transfer_type_risk': {'S': 0.9, 'I': 0.1, 'L': 0.2, 'Q': 0.5, 'O': 0.0}.get(txn.get('transfer_type', 'O'), 0.5),
            'channel_encoded': 0,
            'deviation_from_avg': abs(amount - user_avg),
            'amount_to_max_ratio': amount / user_max,
            'rolling_std': user_stats.get('user_std_amount', 0),
            'transaction_velocity': 3600 / max(time_since_last, 1),
            'weekly_total': user_stats.get('user_weekly_total', 0),           
            'weekly_txn_count': user_stats.get('user_weekly_txn_count', 0),       
            'weekly_avg_amount': weekly_avg,      
            'weekly_deviation': abs(amount - weekly_avg) if weekly_avg > 0 else 0,
            'amount_vs_weekly_avg': amount / max(weekly_avg, 1) if weekly_avg > 0 else 1,
            'current_month_spending': user_stats.get('current_month_spending', 0),
            'monthly_txn_count': user_stats.get('monthly_txn_count', user_stats.get('user_txn_frequency', 0)),
            'monthly_avg_amount': monthly_avg,
            'monthly_deviation': abs(amount - monthly_avg),
            'amount_vs_monthly_avg': amount / max(monthly_avg, 1),
            'hourly_total': amount,
            'hourly_count': 1,
            'daily_total': amount,
            'daily_count': 1,
            'hour': 12,  
            'day_of_week': 0,
            'is_weekend': 0,
            'is_night': 0,
            'time_since_last': time_since_last,
            'recent_burst': 1 if time_since_last < 300 else 0,
            'txn_count_30s': txn.get('txn_count_30s', 1),
            'txn_count_10min': txn.get('txn_count_10min', 1),
            'txn_count_1hour': txn.get('txn_count_1hour', 1),
            'user_avg_amount': user_avg,
            'user_std_amount': user_stats.get('user_std_amount', 0),
            'user_max_amount': user_stats.get('user_max_amount', 0),
            'user_txn_frequency': user_stats.get('user_txn_frequency', 0),
            'intl_ratio': user_stats.get('user_international_ratio', 0),
            'user_high_risk_txn_ratio': user_stats.get('user_high_risk_txn_ratio', 0.5),
            'user_multiple_accounts_flag': 1 if user_stats.get('num_accounts', 1) > 1 else 0,
            'cross_account_transfer_ratio': user_stats.get('cross_account_transfer_ratio', 0),
            'geo_anomaly_flag': 1 if txn.get('bank_country', 'UAE') not in ['UAE', 'United Arab Emirates'] else 0,
            'is_new_beneficiary': txn.get('is_new_beneficiary', 0),
            'beneficiary_txn_count_30d': user_stats.get('beneficiary_txn_count_30d', 1),
        }
        
        ae_result = autoencoder.score_transaction(ae_features)
        
        if ae_result is not None:
            result["ae_reconstruction_error"] = ae_result['reconstruction_error']
            result["ae_threshold"] = ae_result['threshold']
            
            if ae_result['is_anomaly']:
                result["ae_flag"] = True
                result["is_fraud"] = True
                result["reasons"].append(ae_result['reason'])

    return result
