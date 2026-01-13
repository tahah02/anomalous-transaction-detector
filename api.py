from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional
import pandas as pd
import numpy as np
import json
import os
from backend.hybrid_decision import make_decision
from backend.utils import get_feature_engineered_path, load_model
from backend.autoencoder import AutoencoderInference
from backend.rule_engine import calculate_all_limits

app = FastAPI(title="Banking Fraud Detection API", version="1.0.0")

class TransactionRequest(BaseModel):
    customer_id: str
    from_account_no: str
    to_account_no: str
    transaction_amount: float = Field(gt=0)
    transfer_type: str = Field(pattern="^[SILQO]$")
    datetime: datetime
    bank_country: Optional[str] = "UAE"

class TransactionResponse(BaseModel):
    decision: str
    risk_score: float
    confidence_level: float
    reasons: List[str]
    individual_scores: dict
    transaction_id: str
    processing_time_ms: int


class UserStatsManager:
    def __init__(self):
        self.stats_file = "data/user_stats.json"
        self.velocity_file = "data/velocity_counters.json"
        self.stats = self.load_stats()
        self.velocity = self.load_velocity()
        self.historical_data = self.load_historical_data()
    
    def load_historical_data(self):
        try:
            path = get_feature_engineered_path()
            if os.path.exists(path):
                return pd.read_csv(path)
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading historical data: {e}")
            return pd.DataFrame()
    
    def load_stats(self):
        if os.path.exists(self.stats_file):
            with open(self.stats_file, 'r') as f:
                return json.load(f)
        return {}
    
    def load_velocity(self):
        if os.path.exists(self.velocity_file):
            with open(self.velocity_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_stats(self):
        os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f)
        with open(self.velocity_file, 'w') as f:
            json.dump(self.velocity, f)
    
    def get_user_stats(self, customer_id: str, account_no: str):
        key = f"{customer_id}_{account_no}"
        
        if key in self.stats:
            # OPTIONAL: Ensure old cached stats get new keys if missing
            if "user_weekly_total" not in self.stats[key]:
                self.stats[key].update({
                    "user_transaction_velocity": 0,
                    "user_weekly_total": 0.0,
                    "user_weekly_txn_count": 0,
                    "user_weekly_avg_amount": 0.0,
                    "user_weekly_deviation": 0.0
                })
            return self.stats[key]
        
        if not self.historical_data.empty:
            amt_col = 'transaction_amount' if 'transaction_amount' in self.historical_data.columns else 'AmountInAed'
            cust_data = self.historical_data[self.historical_data['CustomerId'].astype(str) == str(customer_id)]
            
            if 'FromAccountNo' in cust_data.columns:
                account_data = cust_data[cust_data['FromAccountNo'].astype(str) == str(account_no)]
            else:
                account_data = cust_data
            
            if len(account_data) > 0:
                overall_avg = account_data[amt_col].mean()
                overall_std = account_data[amt_col].std() if len(account_data) > 1 else 2000.0
                overall_max = account_data[amt_col].max()
                total_txns_count = len(account_data)
                
                intl_ratio = 0.0
                if 'TransferType' in account_data.columns and total_txns_count > 0:
                    count_s = len(account_data[account_data['TransferType'] == 'S'])
                    intl_ratio = count_s / total_txns_count
                
                current_month_spending = self.get_monthly_spending_from_csv(customer_id, account_no)
                
                self.stats[key] = {
                    "user_avg_amount": float(overall_avg),
                    "user_std_amount": float(overall_std),
                    "user_max_amount": float(overall_max),
                    "user_txn_frequency": int(total_txns_count),
                    "user_international_ratio": float(intl_ratio),
                    "current_month_spending": float(current_month_spending),
                    # --- ADDED NEW KEYS (Defaulting to 0 for now) ---
                    "user_transaction_velocity": 0,
                    "user_weekly_total": 0.0,
                    "user_weekly_txn_count": 0,
                    "user_weekly_avg_amount": 0.0,
                    "user_weekly_deviation": 0.0
                }
            else:
                self.stats[key] = {
                    "user_avg_amount": 5000.0,
                    "user_std_amount": 2000.0,
                    "user_max_amount": 15000.0,
                    "user_txn_frequency": 0,
                    "user_international_ratio": 0.0,
                    "current_month_spending": 0.0,
                    # --- ADDED NEW KEYS ---
                    "user_transaction_velocity": 0,
                    "user_weekly_total": 0.0,
                    "user_weekly_txn_count": 0,
                    "user_weekly_avg_amount": 0.0,
                    "user_weekly_deviation": 0.0
                }
        else:
            self.stats[key] = {
                "user_avg_amount": 5000.0,
                "user_std_amount": 2000.0,
                "user_max_amount": 15000.0,
                "user_txn_frequency": 0,
                "user_international_ratio": 0.0,
                "current_month_spending": 0.0,
                # --- ADDED NEW KEYS ---
                "user_transaction_velocity": 0,
                "user_weekly_total": 0.0,
                "user_weekly_txn_count": 0,
                "user_weekly_avg_amount": 0.0,
                "user_weekly_deviation": 0.0
            }
        
        return self.stats[key]
    
    def get_monthly_spending_from_csv(self, customer_id: str, account_no: str):
        if self.historical_data.empty:
            return 0.0
            
        current_month = datetime.now().month
        current_year = datetime.now().year
        
        amt_col = 'transaction_amount' if 'transaction_amount' in self.historical_data.columns else 'AmountInAed'
        cust_data = self.historical_data[self.historical_data['CustomerId'].astype(str) == str(customer_id)]
        
        if 'CreateDate' in cust_data.columns and len(cust_data) > 0:
            cust_data = cust_data.copy()
            if cust_data['CreateDate'].dtype == 'object':
                cust_data['CreateDate'] = pd.to_datetime(cust_data['CreateDate'], errors='coerce')
            
            if 'FromAccountNo' in cust_data.columns:
                account_data = cust_data[cust_data['FromAccountNo'].astype(str) == str(account_no)]
            else:
                account_data = cust_data
                
            monthly_data = account_data[
                (account_data['CreateDate'].dt.month == current_month) & 
                (account_data['CreateDate'].dt.year == current_year)
            ]
            return monthly_data[amt_col].sum() if len(monthly_data) > 0 else 0.0
        return 0.0
    
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
    
    def check_is_new_beneficiary(self, customer_id: str, recipient_account: str):
        if self.historical_data.empty:
            return 1 # Default to new if no history
            
        cust_data = self.historical_data[self.historical_data['CustomerId'].astype(str) == str(customer_id)]
        
        if 'ReceipentAccount' not in cust_data.columns and 'RecipientAccount' not in cust_data.columns:
            return 0 # Cannot verify, assume safe/old to avoid false positives if column missing
            
        col_name = 'ReceipentAccount' if 'ReceipentAccount' in cust_data.columns else 'RecipientAccount'
        
        path_recipients = cust_data[col_name].astype(str).unique()
        return 0 if str(recipient_account) in path_recipients else 1

    def record_transaction(self, customer_id: str, account_no: str, amount: float):
        key = f"{customer_id}_{account_no}"
        now = datetime.now()
        
        if key not in self.velocity:
            self.velocity[key] = []
        self.velocity[key].append(now.isoformat())
        
        stats = self.get_user_stats(customer_id, account_no)
        stats["user_txn_frequency"] += 1
        
        session_key = f"{key}_session_spending"
        if session_key not in self.stats:
            self.stats[session_key] = 0.0
        self.stats[session_key] += amount
        
        csv_monthly = self.get_monthly_spending_from_csv(customer_id, account_no)
        stats["current_month_spending"] = csv_monthly + self.stats[session_key]
        
        self.save_stats()
    
    def save_transaction_history(self, request, decision: str, result: dict, transaction_id: str):
        history_file = "data/transaction_history.csv"
        
        # Simple format with only essential fields
        txn_record = {
            "transaction_id": transaction_id,
            "customer_id": request.customer_id,
            "account_no": request.from_account_no,
            "amount": request.transaction_amount,
            "status": decision,
            "reasons": "|".join(result.get('reasons', [])) if result.get('reasons') else "Normal transaction"
        }
        
        df_new = pd.DataFrame([txn_record])
        
        if os.path.exists(history_file):
            df_new.to_csv(history_file, mode='a', header=False, index=False)
        else:
            df_new.to_csv(history_file, index=False)

stats_manager = UserStatsManager()
model, features, scaler = load_model()
autoencoder = AutoencoderInference()
autoencoder.load()


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

@app.post("/api/analyze-transaction", response_model=TransactionResponse)
def analyze_transaction(request: TransactionRequest):
    start_time = datetime.now()
    
    user_stats = stats_manager.get_user_stats(request.customer_id, request.from_account_no)
    velocity = stats_manager.get_velocity_metrics(request.customer_id, request.from_account_no)
    
    is_new_ben = stats_manager.check_is_new_beneficiary(request.customer_id, request.to_account_no)
    
    txn = {
        "amount": request.transaction_amount,
        "transfer_type": request.transfer_type,
        "bank_country": request.bank_country,
        "txn_count_30s": 1,  # Default to 1 for current transaction
        "txn_count_10min": velocity["txn_count_10min"] + 1,
        "txn_count_1hour": velocity["txn_count_1hour"] + 1,
        "time_since_last_txn": velocity["time_since_last_txn"],
        "is_new_beneficiary": is_new_ben
    }
    
    result = make_decision(txn, user_stats, model, features, autoencoder)
    
    decision = "REQUIRES_USER_APPROVAL" if result['is_fraud'] else "APPROVED"
    
    processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
    transaction_id = f"txn_{request.datetime.strftime('%Y%m%d_%H%M%S')}_{request.customer_id}"
    
    # Save transaction history (both APPROVED and REQUIRES_USER_APPROVAL)
    stats_manager.save_transaction_history(request, decision, result, transaction_id)
    
    if decision == "APPROVED":
        stats_manager.record_transaction(request.customer_id, request.from_account_no, request.transaction_amount)
    
    return TransactionResponse(
        decision=decision,
        risk_score=result.get('risk_score', 0.0),
        confidence_level=0.85,
        reasons=result.get('reasons', []),
        individual_scores={
            "rule_engine": {"violated": result['is_fraud'], "threshold": result.get('threshold', 0)},
            "isolation_forest": {"anomaly_score": result.get('risk_score', 0), "is_anomaly": result.get('ml_flag', False)},
            "autoencoder": {"reconstruction_error": result.get('ae_reconstruction_error'), "is_anomaly": result.get('ae_flag', False)}
        },
        transaction_id=transaction_id,
        processing_time_ms=processing_time
    )
