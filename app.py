import streamlit as st
import pandas as pd
import os
from datetime import datetime
import csv

from backend.utils import get_feature_engineered_path, get_model_path, load_model
from backend.hybrid_decision import make_decision
from backend.rule_engine import calculate_all_limits
from backend.autoencoder import AutoencoderInference

def save_transaction_to_csv(cid, amount, t_type, status="Approved"):
    file_name = 'transaction_history.csv'
    file_exists = os.path.isfile(file_name)
    
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['CustomerID', 'Amount', 'Type', 'Status', 'Timestamp'])
        writer.writerow([cid, amount, t_type, status, datetime.now()])

st.set_page_config(page_title="Banking Fraud Detection", layout="wide")

def init_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'customer_id' not in st.session_state:
        st.session_state.customer_id = None
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'txn_history' not in st.session_state:
        st.session_state.txn_history = {}  
    if 'session_count' not in st.session_state:
        st.session_state.session_count = {}  
    if 'monthly_spending' not in st.session_state:
        st.session_state.monthly_spending = {}  

def get_velocity(cid, account_no):
    account_key = f"{cid}_{account_no}"
    now = datetime.now()
    
    history = st.session_state.txn_history.get(account_key, [])
    
    count_10min = sum(1 for t in history if (now - t).total_seconds() < 600)
    count_1hour = sum(1 for t in history if (now - t).total_seconds() < 3600)
    
    if history:
        time_since_last = (now - max(history)).total_seconds()
    else:
        time_since_last = 3600 
    
    return {
        'txn_count_10min': count_10min,
        'txn_count_1hour': count_1hour,
        'time_since_last_txn': time_since_last
    }

def record_transaction(cid, account_no):
    account_key = f"{cid}_{account_no}"
    if account_key not in st.session_state.txn_history:
        st.session_state.txn_history[account_key] = []
    st.session_state.txn_history[account_key].append(datetime.now())
    
    if account_key not in st.session_state.session_count:
        st.session_state.session_count[account_key] = 0
    st.session_state.session_count[account_key] += 1

def add_monthly_spending(cid, account_no, amount):
    account_key = f"{cid}_{account_no}"
    if account_key not in st.session_state.monthly_spending:
        st.session_state.monthly_spending[account_key] = 0.0
    st.session_state.monthly_spending[account_key] += amount

@st.cache_data
def load_data():
    path = get_feature_engineered_path()
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        st.error(f"Data file not found: {path}")
        st.info("Run `python -m backend.feature_engineering` first.")
        return None

@st.cache_resource
def get_model():
    if os.path.exists(get_model_path()):
        return load_model()
    else:
        st.error(f"Model not found: {get_model_path()}")
        st.info("Run `python -m backend.model_training` first.")
        return None, None

@st.cache_resource
def get_autoencoder():
    ae = AutoencoderInference()
    if ae.load():
        return ae
    else:
        return None

def get_monthly_spending_from_csv(cust_data, account_no, amt_col):
    current_month = datetime.now().month
    current_year = datetime.now().year
    
    if 'CreateDate' in cust_data.columns and len(cust_data) > 0:
        cust_data = cust_data.copy()
        if cust_data['CreateDate'].dtype == 'object':
            cust_data['CreateDate'] = pd.to_datetime(cust_data['CreateDate'], errors='coerce')
        account_data = cust_data[cust_data['FromAccountNo'].astype(str) == str(account_no)]
        monthly_data = account_data[
            (account_data['CreateDate'].dt.month == current_month) & 
            (account_data['CreateDate'].dt.year == current_year)
        ]
        return monthly_data[amt_col].sum() if len(monthly_data) > 0 else 0.0
    return 0.0

def login_page(df):
    st.title("🏦 Banking Fraud Detection System")
    st.subheader("Login")
    
    customers = sorted([str(c) for c in df['CustomerId'].dropna().unique()])
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### Select Customer ID")
        cid = st.selectbox("Customer ID", customers)
        pwd = st.text_input("Password", type="password")
        
        if st.button("Login", type="primary", use_container_width=True):
            if pwd == "12345":
                st.session_state.logged_in = True
                st.session_state.customer_id = cid
                st.rerun()
            else:
                st.error("Invalid password")
        
        st.info("Password: 12345")


def dashboard(df, model, features, scaler=None, autoencoder=None):
    cid = str(st.session_state.customer_id)
    amt_col = 'transaction_amount' if 'transaction_amount' in df.columns else 'AmountInAed'
    cust_data = df[df['CustomerId'].astype(str) == cid]

    st.title("Fraud Detection Dashboard")
    
    st.subheader("Step 1: Select Account")
    accounts = [str(a) for a in cust_data['FromAccountNo'].dropna().unique()] if 'FromAccountNo' in cust_data.columns and len(cust_data) > 0 else ["Default"]
    
    if len(accounts) == 0:
        st.error("No accounts found for this customer")
        return
        
    account = st.selectbox("Select Account", accounts, label_visibility="collapsed")
    
    account_data = cust_data[cust_data['FromAccountNo'].astype(str) == str(account)]
    
    csv_count = len(account_data)
    account_key = f"{cid}_{account}"
    session_count = st.session_state.session_count.get(account_key, 0)
    total_txns = csv_count + session_count

    vel = get_velocity(cid, account)

    st.sidebar.title("Navigation")
    st.sidebar.markdown(f"**Customer ID:** {cid}")
    st.sidebar.markdown(f"**Account:** {account}")
    
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.customer_id = None
        st.session_state.result = None
        st.rerun()
    
    st.sidebar.markdown("---")
    
    if autoencoder is not None:
        st.sidebar.success(" Autoencoder: Active")
    else:
        st.sidebar.warning(" Autoencoder: Unavailable")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Account Statistics")
    
    if len(account_data) > 0:
        avg = account_data[amt_col].mean()
        max_amt = account_data[amt_col].max()
        std = account_data[amt_col].std() if len(account_data) > 1 else 0
        
        st.sidebar.markdown(f"**Average Transaction:** AED {avg:,.2f}")
        st.sidebar.markdown(f"**Max Transaction:** AED {max_amt:,.2f}")
        st.sidebar.metric("Total Transactions", total_txns, 
                         delta=f"+{session_count} this session" if session_count > 0 else None)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Current Velocity")
        st.sidebar.markdown(f"**Last 10 min:** {vel['txn_count_10min']} transactions")
        st.sidebar.markdown(f"**Last 1 hour:** {vel['txn_count_1hour']} transactions")
        
        csv_monthly = get_monthly_spending_from_csv(cust_data, account, amt_col)
        session_monthly = st.session_state.monthly_spending.get(account_key, 0.0)
        total_monthly = csv_monthly + session_monthly
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Monthly Spending")
        st.sidebar.markdown(f"**This Month:** AED {total_monthly:,.2f}")
        st.sidebar.markdown("---")
        st.sidebar.subheader("Transfer Type Limits")
        limits = calculate_all_limits(avg, std)
        st.sidebar.markdown(f"**O - Own Account:** AED {limits['O']:,.2f}")
        st.sidebar.markdown(f"**I - Ajman:** AED {limits['I']:,.2f}")
        st.sidebar.markdown(f"**L - UAE:** AED {limits['L']:,.2f}")
        st.sidebar.markdown(f"**Q - Quick:** AED {limits['Q']:,.2f}")
        st.sidebar.markdown(f"**S - Overseas:** AED {limits['S']:,.2f}")
    else:
        avg, std, max_amt = 5000, 2000, 15000
        st.sidebar.warning("No transaction history for this account")

    st.markdown("---")

    st.subheader("Step 2: Transaction Details")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("**Transaction Amount (AED)**")
        amount = st.number_input("", min_value=0.0, max_value=1000000.0, value=1000.0, step=100.0, key="amt_input")
    
    with c2:
        st.markdown("**Transfer Type**")
        types = {
            'O': 'O - Own Account', 
            'I': 'I - Within Ajman', 
            'L': 'L - Within UAE', 
            'Q': 'Q - Quick Remittance', 
            'S': 'S - Overseas'
        }
        t_type = st.selectbox("", list(types.keys()), format_func=lambda x: types[x], key="type_input")
    
    with c3:
        st.markdown("**Bank Country**")
        countries = ['UAE', 'USA', 'UK', 'India', 'Pakistan', 'Philippines', 'Egypt', 'Other']
        country = st.selectbox("", countries, key="country_input")
    
    st.markdown("---")
    
    st.subheader("Step 3: Process Transaction")
 
    current_vel = get_velocity(cid, account)
    st.caption(f" Debug: Current recorded txns - 10min: {current_vel['txn_count_10min']}, 1hour: {current_vel['txn_count_1hour']}")
    
    if st.button("Process Transaction", type="primary", use_container_width=True):
        current_vel = get_velocity(cid, account)
        
        txn_count_10min = current_vel['txn_count_10min'] + 1
        txn_count_1hour = current_vel['txn_count_1hour'] + 1
        
        csv_monthly = get_monthly_spending_from_csv(cust_data, account, amt_col)
        session_monthly = st.session_state.monthly_spending.get(account_key, 0.0)
        current_spending = csv_monthly + session_monthly
        overall_avg = account_data[amt_col].mean() if len(account_data) > 0 else 5000
        overall_std = account_data[amt_col].std() if len(account_data) > 1 else 2000
        overall_max = account_data[amt_col].max() if len(account_data) > 0 else 15000

        total_txns_count = len(account_data)
        intl_ratio = 0.0
        if total_txns_count > 0:
            count_s = len(account_data[account_data['TransferType'] == 'S'])
            intl_ratio = count_s / total_txns_count

        user_stats = {
            'user_avg_amount': overall_avg,
            'user_std_amount': overall_std,
            'user_max_amount': overall_max,
            'user_txn_frequency': total_txns_count,
            'user_international_ratio': intl_ratio,
            'current_month_spending': current_spending
        }

        txn = {
            'amount': amount, 
            'transfer_type': t_type, 
            'bank_country': country,
            'account': account,
            'txn_count_10min': txn_count_10min,
            'txn_count_1hour': txn_count_1hour,
            'time_since_last_txn': current_vel['time_since_last_txn']
        }

        st.session_state.result = make_decision(txn, user_stats, model, features, autoencoder=autoencoder)
        st.session_state.result['amount'] = amount
        st.session_state.result['t_type'] = t_type
        st.session_state.result['account'] = account
        st.session_state.result['txn_count_10min'] = txn_count_10min
        st.session_state.result['txn_count_1hour'] = txn_count_1hour
        st.rerun()

    st.markdown("---")
    
    st.subheader("Step 4: Transaction Result")

    if st.session_state.result:
        r = st.session_state.result
        t_type = r.get('t_type', 'O')
        result_account = r.get('account', account)

        st.info(f" Velocity: {r.get('txn_count_10min', 0)} txns in 10min | {r.get('txn_count_1hour', 0)} txns in 1hour")
        
        if r.get('ae_reconstruction_error') is not None:
            ae_status = " Anomaly" if r.get('ae_flag', False) else " Normal"
            st.info(f" AE Error: {r['ae_reconstruction_error']:.4f} | Threshold: {r.get('ae_threshold', 'N/A'):.4f} | Status: {ae_status}")
        
        if r['is_fraud']:
            st.error(" FRAUD ALERT - Transaction Flagged!")

            for reason in r['reasons']:
                if isinstance(reason, str):
                    st.warning(reason)
                elif isinstance(reason, list):
                    for r_item in reason:
                        st.warning(r_item)
            
            st.markdown(f"**Risk Score:** {r['risk_score']:.4f}")
            
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Approve (Force)", type="primary"):
                    record_transaction(cid, result_account)
                    add_monthly_spending(cid, result_account, r['amount'])
                    save_transaction_to_csv(cid, r['amount'], t_type, "Force Approved")
                    st.success("Approved & Saved!")
                    st.session_state.result = None
                    st.rerun()
                    
            with c2:
                if st.button("Reject"):
                    save_transaction_to_csv(cid, r['amount'], t_type, "Rejected")
                    st.error("Rejected!")
                    st.session_state.result = None
                    st.rerun()

        else:
            st.success(" SAFE TRANSACTION")
            st.info(f"Amount: AED {r['amount']:,.2f} | Threshold: AED {r['threshold']:,.2f}")

            if st.button("Confirm & Continue", type="primary"):
                record_transaction(cid, result_account)
                add_monthly_spending(cid, result_account, r['amount'])
                save_transaction_to_csv(cid, r['amount'], t_type, "Approved")
                st.session_state.result = None
                st.rerun()

def main():
    init_state()
    
    df = load_data()
    if df is None:
        return
    
    model, features, scaler = get_model()
    if model is None:
        return
    autoencoder = get_autoencoder()
    
    if st.session_state.logged_in:
        dashboard(df, model, features, scaler=scaler, autoencoder=autoencoder)
    else:
        login_page(df)

if __name__ == "__main__":
    main()
