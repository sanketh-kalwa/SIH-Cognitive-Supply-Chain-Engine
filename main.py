import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import base64
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Cognitive Supply Chain Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MockDatabase:
    """A mock database class to simulate data storage using session state."""
    def __init__(self):
        if 'db_data' not in st.session_state:
            st.session_state.db_data = {
                'projects': pd.DataFrame(),
                'inventory': pd.DataFrame(),
                'historical_demand': pd.DataFrame(),
                'price_history': pd.DataFrame(),
                'suppliers': pd.DataFrame(),
                'activity_logs': pd.DataFrame(columns=['timestamp', 'activity', 'user_name', 'status', 'details']),
                'alerts': pd.DataFrame(columns=['id', 'material_name', 'alert_type', 'severity', 'message', 'created_at', 'resolved_at', 'status']),
                'trained_models': pd.DataFrame()
            }
        self.is_available = True

    def get_projects(self): return st.session_state.db_data['projects'].copy()
    def save_projects(self, df): st.session_state.db_data['projects'] = df
    
    def get_inventory(self): return st.session_state.db_data['inventory'].copy()
    def save_inventory(self, df): st.session_state.db_data['inventory'] = df
    
    def get_historical_demand(self): return st.session_state.db_data['historical_demand'].copy()
    def save_historical_demand(self, df): st.session_state.db_data['historical_demand'] = df

    def get_active_projects(self):
        projects = self.get_projects()
        if projects.empty or 'status' not in projects.columns: return []
        return projects[projects['status'].isin(['Planning', 'In Progress'])].to_dict('records')

    def log_activity(self, activity, user_name='System', status='Success', details=None):
        new_log = pd.DataFrame([{'timestamp': datetime.now(), 'activity': activity, 'user_name': user_name, 'status': status, 'details': json.dumps(details) if details else None}])
        st.session_state.db_data['activity_logs'] = pd.concat([st.session_state.db_data['activity_logs'], new_log], ignore_index=True)

    def get_activity_logs(self, limit=10):
        logs = st.session_state.db_data['activity_logs']
        return logs.sort_values('timestamp', ascending=False).head(limit)

    def get_active_alerts(self):
        alerts_df = st.session_state.db_data['alerts']
        if not alerts_df.empty:
            return alerts_df[alerts_df['status'] == 'Active'].to_dict('records')
        return []

class DataProcessor:
    """Handles data loading, validation, and preparation."""
    def __init__(self):
        self.db = MockDatabase()
        self._load_from_database()
    
    def _load_from_database(self):
        self.project_data = self.db.get_projects()
        self.inventory_data = self.db.get_inventory()
        self.historical_demand = self.db.get_historical_demand()
    
    def load_project_data(self, data):
        required_cols = ['project_id', 'project_name', 'status', 'start_date', 'end_date', 'region']
        if not all(col in data.columns for col in required_cols): return False, f"Missing columns. Required: {required_cols}"
        data['start_date'] = pd.to_datetime(data['start_date'], errors='coerce')
        data['end_date'] = pd.to_datetime(data['end_date'], errors='coerce')
        self.db.save_projects(data)
        self._load_from_database()
        self.db.log_activity(f"Loaded {len(data)} projects", status='Success')
        return True, "Project data loaded successfully"
            
    def load_inventory_data(self, data):
        required_cols = ['material_id', 'material_name', 'material_type', 'current_stock', 'unit_price']
        if not all(col in data.columns for col in required_cols): return False, f"Missing columns. Required: {required_cols}"
        for col in ['current_stock', 'unit_price', 'minimum_stock', 'maximum_stock']:
            if col in data.columns: data[col] = pd.to_numeric(data[col], errors='coerce')
        self.db.save_inventory(data)
        self._load_from_database()
        self.db.log_activity(f"Loaded {len(data)} inventory items", status='Success')
        return True, "Inventory data loaded successfully"
            
    def load_historical_demand(self, data):
        required_cols = ['date', 'material_type', 'demand']
        if not all(col in data.columns for col in required_cols): return False, f"Missing columns. Required: {required_cols}"
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        data['demand'] = pd.to_numeric(data['demand'], errors='coerce')
        self.db.save_historical_demand(data)
        self._load_from_database()
        self.db.log_activity(f"Loaded {len(data)} historical demand records", status='Success')
        return True, "Historical demand data loaded successfully"
    
    def get_active_projects(self): return self.db.get_active_projects()
    
    def get_project_status_data(self):
        if self.project_data.empty: return pd.DataFrame()
        status_counts = self.project_data['status'].value_counts().reset_index()
        status_counts.columns = ['status', 'count']
        return status_counts
    
    def get_demand_trend_data(self):
        if self.historical_demand.empty: return pd.DataFrame()
        trend_data = self.historical_demand.groupby(['date', 'material_type'])['demand'].sum().reset_index()
        return trend_data[trend_data['date'] >= datetime.now() - timedelta(days=30)]
    
    def get_total_material_types(self): return self.inventory_data['material_type'].nunique() if not self.inventory_data.empty else 0
    def get_inventory_value(self): return (self.inventory_data['current_stock'] * self.inventory_data['unit_price']).sum() if not self.inventory_data.empty else 0
    def get_procurement_savings(self): return 2500000 
    
    def get_critical_alerts(self):
        alerts = []
        if not self.inventory_data.empty and 'minimum_stock' in self.inventory_data.columns:
            low_stock = self.inventory_data[self.inventory_data['current_stock'] <= self.inventory_data['minimum_stock']]
            for _, item in low_stock.iterrows():
                alerts.append({'material': item['material_name'], 'message': f"Stock level critical: {item['current_stock']} units", 'type': 'critical'})
        return alerts[:5]
    
    def get_ai_recommendations(self):
        recs = []
        if not self.inventory_data.empty and not self.historical_demand.empty:
            avg_demand = self.historical_demand.groupby('material_type')['demand'].mean()
            for mat_type, avg_qty in avg_demand.items():
                stock = self.inventory_data[self.inventory_data['material_type'] == mat_type]['current_stock'].sum()
                if stock < avg_qty * 7: recs.append({'material': mat_type, 'recommendation': f"Procure {int(avg_qty * 30 - stock)} units"})
        return recs[:5]
    
    def get_recent_activity(self):
        logs = self.db.get_activity_logs(limit=10)
        return logs[['timestamp', 'activity', 'user_name', 'status']] if not logs.empty else pd.DataFrame()

    def get_data_summary(self):
        return {
            'projects': {'total': len(self.project_data), 'active': len(self.get_active_projects())},
            'inventory': {'total_items': len(self.inventory_data), 'material_types': self.get_total_material_types(), 'total_value': self.get_inventory_value()},
            'historical_data': {'records': len(self.historical_demand), 'date_range': f"{self.historical_demand['date'].min().date()} to {self.historical_demand['date'].max().date()}" if not self.historical_demand.empty else 'N/A'}
        }

class ForecastingModel:
    """Handles AI model training and prediction."""
    def __init__(self):
        self.model, self.label_encoders, self.feature_columns = None, {}, []
        self.is_trained = False
    
    def prepare_features(self, df):
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy['year'] = df_copy['date'].dt.year
        df_copy['month'] = df_copy['date'].dt.month
        df_copy['quarter'] = df_copy['date'].dt.quarter
        for col in ['material_type', 'project_type', 'region', 'supplier']:
            if col in df_copy.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder().fit(df_copy[col].astype(str))
                
                # Handle unseen labels during transform
                new_labels = set(df_copy[col].astype(str)) - set(self.label_encoders[col].classes_)
                if new_labels:
                    self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, sorted(list(new_labels)))

                df_copy[f'{col}_encoded'] = self.label_encoders[col].transform(df_copy[col].astype(str))
        return df_copy
    
    def train_model(self, training_data):
        try:
            df = self.prepare_features(training_data.copy())
            exclude = ['date', 'demand', 'material_type', 'project_type', 'region', 'supplier', 'project_id', 'project_name', 'material_id', 'material_name']
            self.feature_columns = [c for c in df.columns if c not in exclude and df[c].dtype in ['int64', 'float64', 'int32']]
            
            if not self.feature_columns:
                return {'success': False, 'error': 'No numeric features found for training.'}

            X, y = df[self.feature_columns].fillna(0), df['demand'].fillna(0)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_test)
            self.is_trained = True
            return {'success': True, 'mae': mean_absolute_error(y_test, preds), 'rmse': np.sqrt(mean_squared_error(y_test, preds))}
        except Exception as e: return {'success': False, 'error': str(e)}
    
    def predict_demand(self, prediction_data, forecast_days=30):
        if not self.is_trained: return {'success': False, 'error': 'Model not trained.'}
        try:
            forecasts = {}
            unique_materials = prediction_data['material_type'].unique()
            for material in unique_materials:
                material_data = prediction_data[prediction_data['material_type'] == material]
                if material_data.empty: continue
                last_date = material_data['date'].max()
                future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
                
                # Create future dataframe based on last known record
                last_record = material_data.sort_values('date').iloc[-1:]
                future_df = pd.concat([last_record]*forecast_days, ignore_index=True)
                future_df['date'] = future_dates
                
                future_df = self.prepare_features(future_df)
                
                # Align columns and fill missing with 0
                X_pred = future_df[self.feature_columns].fillna(0)
                
                predictions = self.model.predict(X_pred)
                forecasts[material] = {
                    'dates': future_dates, 'predictions': predictions.tolist(),
                    'confidence_lower': (predictions * 0.8).tolist(),
                    'confidence_upper': (predictions * 1.2).tolist()
                }
            return {'success': True, 'forecasts': forecasts}
        except Exception as e: return {'success': False, 'error': str(e)}

    def get_feature_importance(self):
        if not self.is_trained: return {}
        imp = dict(zip(self.feature_columns, self.model.feature_importances_))
        return dict(sorted(imp.items(), key=lambda item: item[1], reverse=True))

class ProcurementOptimizer:
    """Provides strategic procurement advice."""
    def _generate_mock_price_trend(self, material_type, days_ahead=90):
        base_prices = {'Electrical Cables': 850, 'Transformers': 125000, 'Steel Structures': 75000}
        base_price = base_prices.get(material_type, 1000)
        dates = [datetime.now() + timedelta(days=i) for i in range(1, days_ahead + 1)]
        prices = [base_price * (1 + 0.0002*i + np.sin(i*0.1)*0.03 + np.random.normal(0,0.01)) for i,_ in enumerate(dates)]
        return {'dates': dates, 'predicted_prices': prices, 'current_price': base_price, 'trend': 'Increasing'}

    def predict_price_trend(self, material_type, days_ahead=90):
        return self._generate_mock_price_trend(material_type, days_ahead)

    def optimize_procurement_timing(self, material_type, required_quantity, urgency_level='normal'):
        pred = self.predict_price_trend(material_type, 180)
        cost = pred['current_price'] * required_quantity
        results = [{'strategy': 'Immediate Procurement', 'procurement_date': datetime.now(), 'total_cost': cost, 'savings': 0, 'risk_level': 'Low', 'recommendation_score': 70 if urgency_level == 'high' else 50}]
        min_idx = np.argmin(pred['predicted_prices'][:90])
        min_price, min_date = pred['predicted_prices'][min_idx], pred['dates'][min_idx]
        min_cost = min_price * required_quantity
        results.append({'strategy': 'Optimal Price Window', 'procurement_date': min_date, 'total_cost': min_cost, 'savings': cost - min_cost, 'risk_level': 'Medium', 'recommendation_score': 90 if urgency_level == 'normal' else 60})
        results.sort(key=lambda x: x['recommendation_score'], reverse=True)
        return {'optimization_strategies': results, 'recommended_strategy': results[0]['strategy']}

@st.cache_resource
def initialize_components():
    dp = DataProcessor()
    fm = ForecastingModel()
    po = ProcurementOptimizer()
    if dp.historical_demand.empty:
        dp.load_historical_demand(pd.DataFrame({'date': pd.to_datetime(['2023-01-01']), 'material_type': ['Electrical Cables'], 'demand': [100]}))
    if not fm.is_trained: fm.train_model(dp.historical_demand)
    return dp, fm, po

dp, fm, po = initialize_components()
st.session_state.data_processor = dp
st.session_state.forecasting_model = fm
st.session_state.procurement_optimizer = po

def page_dashboard():
    st.title("Cognitive Supply Chain Engine")
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Projects", len(dp.get_active_projects()), "2 new")
    col2.metric("Material Categories", dp.get_total_material_types(), "5 new")
    col3.metric("Inventory Value", f"₹{dp.get_inventory_value():,.0f}", "-2.3%")
    col4.metric("Est. Monthly Savings", f"₹{dp.get_procurement_savings():,.0f}", "12.5%")
    st.header("🔄 Live Digital Twin - Supply Chain Overview")
    status_data = dp.get_project_status_data()
    if not status_data.empty:
        fig = px.pie(status_data, values='count', names='status', title="Current Project Status")
        st.plotly_chart(fig, use_container_width=True)
    st.header("🚨 Current Alerts & AI Recommendations")
    for alert in dp.get_critical_alerts(): st.error(f"**{alert['material']}**: {alert['message']}")
    for rec in dp.get_ai_recommendations(): st.info(f"**{rec['material']}**: {rec['recommendation']}")

def page_data_upload():
    st.title("📤 Data Upload & Management")
    tab1, tab2, tab3 = st.tabs(["Project Data", "Inventory Data", "Historical Demand"])
    with tab1:
        uploaded_file = st.file_uploader("Upload Project CSV", type="csv", key="proj_csv")
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.dataframe(data.head())
            if st.button("Load Project Data"):
                success, msg = dp.load_project_data(data)
                st.success(msg) if success else st.error(msg)
    with tab2:
        uploaded_file = st.file_uploader("Upload Inventory CSV", type="csv", key="inv_csv")
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.dataframe(data.head())
            if st.button("Load Inventory Data"):
                success, msg = dp.load_inventory_data(data)
                st.success(msg) if success else st.error(msg)
    with tab3:
        uploaded_file = st.file_uploader("Upload Demand CSV", type="csv", key="demand_csv")
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.dataframe(data.head())
            if st.button("Load Historical Demand Data"):
                success, msg = dp.load_historical_demand(data)
                st.success(msg) if success else st.error(msg)

def page_demand_forecasting():
    st.title("🔮 AI Demand Forecasting Engine")
    if dp.historical_demand.empty:
        st.warning("Please upload historical demand data first.")
        return

    if st.sidebar.button("🧠 Train/Retrain Model"):
        with st.spinner("Training model..."):
            res = fm.train_model(dp.historical_demand)
            if res['success']: st.sidebar.success(f"Model trained! MAE: {res['mae']:.2f}")
            else: st.sidebar.error(res['error'])
    
    if not fm.is_trained:
        st.info("Train the model to generate forecasts.")
        return

    forecast_days = st.sidebar.slider("Forecast Period (Days)", 7, 180, 30)
    if st.sidebar.button("Generate Forecast"):
        with st.spinner("Generating forecast..."):
            res = fm.predict_demand(dp.historical_demand, forecast_days)
            st.session_state.forecast_results = res
    
    if 'forecast_results' in st.session_state and st.session_state.forecast_results['success']:
        for material, data in st.session_state.forecast_results['forecasts'].items():
            st.subheader(f"Forecast for {material}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['dates'], y=data['predictions'], name='Forecast'))
            fig.add_trace(go.Scatter(x=data['dates']+data['dates'][::-1], y=data['confidence_upper']+data['confidence_lower'][::-1], fill='toself', name='Confidence'))
            st.plotly_chart(fig)

def page_inventory_management():
    st.title("📦 Net Procurement Calculator")
    if dp.inventory_data.empty:
        st.warning("Please upload inventory data.")
        return
    
    material = st.selectbox("Material Type", dp.inventory_data['material_type'].unique())
    period = st.slider("Planning Period (Days)", 7, 180, 30)
    if st.button("Calculate Net Requirement"):
        forecast = fm.predict_demand(dp.historical_demand, period) if fm.is_trained else None
        gross = sum(forecast['forecasts'][material]['predictions']) if forecast and forecast['success'] and material in forecast['forecasts'] else 0
        if gross == 0: # Fallback
            avg = dp.historical_demand[dp.historical_demand['material_type'] == material]['demand'].mean()
            gross = (avg if pd.notna(avg) else 0) * period
        
        current_stock = dp.inventory_data[dp.inventory_data['material_type'] == material]['current_stock'].sum()
        safety_stock_df = dp.inventory_data[dp.inventory_data['material_type'] == material]
        safety_stock = safety_stock_df['minimum_stock'].sum() if 'minimum_stock' in safety_stock_df.columns else 0
        net = max(0, gross - current_stock + safety_stock)
        st.metric("Gross Demand", f"{gross:,.0f} units")
        st.metric("Current Stock", f"{current_stock:,.0f} units")
        st.metric("Net Procurement Requirement", f"{net:,.0f} units")


def page_procurement_optimization():
    st.title("💰 Strategic Procurement Optimizer")
    if dp.inventory_data.empty:
        st.warning("Please upload inventory data.")
        return
    material = st.selectbox("Select Material Type", dp.inventory_data['material_type'].unique())
    quantity = st.number_input("Required Quantity", 1, value=1000)
    if st.button("Optimize Procurement"):
        res = po.optimize_procurement_timing(material, quantity)
        st.success(f"Recommended Strategy: {res['recommended_strategy']}")
        df = pd.DataFrame(res['optimization_strategies'])
        st.dataframe(df)
        fig = px.bar(df, x='strategy', y='total_cost', color='risk_level', title="Strategy Cost vs. Risk")
        st.plotly_chart(fig)

def page_reports_export():
    st.title("📊 Reports & Data Export")
    st.info("Generate comprehensive reports for different aspects of the supply chain.")
    
    report_type = st.selectbox("Select Report Type", ["Inventory Status", "Demand Forecast Summary", "Executive Summary"])

    if st.button("Generate & Download Report"):
        if report_type == "Inventory Status":
            data = dp.inventory_data
            filename = "inventory_status_report.csv"
        elif report_type == "Demand Forecast Summary":
            if 'forecast_results' in st.session_state and st.session_state.forecast_results['success']:
                all_forecasts = []
                for material, forecast_data in st.session_state.forecast_results['forecasts'].items():
                    total_demand = sum(forecast_data['predictions'])
                    all_forecasts.append({'Material': material, 'Total Forecast Demand': total_demand})
                data = pd.DataFrame(all_forecasts)
                filename = "demand_forecast_summary.csv"
            else:
                st.warning("Please generate a forecast first on the 'Demand Forecasting' page.")
                return
        else: # Executive Summary
            summary = dp.get_data_summary()
            data = pd.DataFrame({
                "Metric": ["Total Projects", "Active Projects", "Total Inventory Items", "Inventory Value (INR)"],
                "Value": [summary['projects']['total'], summary['projects']['active'], summary['inventory']['total_items'], f"{summary['inventory']['total_value']:,.0f}"]
            })
            filename = "executive_summary.csv"

        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Report as CSV",
            data=csv,
            file_name=filename,
            mime='text/csv',
        )

def page_alert_management():
    st.title("🚨 Alert Management")
    st.info("Monitor critical supply chain alerts.")
    alerts = dp.get_critical_alerts()
    if alerts:
        st.subheader("Active Alerts")
        for alert in alerts:
            st.error(f"**{alert['material']}**: {alert['message']}")
    else:
        st.success("No active alerts.")

def page_logistics_optimization():
    st.title("🚚 Logistics Optimization")
    st.info("Optimize transportation routes and predict bottlenecks.")
    
    # Mock data for demonstration
    locations = ["Delhi Warehouse", "Mumbai Port", "Bangalore Factory", "Kolkata Hub"]
    start = st.selectbox("Start Location", locations, index=0)
    end = st.selectbox("End Location", locations, index=1)

    if st.button("Optimize Route"):
        # Mock optimization
        distance = np.random.randint(500, 2000)
        time = distance / 60 
        cost = distance * 15.5
        
        st.subheader("Optimized Route")
        col1, col2, col3 = st.columns(3)
        col1.metric("Distance", f"{distance} km")
        col2.metric("Est. Time", f"{time:.1f} hours")
        col3.metric("Est. Cost", f"₹{cost:,.2f}")

        # Mock map
        map_data = pd.DataFrame(
            np.random.randn(1, 2) / [50, 50] + [28.61, 77.23], # Delhi as start
            columns=['lat', 'lon'])
        st.map(map_data)
        st.success("Route optimized for cost and time efficiency.")


st.sidebar.title("⚡ Navigation")
PAGES = {
    "🏠 Dashboard": page_dashboard,
    "📤 Data Upload": page_data_upload,
    "🔮 Demand Forecasting": page_demand_forecasting,
    "📦 Inventory Management": page_inventory_management,
    "💰 Procurement Optimization": page_procurement_optimization,
    "📊 Reports & Export": page_reports_export,
    "🚨 Alert Management": page_alert_management,
    "🚚 Logistics Optimization": page_logistics_optimization,
}
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page_function = PAGES[selection]
page_function()

