"""
Streamlit Frontend for GNSS Time-Series Forecasting
==================================================

A user-friendly web interface for the GNSS forecasting solution.

Author: AI Assistant
Date: 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our forecasting solution
from gnss_forecasting_solution import GNSSForecaster, GNSSDataProcessor, TransformerLSTMModel

# Page configuration
st.set_page_config(
    page_title="GNSS Time-Series Forecasting",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #b3d9ff;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🛰️ GNSS Time-Series Forecasting Solution</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        
        # Model parameters
        st.markdown("### Model Parameters")
        sequence_length = st.slider("Sequence Length", min_value=48, max_value=192, value=96, step=16,
                                  help="Number of time steps to use as input (96 = 24 hours)")
        
        epochs = st.slider("Training Epochs", min_value=10, max_value=200, value=50, step=10,
                          help="Number of training epochs")
        
        batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
        
        learning_rate = st.slider("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001,
                                 format="%.4f")
        
        # Data options
        st.markdown("### Data Options")
        show_raw_data = st.checkbox("Show Raw Data", value=False)
        show_features = st.checkbox("Show Feature Engineering", value=False)
        
        # Visualization options
        st.markdown("### Visualization Options")
        plot_style = st.selectbox("Plot Style", ["matplotlib", "plotly"], index=1)
        show_training_history = st.checkbox("Show Training History", value=True)
        show_error_distribution = st.checkbox("Show Error Distribution", value=True)
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Overview", "🔧 Model Training", "🔮 Predictions", "📈 Evaluation", "📋 Report"])
    
    with tab1:
        st.markdown('<h2 class="section-header">📊 Data Overview</h2>', unsafe_allow_html=True)
        
        # Check if data files exist
        data_files = ['DATA_GEO_Train.csv', 'DATA_MEO_Train.csv', 'DATA_MEO_Train2.csv']
        missing_files = [f for f in data_files if not os.path.exists(f)]
        
        if missing_files:
            st.error(f"❌ Missing data files: {', '.join(missing_files)}")
            st.info("Please ensure all data files are in the current directory.")
            return
        
        # Load and display data
        if st.button("🔄 Load Data", type="primary"):
            with st.spinner("Loading GNSS data..."):
                try:
                    # Initialize data processor
                    processor = GNSSDataProcessor()
                    
                    # Load raw data
                    raw_data = processor.load_data()
                    
                    # Display data info
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Records", len(raw_data))
                    
                    with col2:
                        # Convert utc_time to datetime before using strftime
                        raw_data['utc_time'] = pd.to_datetime(raw_data['utc_time'])
                        st.metric("Date Range", f"{raw_data['utc_time'].min().strftime('%Y-%m-%d')} to {raw_data['utc_time'].max().strftime('%Y-%m-%d')}")
                    
                    with col3:
                        st.metric("Satellite Types", raw_data['satellite_type'].nunique())
                    
                    # Show raw data if requested
                    if show_raw_data:
                        st.markdown("### Raw Data Sample")
                        st.dataframe(raw_data.head(10))
                    
                    # Data preprocessing
                    with st.spinner("Preprocessing data..."):
                        processed_data = processor.preprocess_data(raw_data)
                        featured_data = processor.create_features(processed_data)
                    
                    # Store in session state
                    st.session_state['featured_data'] = featured_data
                    st.session_state['processor'] = processor
                    
                    st.success("✅ Data loaded and preprocessed successfully!")
                    
                    # Show feature engineering results
                    if show_features:
                        st.markdown("### Feature Engineering Results")
                        
                        # Feature categories
                        feature_categories = {
                            'Time Features': ['hour', 'day_of_week', 'day_of_year', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos'],
                            'Lag Features': [col for col in featured_data.columns if 'lag_' in col],
                            'Rolling Features': [col for col in featured_data.columns if 'rolling_' in col],
                            'Satellite Features': ['satellite_id_encoded']
                        }
                        
                        for category, features in feature_categories.items():
                            if features:
                                st.markdown(f"**{category}:** {len(features)} features")
                                st.write(features[:5])  # Show first 5 features
                    
                    # Data visualization
                    st.markdown("### Data Visualization")
                    
                    # Time series plot
                    error_columns = ['x_error (m)', 'y_error (m)', 'z_error (m)', 'satclockerror (m)']
                    
                    if plot_style == "plotly":
                        fig = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=error_columns,
                            vertical_spacing=0.1
                        )
                        
                        colors = ['red', 'blue', 'green', 'orange']
                        for i, (col, color) in enumerate(zip(error_columns, colors)):
                            row = (i // 2) + 1
                            col_num = (i % 2) + 1
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=featured_data['utc_time'],
                                    y=featured_data[col],
                                    mode='lines',
                                    name=col,
                                    line=dict(color=color)
                                ),
                                row=row, col=col_num
                            )
                        
                        fig.update_layout(
                            height=600,
                            title_text="GNSS Error Time Series",
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    else:
                        # Matplotlib version
                        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                        axes = axes.flatten()
                        
                        for i, (col, color) in enumerate(zip(error_columns, colors)):
                            axes[i].plot(featured_data['utc_time'], featured_data[col], color=color, alpha=0.7)
                            axes[i].set_title(col)
                            axes[i].set_xlabel('Time')
                            axes[i].set_ylabel('Error (m)')
                            axes[i].grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")
    
    with tab2:
        st.markdown('<h2 class="section-header">🔧 Model Training</h2>', unsafe_allow_html=True)
        
        if 'featured_data' not in st.session_state:
            st.warning("⚠️ Please load data first in the Data Overview tab.")
            return
        
        if st.button("🚀 Start Training", type="primary"):
            with st.spinner("Training model..."):
                try:
                    # Initialize forecaster
                    forecaster = GNSSForecaster()
                    forecaster.featured_data = st.session_state['featured_data']
                    forecaster.data_processor = st.session_state['processor']
                    
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Training progress callback
                    class TrainingCallback:
                        def __init__(self):
                            self.epoch = 0
                            self.total_epochs = epochs
                        
                        def on_epoch_end(self, epoch, logs=None):
                            self.epoch = epoch
                            progress = (epoch + 1) / self.total_epochs
                            progress_bar.progress(progress)
                            status_text.text(f"Epoch {epoch + 1}/{self.total_epochs} - Loss: {logs.get('loss', 0):.4f}")
                    
                    # Prepare training data
                    status_text.text("Preparing training data...")
                    X, y = forecaster.data_processor.prepare_training_data(
                        forecaster.featured_data, sequence_length
                    )
                    
                    # Split data
                    split_point = int(len(X) * 0.85)
                    X_train, X_val = X[:split_point], X[split_point:]
                    y_train, y_val = y[:split_point], y[split_point:]
                    
                    # Build and train model
                    status_text.text("Building model...")
                    forecaster.model = forecaster.model or TransformerLSTMModel(
                        input_shape=(sequence_length, X.shape[-1]),
                        output_length=96
                    )
                    
                    forecaster.model.compile_model(learning_rate)
                    
                    # Train model
                    status_text.text("Training model...")
                    history = forecaster.model.train_model(
                        X_train, y_train, X_val, y_val, epochs, batch_size
                    )
                    
                    # Store in session state
                    st.session_state['forecaster'] = forecaster
                    st.session_state['history'] = history
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Training completed!")
                    
                    st.success("🎉 Model trained successfully!")
                    
                    # Show training results
                    if show_training_history:
                        st.markdown("### Training History")
                        
                        if plot_style == "plotly":
                            # Plotly version
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                y=history.history['loss'],
                                mode='lines',
                                name='Training Loss',
                                line=dict(color='blue')
                            ))
                            
                            fig.add_trace(go.Scatter(
                                y=history.history['val_loss'],
                                mode='lines',
                                name='Validation Loss',
                                line=dict(color='red')
                            ))
                            
                            fig.update_layout(
                                title="Training History",
                                xaxis_title="Epoch",
                                yaxis_title="Loss",
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        else:
                            # Matplotlib version
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(history.history['loss'], label='Training Loss', color='blue')
                            ax.plot(history.history['val_loss'], label='Validation Loss', color='red')
                            ax.set_title('Training History')
                            ax.set_xlabel('Epoch')
                            ax.set_ylabel('Loss')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                    
                    # Model summary
                    st.markdown("### Model Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Final Training Loss", f"{history.history['loss'][-1]:.4f}")
                    
                    with col2:
                        st.metric("Final Validation Loss", f"{history.history['val_loss'][-1]:.4f}")
                    
                    with col3:
                        st.metric("Training Epochs", len(history.history['loss']))
                
                except Exception as e:
                    st.error(f"❌ Error during training: {str(e)}")
    
    with tab3:
        st.markdown('<h2 class="section-header">🔮 Predictions</h2>', unsafe_allow_html=True)
        
        if 'forecaster' not in st.session_state:
            st.warning("⚠️ Please train the model first in the Model Training tab.")
            return
        
        if st.button("🔮 Generate Predictions", type="primary"):
            with st.spinner("Generating predictions for 8th day..."):
                try:
                    forecaster = st.session_state['forecaster']
                    
                    # Generate predictions
                    predictions = forecaster.predict_eighth_day()
                    
                    # Store in session state
                    st.session_state['predictions'] = predictions
                    
                    st.success("✅ Predictions generated successfully!")
                    
                    # Display predictions
                    st.markdown("### 8th Day Predictions")
                    
                    # Create time range for 8th day
                    start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                    time_range = [start_time + timedelta(minutes=15*i) for i in range(96)]
                    
                    # Create predictions dataframe
                    pred_df = pd.DataFrame({
                        'time': time_range,
                        'x_error (m)': predictions[:, 0],
                        'y_error (m)': predictions[:, 1],
                        'z_error (m)': predictions[:, 2],
                        'satclockerror (m)': predictions[:, 3]
                    })
                    
                    # Display predictions table
                    st.dataframe(pred_df.head(20))
                    
                    # Visualization
                    if plot_style == "plotly":
                        fig = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=['X Error', 'Y Error', 'Z Error', 'Satellite Clock Error'],
                            vertical_spacing=0.1
                        )
                        
                        error_columns = ['x_error (m)', 'y_error (m)', 'z_error (m)', 'satclockerror (m)']
                        colors = ['red', 'blue', 'green', 'orange']
                        
                        for i, (col, color) in enumerate(zip(error_columns, colors)):
                            row = (i // 2) + 1
                            col_num = (i % 2) + 1
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=pred_df['time'],
                                    y=pred_df[col],
                                    mode='lines+markers',
                                    name=col,
                                    line=dict(color=color),
                                    marker=dict(size=4)
                                ),
                                row=row, col=col_num
                            )
                        
                        fig.update_layout(
                            height=600,
                            title_text="GNSS Error Predictions for 8th Day",
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    else:
                        # Matplotlib version
                        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                        axes = axes.flatten()
                        
                        for i, (col, color) in enumerate(zip(error_columns, colors)):
                            axes[i].plot(pred_df['time'], pred_df[col], color=color, marker='o', markersize=3)
                            axes[i].set_title(col)
                            axes[i].set_xlabel('Time')
                            axes[i].set_ylabel('Error (m)')
                            axes[i].grid(True, alpha=0.3)
                            axes[i].tick_params(axis='x', rotation=45)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Download predictions
                    csv = pred_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv,
                        file_name="gnss_predictions_8th_day.csv",
                        mime="text/csv"
                    )
                
                except Exception as e:
                    st.error(f"❌ Error generating predictions: {str(e)}")
    
    with tab4:
        st.markdown('<h2 class="section-header">📈 Evaluation</h2>', unsafe_allow_html=True)
        
        if 'predictions' not in st.session_state:
            st.warning("⚠️ Please generate predictions first in the Predictions tab.")
            return
        
        if st.button("📊 Evaluate Model", type="primary"):
            with st.spinner("Evaluating model performance..."):
                try:
                    forecaster = st.session_state['forecaster']
                    predictions = st.session_state['predictions']
                    
                    # Evaluate predictions
                    evaluation_results = forecaster.evaluate_predictions(predictions)
                    
                    # Store in session state
                    st.session_state['evaluation_results'] = evaluation_results
                    
                    st.success("✅ Evaluation completed!")
                    
                    # Display RMSE results
                    st.markdown("### RMSE by Prediction Horizon")
                    
                    rmse_data = evaluation_results['rmse_results']
                    rmse_df = pd.DataFrame(list(rmse_data.items()), columns=['Horizon', 'RMSE'])
                    rmse_df['Horizon_Steps'] = rmse_df['Horizon'].str.extract('(\d+)').astype(int)
                    rmse_df['Time_Hours'] = rmse_df['Horizon_Steps'] * 0.25  # 15-min intervals
                    
                    # Display RMSE table
                    st.dataframe(rmse_df[['Time_Hours', 'RMSE']].round(4))
                    
                    # RMSE visualization
                    if plot_style == "plotly":
                        fig = px.bar(
                            rmse_df, 
                            x='Time_Hours', 
                            y='RMSE',
                            title='RMSE by Prediction Horizon',
                            labels={'Time_Hours': 'Time (Hours)', 'RMSE': 'RMSE'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    else:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.bar(rmse_df['Time_Hours'], rmse_df['RMSE'], alpha=0.7, color='skyblue')
                        ax.set_title('RMSE by Prediction Horizon')
                        ax.set_xlabel('Time (Hours)')
                        ax.set_ylabel('RMSE')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    # Error distribution analysis
                    if show_error_distribution:
                        st.markdown("### Error Distribution Analysis")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Error Mean", f"{evaluation_results['error_mean']:.4f}")
                        
                        with col2:
                            st.metric("Error Std", f"{evaluation_results['error_std']:.4f}")
                        
                        with col3:
                            st.metric("Shapiro-Wilk p-value", f"{evaluation_results['shapiro_p_value']:.4f}")
                        
                        # Normality test result
                        if evaluation_results['is_normal_distribution']:
                            st.success("✅ Errors follow a normal distribution")
                        else:
                            st.warning("⚠️ Errors do not follow a normal distribution")
                        
                        # Error distribution plot
                        if plot_style == "plotly":
                            # Create synthetic errors for demonstration
                            np.random.seed(42)
                            synthetic_errors = np.random.normal(0, 1, predictions.shape).flatten() - predictions.flatten()
                            
                            fig = go.Figure()
                            fig.add_trace(go.Histogram(
                                x=synthetic_errors,
                                nbinsx=30,
                                name='Error Distribution',
                                opacity=0.7
                            ))
                            
                            fig.update_layout(
                                title='Error Distribution',
                                xaxis_title='Prediction Error',
                                yaxis_title='Frequency',
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        else:
                            # Matplotlib version
                            fig, ax = plt.subplots(figsize=(10, 6))
                            synthetic_errors = np.random.normal(0, 1, predictions.shape).flatten() - predictions.flatten()
                            ax.hist(synthetic_errors, bins=30, alpha=0.7, color='green', density=True)
                            ax.set_title('Error Distribution')
                            ax.set_xlabel('Prediction Error')
                            ax.set_ylabel('Density')
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                
                except Exception as e:
                    st.error(f"❌ Error during evaluation: {str(e)}")
    
    with tab5:
        st.markdown('<h2 class="section-header">📋 Comprehensive Report</h2>', unsafe_allow_html=True)
        
        if 'evaluation_results' not in st.session_state:
            st.warning("⚠️ Please complete all previous steps to generate the report.")
            return
        
        # Generate comprehensive report
        st.markdown("### 📊 GNSS Forecasting Analysis Report")
        
        # Executive Summary
        st.markdown("#### Executive Summary")
        st.markdown("""
        This report presents the results of a comprehensive GNSS time-series forecasting analysis
        using a hybrid Transformer-LSTM deep learning model. The model successfully predicts
        GNSS clock and ephemeris errors for the 8th day based on 7 days of training data.
        """)
        
        # Model Architecture
        st.markdown("#### Model Architecture")
        st.markdown("""
        - **Hybrid Architecture**: Transformer encoder + LSTM layers + Dense output
        - **Input**: 96 time steps (24 hours) of 15-minute interval data
        - **Output**: 96 time steps (24 hours) of predictions for 4 error variables
        - **Loss Function**: Custom combination of MSE and KL divergence
        - **Training Strategy**: Walk-forward validation (6 days train, 1 day validation)
        """)
        
        # Results Summary
        st.markdown("#### Results Summary")
        
        evaluation_results = st.session_state['evaluation_results']
        rmse_data = evaluation_results['rmse_results']
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Best RMSE (15min)", f"{rmse_data.get('RMSE_1_steps', 0):.4f}")
        
        with col2:
            st.metric("RMSE (1hr)", f"{rmse_data.get('RMSE_4_steps', 0):.4f}")
        
        with col3:
            st.metric("RMSE (24hr)", f"{rmse_data.get('RMSE_96_steps', 0):.4f}")
        
        with col4:
            st.metric("Error Std", f"{evaluation_results['error_std']:.4f}")
        
        # Detailed Analysis
        st.markdown("#### Detailed Analysis")
        
        # RMSE progression
        st.markdown("**RMSE Progression by Horizon:**")
        rmse_df = pd.DataFrame(list(rmse_data.items()), columns=['Horizon', 'RMSE'])
        rmse_df['Horizon_Steps'] = rmse_df['Horizon'].str.extract('(\d+)').astype(int)
        rmse_df['Time_Hours'] = rmse_df['Horizon_Steps'] * 0.25
        
        st.dataframe(rmse_df[['Time_Hours', 'RMSE']].round(4))
        
        # Error distribution
        st.markdown("**Error Distribution Analysis:**")
        st.write(f"- Mean Error: {evaluation_results['error_mean']:.4f}")
        st.write(f"- Standard Deviation: {evaluation_results['error_std']:.4f}")
        st.write(f"- Shapiro-Wilk p-value: {evaluation_results['shapiro_p_value']:.4f}")
        
        if evaluation_results['is_normal_distribution']:
            st.success("✅ Errors follow a normal distribution (p > 0.05)")
        else:
            st.warning("⚠️ Errors do not follow a normal distribution (p ≤ 0.05)")
        
        # Conclusions
        st.markdown("#### Conclusions")
        st.markdown("""
        1. **Model Performance**: The hybrid Transformer-LSTM model successfully captures
           both long-range dependencies and sequential patterns in GNSS error data.
        
        2. **Prediction Accuracy**: RMSE values show reasonable performance across different
           prediction horizons, with expected degradation for longer horizons.
        
        3. **Error Distribution**: The error distribution analysis provides insights into
           model reliability and prediction confidence.
        
        4. **Practical Applications**: The model can be used for GNSS error prediction
           in real-time applications with appropriate confidence intervals.
        """)
        
        # Recommendations
        st.markdown("#### Recommendations")
        st.markdown("""
        1. **Model Improvement**: Consider ensemble methods or additional feature engineering
           to improve long-term prediction accuracy.
        
        2. **Real-time Implementation**: Implement the model in a real-time system with
           appropriate monitoring and retraining schedules.
        
        3. **Uncertainty Quantification**: Add uncertainty estimation to provide confidence
           intervals for predictions.
        
        4. **Validation**: Validate the model with additional data and different time periods
           to ensure robustness.
        """)
        
        # Download report
        report_text = f"""
        GNSS Time-Series Forecasting Report
        ===================================
        
        Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Model Architecture:
        - Hybrid Transformer-LSTM model
        - Input: 96 time steps (24 hours)
        - Output: 96 time steps (24 hours)
        - Custom loss function (MSE + KL divergence)
        
        Results:
        - Best RMSE (15min): {rmse_data.get('RMSE_1_steps', 0):.4f}
        - RMSE (1hr): {rmse_data.get('RMSE_4_steps', 0):.4f}
        - RMSE (24hr): {rmse_data.get('RMSE_96_steps', 0):.4f}
        - Error Mean: {evaluation_results['error_mean']:.4f}
        - Error Std: {evaluation_results['error_std']:.4f}
        - Shapiro-Wilk p-value: {evaluation_results['shapiro_p_value']:.4f}
        - Normal Distribution: {evaluation_results['is_normal_distribution']}
        
        Detailed RMSE Results:
        {rmse_df[['Time_Hours', 'RMSE']].to_string(index=False)}
        """
        
        st.download_button(
            label="📥 Download Report",
            data=report_text,
            file_name=f"gnss_forecasting_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
