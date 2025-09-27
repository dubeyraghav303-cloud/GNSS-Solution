# GNSS Time-Series Forecasting Solution

A complete end-to-end Python solution for predicting Global Navigation Satellite Systems (GNSS) clock and ephemeris errors using a hybrid Transformer-LSTM deep learning model.

## 🚀 Features

- **Hybrid Deep Learning Model**: Transformer encoder + LSTM layers for capturing both long-range dependencies and sequential patterns
- **Custom Loss Function**: Combines Mean Squared Error (MSE) with Kullback-Leibler (KL) divergence
- **Comprehensive Feature Engineering**: Lag features, rolling statistics, and cyclical time features
- **Walk-Forward Validation**: Robust training strategy using 6 days for training and 1 day for validation
- **Interactive Web Interface**: Streamlit-based frontend for easy model interaction and visualization
- **Statistical Analysis**: Error distribution analysis using Shapiro-Wilk test for normality

## 📁 Project Structure

```
PS 25176/
├── gnss_forecasting_solution.py    # Main forecasting solution
├── streamlit_app.py                # Streamlit web interface
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── DATA_GEO_Train.csv             # GEO satellite training data
├── DATA_MEO_Train.csv             # MEO satellite training data (Part 1)
├── DATA_MEO_Train2.csv            # MEO satellite training data (Part 2)
└── SIH_Data_Discription.pdf       # Data description and problem statement
```

## 🛠️ Installation

1. **Clone or download the project files**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Option 1: Command Line Interface

Run the main solution directly:

```bash
python gnss_forecasting_solution.py
```

This will:
- Load and preprocess the GNSS data
- Train the hybrid Transformer-LSTM model
- Generate predictions for the 8th day
- Evaluate model performance
- Create visualization plots

### Option 2: Streamlit Web Interface

Launch the interactive web interface:

```bash
streamlit run streamlit_app.py
```

The web interface provides:
- **Data Overview**: Load and visualize the GNSS data
- **Model Training**: Interactive model training with parameter tuning
- **Predictions**: Generate and visualize 8th day predictions
- **Evaluation**: Comprehensive model evaluation and analysis
- **Report**: Detailed analysis report with downloadable results

## 📊 Model Architecture

### Hybrid Transformer-LSTM Model

1. **Transformer Encoder**:
   - Multi-head self-attention (8 heads)
   - Layer normalization and feed-forward networks
   - Captures long-range dependencies in time-series data

2. **LSTM Layers**:
   - Three LSTM layers with decreasing hidden units (128 → 64 → 32)
   - Dropout regularization for overfitting prevention
   - Handles sequential nature and short-term dependencies

3. **Output Layer**:
   - Dense layers for final prediction
   - Outputs 96 time steps (24 hours) for 4 error variables
   - Linear activation for regression task

### Custom Loss Function

```python
Total Loss = MSE Loss + 0.1 × KL Divergence Loss
```

- **MSE Loss**: Standard regression loss for prediction accuracy
- **KL Divergence**: Encourages predictions to follow normal distribution
- **Weight**: 0.1 balance between accuracy and distribution shape

## 🔧 Feature Engineering

### 1. Lag Features
- **Time Steps**: t-1, t-4, t-8 (15min, 1hr, 2hr lags)
- **Variables**: All error types (x_error, y_error, z_error, satclockerror)

### 2. Rolling Statistics
- **Windows**: 1 hour (4 steps) and 4 hours (16 steps)
- **Statistics**: Rolling mean and standard deviation
- **Purpose**: Capture short-term volatility and trends

### 3. Time-Based Features
- **Cyclical Encoding**: Sine and cosine transformations
- **Features**: hour_of_day, day_of_week, day_of_year
- **Benefit**: Preserves cyclical nature of time features

### 4. Satellite-Specific Features
- **Categorical Encoding**: Satellite ID and type
- **Purpose**: Learn satellite-specific biases and characteristics

## 📈 Evaluation Metrics

### 1. Root Mean Squared Error (RMSE)
- **Horizons**: 15min, 30min, 1hr, 2hr, 4hr, 8hr, 16hr, 24hr
- **Purpose**: Measure prediction accuracy at different time horizons

### 2. Error Distribution Analysis
- **Shapiro-Wilk Test**: Statistical test for normality
- **Statistics**: Mean, standard deviation, distribution shape
- **Interpretation**: p-value > 0.05 indicates normal distribution

### 3. Training Metrics
- **Loss Curves**: Training and validation loss over epochs
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rate reduction

## 🎯 Key Results

The solution provides:

1. **Accurate Predictions**: RMSE values for different prediction horizons
2. **Statistical Validation**: Error distribution analysis with normality tests
3. **Visual Insights**: Comprehensive plots and visualizations
4. **Export Capabilities**: Downloadable predictions and reports
5. **Interactive Interface**: User-friendly web interface for model interaction

## 🔍 Technical Details

### Data Preprocessing
- **Resampling**: Uniform 15-minute intervals
- **Interpolation**: Linear interpolation for missing values
- **Scaling**: StandardScaler for feature normalization

### Training Strategy
- **Walk-Forward Validation**: 6 days training, 1 day validation
- **Sequence Length**: 96 time steps (24 hours) input
- **Output Length**: 96 time steps (24 hours) predictions
- **Batch Size**: 32 (configurable)

### Model Parameters
- **Learning Rate**: 0.001 (configurable)
- **Epochs**: 50 (configurable)
- **Dropout**: 0.1-0.3 (various layers)
- **Attention Heads**: 8

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.10+
- Pandas 1.5+
- NumPy 1.21+
- Streamlit 1.25+
- Plotly 5.15+
- Scikit-learn 1.1+
- SciPy 1.9+

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TensorFlow/Keras team for the deep learning framework
- Streamlit team for the web interface framework
- The GNSS community for providing the dataset and problem statement

## 📞 Support

For questions or issues, please:
1. Check the documentation
2. Review the code comments
3. Open an issue in the repository
4. Contact the development team

---

**Note**: This solution is designed for educational and research purposes. For production use, additional validation, testing, and optimization may be required.
