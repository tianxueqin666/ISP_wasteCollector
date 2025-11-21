# GRU Model for Waste Bin Fill Level Prediction

This directory contains the implementation of the GRU (Gated Recurrent Unit) model for predicting waste bin fill levels, based on the paper "A Machine Learning Approach to Predicting Waste Bin Fill Levels for Smart Waste Management Systems".

## Model Architecture

The GRU model follows the architecture specified in the paper:

```
Input (30 timesteps, 1 feature)
    ↓
GRU Layer (100 units)
    ↓
Dropout (rate: 0.2)
    ↓
Dense Layer (1 unit)
    ↓
Output (predicted fill level)
```

### Model Summary
- **GRU Layer**: 100 units, tanh activation, sigmoid recurrent activation
- **Dropout Layer**: 0.2 dropout rate for regularization
- **Dense Layer**: 1 unit for regression output
- **Total Parameters**: ~31,001 (may vary by TensorFlow version)

### GRU Equations (from paper)
The GRU implementation uses the following equations:
- **Update gate**: $z_t = \sigma(W_z·x_t + V_z·h_{t-1} + b_z)$
- **Reset gate**: $r_t = \sigma(W_r·x_t + V_r·h_{t-1} + b_r)$
- **Candidate activation**: $\tilde{h}_t = \tanh(W_c·x_t + V_c·(r_t · h_{t-1}))$
- **Final hidden state**: $h_t = z_t · h_{t-1} + (1 - z_t) · \tilde{h}_t$

## Methodology

This implementation follows the same methodology as the 1D_CNN baseline:

### Data Processing
- **Individual Bin Prediction**: Each bin's data is treated separately
- **Temporal Splitting**: Chronological 80/20 train/test split per bin
- **Sequence Length**: 30 days (as specified in paper)
- **Normalization**: MinMax scaling per bin (0-1 range)
- **Gap Filling**: Forward/backward fill for missing daily values

### Training Configuration
- **Epochs**: 20
- **Batch Size**: 70
- **Optimizer**: Adam (default learning rate: 0.001)
- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: Mean Absolute Error (MAE)

### Evaluation Metrics
The model is evaluated using four metrics on the original scale (0-10):
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)

## Files

- `main.py` - Main training script with end-to-end pipeline
- `modelBuilding.py` - GRU model architecture definition
- `trainEvaluate.py` - Training and evaluation utilities
- `dataLoader.py` - Data loading and sequence creation
- `dataCleaner.py` - Data cleaning and preprocessing utilities

## Usage

### Training the Model

```bash
cd code/GRU
python main.py
```

This will:
1. Load the cleaned dataset from `../../data/wyndham_waste_data_cleaned.csv`
2. Create individual bin sequences with temporal splitting
3. Build and train the GRU model
4. Evaluate performance on test set
5. Save outputs to `../../outputs/`:
   - Model: `models/gru_individual_bin_model.keras`
   - Scalers: `models/gru_bin_scalers.pkl`
   - Results: `gru_training_results.json`
   - Plots: `gru_training_results.png`

### Model Outputs

**Model Files:**
- `gru_individual_bin_model.keras` - Trained model in Keras format
- `gru_bin_scalers.pkl` - MinMax scalers for each bin (for inverse transform)

**Results:**
- `gru_training_results.json` - Training/test metrics and configuration
- `gru_training_results.png` - Visualization of training history and predictions

## Dataset

The model uses the Wyndham City Council waste bin dataset:
- **Period**: June 2018 - May 2021
- **Bins**: 31 smart waste bins
- **Records**: 947 daily readings per bin
- **Features**: `latestFullness` (bin fill level 0-10)
- **Target**: Next day's fill level

## Requirements

See `../../requirements.txt` for dependencies. Key requirements:
- tensorflow >= 2.16.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.5.0
- matplotlib >= 3.7.0

## Notes

- The parameter count (~31k) differs from the paper's reported ~39.6k parameters. This is likely due to different TensorFlow/Keras versions or implementation details.
- The architecture (GRU 100 units → Dropout → Dense) matches the paper's specification.
- For reproducibility, random seeds are set to 42.
- Training is performed with deterministic operations enabled (where supported).

## Comparison with Paper

The implementation aims to replicate the paper's GRU model as closely as possible:
- ✓ Same architecture: GRU(100) → Dropout → Dense(1)
- ✓ Same methodology: Individual bin prediction with temporal splitting
- ✓ Same hyperparameters: 20 epochs, batch_size=70, Adam optimizer
- ✓ Same metrics: MAE, MAPE, RMSE, R²
- ~ Parameter count differs (31k vs 39k) due to framework version

## References

Paper: "A Machine Learning Approach to Predicting Waste Bin Fill Levels for Smart Waste Management Systems"
