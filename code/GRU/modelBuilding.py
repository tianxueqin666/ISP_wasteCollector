"""
@file modelBuilding.py
@brief Build a GRU model for waste bin fill level prediction
       according to the architecture described in the paper:
       "A Machine Learning Approach to Predicting Waste Bin Fill Levels
        for Smart Waste Management Systems"
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_gru_model(n_steps: int, n_features: int = 1) -> keras.Model:
    """
    Build GRU model according to the paper's architecture.

    Paper specifications:
    - GRU layer with 100 units (output shape: (None, 100))
    - Dropout layer
    - Dense output layer (1 unit)
    - Paper reports ~39,600 params for GRU (actual may vary by TF version)
    
    GRU equations from paper:
    - Update gate: z_t = σ(W_z·x_t + V_z·h_{t-1} + b_z)
    - Reset gate: r_t = σ(W_r·x_t + V_r·h_{t-1} + b_r)
    - Candidate activation: h̃_t = tanh(W_c·x_t + V_c·(r_t · h_{t-1}))
    - Final hidden state: h_t = z_t · h_{t-1} + (1 - z_t) · h̃_t
    
    Args:
        n_steps (int): Number of time steps in the input sequences.
        n_features (int): Number of features per time step. Default is 1.
    Returns:
        keras.Model: Compiled GRU model.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(n_steps, n_features)),
            # GRU layer - 100 units as specified in paper
            # Output shape: (None, 100)
            # Using standard GRU implementation with tanh activation and sigmoid gates
            layers.GRU(
                units=100,
                activation='tanh',
                recurrent_activation='sigmoid',
                name="gru",
            ),
            # Dropout layer for regularization
            # Paper may have used dropout rate between 0.2-0.5
            layers.Dropout(0.2, name="dropout"),
            # Dense output layer (1 unit for regression)
            # 101 parameters: 100 weights + 1 bias
            layers.Dense(1, name="dense"),
        ]
    )

    return model
