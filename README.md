# EdgeDyn: Hebbian Plasticity-Driven GNN for Traffic Prediction

## Overview
This project implements a biologically inspired Graph Neural Network (GNN) with Hebbian plasticity and ODE-based node dynamics to model traffic sensor data. The model is trained and tested on the **METR-LA** dataset.

## Data Source

The dataset used is **METR-LA**, a traffic speed dataset collected from loop detectors in the Los Angeles County highway system.

- **Original Source**: [https://github.com/liyaguang/DCRNN](https://github.com/liyaguang/DCRNN)
- **Required files**:
  - `data/metr-la.h5` — traffic time series data
  - `data/sensor_graph/adj_mx.pkl` — adjacency matrix based on sensor distances

## Download Instructions

metr-la.h5: https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX
adj_mx.pkl: https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph

Save both files to: 
data/metr-la.h5
data/sensor_graph/adj_mx.pkl

## Preprocessing

mkdir -p data/METR-LA

python generate_training_data.py \
    --output_dir=data/METR-LA \
    --traffic_df_filename=data/metr-la.h5
