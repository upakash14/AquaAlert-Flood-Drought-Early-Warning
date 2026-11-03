# AquaAlert — Flood & Drought Early Warning System

**Hackathon:** Neural Storm 2026  
**Track:** Green Planet — GP-03  
**Participant:** Akash U P  
**College:** Malnad College of Engineering

## Overview
AquaAlert is an educational prototype that demonstrates a machine-learning workflow for classifying environmental conditions as Normal, Flood-like, or Drought-like using rainfall, soil moisture, river level, temperature, and humidity.

## Features
- Random Forest classification
- Manual environmental risk assessment
- Illustrative Karnataka location map
- Climate data charts and sample dataset
- Downloadable synthetic demonstration dataset

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data and safety disclosure
The built-in dataset is synthetically generated using demonstration rules. It is not observed climate data. The displayed score measures agreement with synthetic labels only and is not evidence of real-world forecasting accuracy. The app does not provide official alerts and must not be used for emergency decisions.

A real deployment would require verified historical climate observations, time-aware validation, regional calibration, suitable geospatial data, and integration with official agencies and alert feeds.
