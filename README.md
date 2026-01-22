# MWB Bubble Chart - Equipment Data Analysis

This project analyzes equipment operational data including NOx emissions, fuel consumption, and CO2 metrics.

## Setup

### Virtual Environment

A Python virtual environment has been created for this project with all necessary dependencies.

**Activate the virtual environment:**

```bash
source venv/bin/activate
```

**Deactivate when done:**

```bash
deactivate
```

### Installed Packages

- pandas >= 2.0.0 - Data manipulation and analysis
- numpy >= 1.24.0 - Numerical computing
- plotly >= 5.14.0 - Interactive visualizations
- jupyter >= 1.0.0 - Jupyter notebook support
- notebook >= 6.5.0 - Notebook interface
- ipykernel >= 6.21.0 - IPython kernel for Jupyter
- kaleido >= 0.2.1 - Static image export for Plotly

## Running the Analysis

1. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Open [analysis.ipynb](analysis.ipynb) in the browser that opens

4. Run the cells to perform the analysis

## Project Structure

```
MWB bubble chart/
├── venv/                    # Virtual environment (do not commit)
├── Exploratory/            # Data files
│   └── untitled - 2025-11-18T141129.591.csv
├── analysis.ipynb          # Main analysis notebook
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Data Overview

The dataset contains equipment operational metrics including:
- Equipment identification and classification
- NOx and CO2 emissions
- Fuel consumption from multiple sources (FF, CANBUS, NOxMAF)
- Engine load and power metrics
- Operational duration
- Pilot program information

## Analysis Sections

The notebook includes 15 comprehensive analysis sections:

1. Data Loading & Exploration
2. Equipment Overview
3. NOx Emissions Analysis
4. Fuel Consumption Analysis
5. Engine Load Analysis
6. CO2 Emissions Analysis
7. Operational Duration Analysis
8. Engine Classification Comparison
9. Pilot Program Comparison
10. Correlation Analysis
11. Summary Statistics by Equipment Type
12. Time-based Analysis
13. Equipment Efficiency Analysis
14. Load Type Analysis
15. Key Insights & Conclusions

All visualizations are created using Plotly for interactive exploration.
