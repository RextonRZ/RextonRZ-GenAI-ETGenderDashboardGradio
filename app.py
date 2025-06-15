# ==============================================================================
# FINAL PANEL DASHBOARD SCRIPT (app.py)
# ==============================================================================

# --- 1. Imports and Setup ---
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import panel as pn
import warnings
import re
import os
from functools import reduce
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
pn.extension('plotly', sizing_mode="stretch_width", design="dark", template="material")

# --- 2. Styling and Global Configuration ---
MODERN_COLORS = {
    'primary': '#00D4FF', 'secondary': '#FF6B6B', 'accent': '#4ECDC4', 'dark': '#1A1A2E',
    'light': '#16213E', 'success': '#00F5A0', 'warning': '#FFD93D', 'text': '#FFFFFF'
}
gender_palette = {'Male': MODERN_COLORS['primary'], 'Female': MODERN_COLORS['secondary']}

# --- 3. DATA LOADING AND PROCESSING (Runs ONCE on app startup) ---
@pn.cache
def load_and_process_data():
    """Performs the entire data loading and processing pipeline and returns the final dataframes."""
    print("--- Starting data loading and processing (this runs only once) ---")
    
    base_path = 'GenAIEyeTrackingCleanedDataset/'
    questions_config = {
        'Q1': {'file_path': os.path.join(base_path, 'Filtered_GenAI_Metrics_cleaned_Q1.xlsx'), 'aoi_columns': ['1 Eyebrow A', '1 Eyebrow B', '1 Eyes A', '1 Eyes B', '1 Hair A', '1 Hair B', '1 Nose A', '1 Nose B']},
        'Q2': {'file_path': os.path.join(base_path, 'Filtered_GenAI_Metrics_cleaned_Q2.xlsx'), 'aoi_columns': ['2 Body A', '2 Body B', '2 Face A', '2 Face B', '2 Hair A', '2 Hair B']},
        'Q3': {'file_path': os.path.join(base_path, 'Filtered_GenAI_Metrics_cleaned_Q3.xlsx'), 'aoi_columns': ['3 Back Mountain A', '3 Back Mountain B', '3 Front Mountain A', '3 Front Mountain B', '3 Midground A', '3 Midground B', '3 Plain A', '3 River B', '3 Sky A', '3 Sky B']},
        'Q4': {'file_path': os.path.join(base_path, 'Filtered_GenAI_Metrics_cleaned_Q4.xlsx'), 'aoi_columns': ['4 Chilli B', '4 Jalapeno B', '4 Mushroom A1', '4 Mushroom A2', '4 Mushroom B', '4 Olive A', '4 Pepperoni A', '4 Pepperoni B']},
        'Q5': {'file_path': os.path.join(base_path, 'Filtered_GenAI_Metrics_cleaned_Q5.xlsx'), 'aoi_columns': ['5 Sea A', '5 Sea B', '5 Sky A', '5 Sky B']},
        'Q6': {'file_path': os.path.join(base_path, 'Filtered_GenAI_Metrics_cleaned_Q6.xlsx'), 'aoi_columns': ['6 Background B1','6 Background B2','6 Flower A', '6 Flower B', '6 Inside A', '6 Inside B', '6 Leaf A', '6 Leaf B', '6 Sky A', '6 Sky B']}
    }
    selected_metric_sheets = ["Tot Fixation dur", "Fixation count", "Time to first Fixation", "Tot Visit dur"]
    
    # (The rest of your data processing pipeline from the notebook goes here)
    # This is a condensed version of your logic to ensure it runs correctly.
    
    participant_df_global = pd.read_excel(os.path.join(base_path, 'ParticipantList.xlsx'), sheet_name='GENAI', header=2, usecols=['Gender', 'Participant ID'])
    participant_df_global = participant_df_global.rename(columns={'Participant ID': 'Participant_ID'}).dropna(subset=['Gender', 'Participant_ID']).drop_duplicates(subset='Participant_ID')
    
    all_cleaned_metrics_dfs = {}
    for q_name, config in questions_config.items():
        try:
            xls = pd.ExcelFile(config['file_path'])
            data_sheets = {sheet: xls.parse(sheet) for sheet in xls.sheet_names if sheet in selected_metric_sheets}
            cleaned_q = {}
            for sheet_name, df in data_sheets.items():
                if 'Participant' in df.columns: df = df.rename(columns={'Participant': 'Participant_ID'})
                if 'Participant_ID' in df.columns:
                    df['Participant_ID'] = df['Participant_ID'].apply(lambda x: f'P{int(str(x)[1:]):02d}' if isinstance(x, str) and str(x).startswith('P') and x[1:].isdigit() else (f'P{int(x):02d}' if pd.notna(x) and isinstance(x, (int, float)) else x))
                    df_merged = df.merge(participant_df_global, on='Participant_ID', how='left')
                    cleaned_q[sheet_name] = df_merged.dropna(subset=['Participant_ID', 'Gender'])
            all_cleaned_metrics_dfs[q_name] = cleaned_q
        except FileNotFoundError:
            all_cleaned_metrics_dfs[q_name] = {}
    
    all_merged_long_dfs = {}
    for q_name, cleaned_metrics in all_cleaned_metrics_dfs.items():
        if not cleaned_metrics: continue
        # Simplified melting for deployment speed
        long_dfs = []
        for sheet_name, df_sheet in cleaned_metrics.items():
            aoi_cols = [c for c in questions_config[q_name]['aoi_columns'] if c in df_sheet.columns]
            if aoi_cols:
                id_vars = ['Participant_ID', 'Gender']
                df_long = df_sheet.melt(id_vars=id_vars, value_vars=aoi_cols, var_name='AOI', value_name=sheet_name)
                long_dfs.append(df_long)
        
        if long_dfs:
            merged_df = reduce(lambda left, right: pd.merge(left, right, on=['Participant_ID', 'Gender', 'AOI'], how='outer'), long_dfs)
            merged_df['Image_Type'] = merged_df['AOI'].apply(lambda a: 'AI' if ' B' in str(a) else 'Real')
            all_merged_long_dfs[q_name] = merged_df

    all_q_dfs = [df.copy().assign(Question=q) for q, df in all_merged_long_dfs.items() if not df.empty]
    final_combined_long_df = pd.concat(all_q_dfs, ignore_index=True) if all_q_dfs else pd.DataFrame()
    
    print("--- Data processing finished. ---")
    return all_merged_long_dfs, final_combined_long_df, selected_metric_sheets

# --- 4. Plotting Functions (Unchanged, they are correct) ---
# (Paste all your create_..._plot and _create_..._dashboard functions here)
# ...

# --- 5. Main App Body ---
# Load the data when the script starts
all_merged_long_dfs, final_combined_long_df, selected_metric_sheets = load_and_process_data()

# Create Widgets
question_options = ['All Combined'] + list(all_merged_long_dfs.keys())
question_select = pn.widgets.Select(name='📋 Select Question Set', options=question_options, value=question_options[0])
metric_select = pn.widgets.Select(name='📊 Select Metric', options=selected_metric_sheets)

# Define functions that update plots based on widget values
@pn.depends(question_select.param.value, metric_select.param.value)
def get_bar_chart(question, metric):
    df = final_combined_long_df if question == 'All Combined' else all_merged_long_dfs.get(question)
    if df is None or df.empty: return pn.pane.Alert("No data available for this selection.", alert_type='warning')
    agg = 'mean' if 'Time to first Fixation' in metric else 'sum'
    if question != 'All Combined':
        return create_modern_bar_plot(df, metric, agg, f"({question})")
    else:
        return create_combined_bar_plot(df, metric, agg, f"({question})")

@pn.depends(question_select.param.value)
def get_scatter_plot(question):
    df = final_combined_long_df if question == 'All Combined' else all_merged_long_dfs.get(question)
    if df is None or df.empty: return pn.pane.Alert("No data available for this selection.", alert_type='warning')
    return create_modern_scatter_plot(df, 'Tot Fixation dur', 'Fixation count', f"({question})")

@pn.depends(question_select.param.value, metric_select.param.value)
def get_comparison_dashboard(question, metric):
    df = final_combined_long_df if question == 'All Combined' else all_merged_long_dfs.get(question)
    if df is None or df.empty: return pn.pane.Alert("No data available for this selection.", alert_type='warning')
    # The comparison function returns two figures, we'll display them in a column
    dash, heat = create_comparison_dashboard(df, metric, selected_metric_sheets, f"({question})")
    return pn.Column(dash, heat, sizing_mode='stretch_width')

# --- 6. Define the Dashboard Layout ---
header = pn.pane.Markdown("""
# 🧠 Eye-Tracking Analytics Dashboard
### Advanced Visual Analytics & Data Exploration Platform
""", styles={'text-align': 'center'})

controls = pn.Row(question_select, metric_select, styles={'justify-content': 'center'})

# Assemble the final app layout
pn.Column(
    header,
    controls,
    pn.pane.Markdown("---"),
    pn.pane.Markdown("## Interactive Bar Chart Analysis", styles={'color': MODERN_COLORS['primary']}),
    get_bar_chart,
    pn.pane.Markdown("---"),
    pn.pane.Markdown("## Correlation Scatter Analysis", styles={'color': MODERN_COLORS['primary']}),
    get_scatter_plot,
    pn.pane.Markdown("---"),
    pn.pane.Markdown("## Multi-Dimensional Analysis & Heatmaps", styles={'color': MODERN_COLORS['primary']}),
    get_comparison_dashboard,
).servable(title="Eye-Tracking Dashboard")