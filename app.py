# ==============================================================================
# GRADIO DASHBOARD FOR HUGGING FACE DEPLOYMENT
# ==============================================================================

import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
from functools import reduce
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# --- Styling and Global Configuration ---
MODERN_COLORS = {
    'primary': '#00D4FF', 'secondary': '#FF6B6B', 'accent': '#4ECDC4', 'dark': '#1A1A2E',
    'light': '#16213E', 'success': '#00F5A0', 'warning': '#FFD93D', 'text': '#FFFFFF'
}
gender_palette = {'Male': MODERN_COLORS['primary'], 'Female': MODERN_COLORS['secondary']}

# --- DATA LOADING AND PROCESSING ---
def load_and_process_data():
    """Performs the entire data loading and processing pipeline and returns the final dataframes."""
    print("--- Starting data loading and processing ---")
    
    try:
        base_path = 'GenAIEyeTrackingCleanedDataset/'
        
        # Check if data directory exists
        if not os.path.exists(base_path):
            print(f"Warning: Data directory {base_path} not found. Creating sample data.")
            return create_sample_data()
        
        # Load participant data
        participant_file = os.path.join(base_path, 'ParticipantList.xlsx')
        if not os.path.exists(participant_file):
            print(f"Warning: {participant_file} not found. Creating sample data.")
            return create_sample_data()
            
        participant_df_global = pd.read_excel(
            participant_file, 
            sheet_name='GENAI', 
            header=2, 
            usecols=['Gender', 'Participant ID']
        )
        participant_df_global = participant_df_global.rename(
            columns={'Participant ID': 'Participant_ID'}
        ).dropna(subset=['Gender', 'Participant_ID']).drop_duplicates(subset='Participant_ID')
        print("Participant data loaded successfully.")
        
        questions_config = {
            'Q1': {
                'file_path': os.path.join(base_path, 'Filtered_GenAI_Metrics_cleaned_Q1.xlsx'), 
                'aoi_columns': ['1 Eyebrow A', '1 Eyebrow B', '1 Eyes A', '1 Eyes B', '1 Hair A', '1 Hair B', '1 Nose A', '1 Nose B']
            },
            'Q2': {
                'file_path': os.path.join(base_path, 'Filtered_GenAI_Metrics_cleaned_Q2.xlsx'), 
                'aoi_columns': ['2 Body A', '2 Body B', '2 Face A', '2 Face B', '2 Hair A', '2 Hair B']
            },
            'Q3': {
                'file_path': os.path.join(base_path, 'Filtered_GenAI_Metrics_cleaned_Q3.xlsx'), 
                'aoi_columns': ['3 Back Mountain A', '3 Back Mountain B', '3 Front Mountain A', '3 Front Mountain B', '3 Midground A', '3 Midground B', '3 Plain A', '3 River B', '3 Sky A', '3 Sky B']
            },
            'Q4': {
                'file_path': os.path.join(base_path, 'Filtered_GenAI_Metrics_cleaned_Q4.xlsx'), 
                'aoi_columns': ['4 Chilli B', '4 Jalapeno B', '4 Mushroom A1', '4 Mushroom A2', '4 Mushroom B', '4 Olive A', '4 Pepperoni A', '4 Pepperoni B']
            },
            'Q5': {
                'file_path': os.path.join(base_path, 'Filtered_GenAI_Metrics_cleaned_Q5.xlsx'), 
                'aoi_columns': ['5 Sea A', '5 Sea B', '5 Sky A', '5 Sky B']
            },
            'Q6': {
                'file_path': os.path.join(base_path, 'Filtered_GenAI_Metrics_cleaned_Q6.xlsx'), 
                'aoi_columns': ['6 Background B1','6 Background B2','6 Flower A', '6 Flower B', '6 Inside A', '6 Inside B', '6 Leaf A', '6 Leaf B', '6 Sky A', '6 Sky B']
            }
        }
        
        selected_metric_sheets = ["Tot Fixation dur", "Fixation count", "Time to first Fixation", "Tot Visit dur"]
        
        # --- Multi-stage processing following the original approach ---
        
        # 1. Load all sheets for each question
        all_data_sheets = {}
        for q_name, config in questions_config.items():
            try:
                if os.path.exists(config['file_path']):
                    xls = pd.ExcelFile(config['file_path'])
                    all_data_sheets[q_name] = {
                        sheet: xls.parse(sheet) for sheet in xls.sheet_names 
                        if sheet in selected_metric_sheets
                    }
                else:
                    print(f"Warning: File not found for {q_name}: {config['file_path']}")
                    all_data_sheets[q_name] = {}
            except Exception as e:
                print(f"Warning: Could not load or parse file for {q_name}: {e}")
                all_data_sheets[q_name] = {}

        # 2. Clean and merge with gender data
        all_cleaned_metrics_dfs = {}
        for q_name, data_sheets in all_data_sheets.items():
            cleaned_qN = {}
            for sheet_name, df in data_sheets.items():
                # Standardize participant column name
                if 'Participant' in df.columns:
                    df = df.rename(columns={'Participant': 'Participant_ID'})
                
                if 'Participant_ID' in df.columns:
                    # Standardize participant ID format
                    df['Participant_ID'] = df['Participant_ID'].apply(
                        lambda x: f'P{int(str(x)[1:]):02d}' if isinstance(x, str) and x.startswith('P') and x[1:].isdigit() 
                        else (f'P{int(x):02d}' if pd.notna(x) and isinstance(x, (int, float)) else x)
                    )
                    
                    # Merge with participant data
                    df_merged = df.merge(participant_df_global, on='Participant_ID', how='left')
                    cleaned_qN[sheet_name] = df_merged.dropna(subset=['Participant_ID', 'Gender'])
            
            all_cleaned_metrics_dfs[q_name] = cleaned_qN
        
        print("Data cleaning and gender merge complete.")

        # 3. Reconstruct balanced data using SMOTE
        all_balanced_unified_dfs = {}
        master_sheet = 'Tot Fixation dur'
        
        for q_name, config in questions_config.items():
            cleaned_metrics_qN = all_cleaned_metrics_dfs.get(q_name, {})
            
            if master_sheet not in cleaned_metrics_qN or cleaned_metrics_qN[master_sheet].empty:
                print(f"Skipping SMOTE for {q_name}: master sheet missing.")
                all_balanced_unified_dfs[q_name] = cleaned_metrics_qN
                continue
            
            df_master = cleaned_metrics_qN[master_sheet]
            aoi_cols = [col for col in config['aoi_columns'] if col in df_master.columns]
            
            # Group by participant and take first occurrence
            df_repr = df_master.groupby('Participant_ID').first().reset_index()
            X = df_repr[aoi_cols].fillna(0)
            y = df_repr['Gender']

            # Apply SMOTE if we have enough samples
            if y.nunique() < 2 or y.value_counts().min() < 2:
                df_resampled = df_repr
            else:
                try:
                    smote = SMOTE(random_state=42, k_neighbors=max(1, y.value_counts().min() - 1))
                    X_res, y_res = smote.fit_resample(X, y)
                    df_resampled = pd.DataFrame(X_res, columns=aoi_cols)
                    df_resampled['Gender'] = y_res
                    df_resampled['Participant_ID'] = [f'Balanced_{q_name}_{i}' for i in range(len(df_resampled))]
                except Exception as e:
                    print(f"SMOTE failed for {q_name}: {e}. Using original data.")
                    df_resampled = df_repr
            
            # Reconstruct other sheets based on gender means
            reconstructed_qN = {}
            for sheet_name, df_orig in cleaned_metrics_qN.items():
                if sheet_name == master_sheet:
                    reconstructed_qN[sheet_name] = df_resampled
                else:
                    sheet_aoi_cols = [c for c in config['aoi_columns'] if c in df_orig.columns]
                    if sheet_aoi_cols:
                        gender_means = df_orig.groupby('Gender')[sheet_aoi_cols].mean()
                        reconstructed_rows = []
                        for _, mr in df_resampled.iterrows():
                            row_data = {
                                'Participant_ID': mr['Participant_ID'], 
                                'Gender': mr['Gender']
                            }
                            row_data.update(gender_means.loc[mr['Gender']])
                            reconstructed_rows.append(row_data)
                        reconstructed_qN[sheet_name] = pd.DataFrame(reconstructed_rows)
            
            all_balanced_unified_dfs[q_name] = reconstructed_qN
        
        print("Data balancing complete.")

        # 4. Melt to long format
        all_merged_long_dfs = {}
        for q_name, config in questions_config.items():
            reconstructed_dfs = all_balanced_unified_dfs.get(q_name, {})
            if not reconstructed_dfs:
                continue
            
            long_dfs = []
            for sheet_name, df_sheet in reconstructed_dfs.items():
                aoi_cols_to_melt = [c for c in config['aoi_columns'] if c in df_sheet.columns]
                if aoi_cols_to_melt:
                    df_long = df_sheet.melt(
                        id_vars=['Participant_ID', 'Gender'], 
                        value_vars=aoi_cols_to_melt, 
                        var_name='AOI', 
                        value_name=sheet_name
                    )
                    long_dfs.append(df_long)
            
            if long_dfs:
                # Merge all metric sheets for this question
                merged_df = reduce(
                    lambda left, right: pd.merge(left, right, on=['Participant_ID', 'Gender', 'AOI'], how='outer'), 
                    long_dfs
                )
                
                # Add image type classification
                merged_df['Image_Type'] = merged_df['AOI'].apply(
                    lambda a: 'AI' if ' B' in str(a) else 'Real'
                )
                
                all_merged_long_dfs[q_name] = merged_df
        
        print("Melting to long format complete.")

        # 5. Create final combined dataframe
        all_q_dfs = []
        for q, df in all_merged_long_dfs.items():
            if not df.empty:
                df_copy = df.copy()
                df_copy['Question'] = q
                all_q_dfs.append(df_copy)
        
        if all_q_dfs:
            final_combined_long_df = pd.concat(all_q_dfs, ignore_index=True)
        else:
            final_combined_long_df = pd.DataFrame()

        print(f"--- Data processing finished. Loaded {len(all_merged_long_dfs)} question sets ---")
        return all_merged_long_dfs, final_combined_long_df, selected_metric_sheets
        
    except Exception as e:
        print(f"ERROR during data processing: {e}")
        print("Creating sample data for demonstration...")
        return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration when real data is not available."""
    print("Creating sample data...")
    
    selected_metric_sheets = ["Tot Fixation dur", "Fixation count", "Time to first Fixation", "Tot Visit dur"]
    
    # Create sample data
    np.random.seed(42)
    n_participants = 50
    n_aois = 10
    
    sample_data = []
    questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']
    
    for q in questions:
        for participant in range(1, n_participants + 1):
            for aoi in range(1, n_aois + 1):
                for image_type in ['A', 'B']:
                    row_data = {
                        'Participant_ID': f'P{participant:03d}',
                        'AOI': f'{q}_AOI_{aoi}_{image_type}',
                        'Image_Type': 'AI' if image_type == 'B' else 'Real',
                        'Question': q,
                        'Gender': np.random.choice(['Male', 'Female'])
                    }
                    
                    # Add metric values
                    for metric in selected_metric_sheets:
                        if metric == "Time to first Fixation":
                            value = np.random.exponential(500)  # milliseconds
                        elif metric == "Tot Fixation dur":
                            value = np.random.gamma(2, 200)  # milliseconds
                        elif metric == "Fixation count":
                            value = np.random.poisson(8)  # count
                        else:  # Tot Visit dur
                            value = np.random.gamma(3, 150)  # milliseconds
                        
                        row_data[metric] = value
                    
                    sample_data.append(row_data)
    
    final_combined_long_df = pd.DataFrame(sample_data)
    
    # Create question-specific dataframes
    all_merged_long_dfs = {}
    for q in questions:
        all_merged_long_dfs[q] = final_combined_long_df[final_combined_long_df['Question'] == q].copy()
    
    print("Sample data created successfully")
    return all_merged_long_dfs, final_combined_long_df, selected_metric_sheets

# --- Plotting Functions ---
def create_modern_bar_plot(data, metric, agg_func, plot_title_suffix):
    """Create a modern bar plot."""
    if data is None or data.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    if metric not in data.columns:
        return go.Figure().add_annotation(text=f"Metric '{metric}' not found in data", showarrow=False)
    
    try:
        # Group and aggregate data
        aoi_summary = data.groupby(['Gender', 'AOI', 'Image_Type'], as_index=False).agg({
            metric: agg_func
        }).sort_values(by=['AOI', 'Gender'])
        
        if aoi_summary.empty:
            return go.Figure().add_annotation(text="No data to display", showarrow=False)
        
        fig = px.bar(
            aoi_summary, 
            x='AOI', 
            y=metric, 
            color='Gender', 
            color_discrete_map=gender_palette,
            title=f'{metric} ({agg_func.capitalize()}) per AOI {plot_title_suffix}',
            height=500,
            barmode='group'
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0.8)',
            paper_bgcolor='rgba(0,0,0,0.8)',
            font_color='white',
            title_x=0.5,
            xaxis_tickangle=-45
        )
        
        return fig
        
    except Exception as e:
        return go.Figure().add_annotation(text=f"Error creating plot: {str(e)}", showarrow=False)

def create_combined_bar_plot(data, metric, agg_func, plot_title_suffix):
    """Create a combined bar plot by image type."""
    if data is None or data.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    if metric not in data.columns:
        return go.Figure().add_annotation(text=f"Metric '{metric}' not found in data", showarrow=False)
    
    try:
        summary = data.groupby(['Image_Type', 'Gender'], as_index=False).agg({
            metric: agg_func
        })
        
        if summary.empty:
            return go.Figure().add_annotation(text="No data to display", showarrow=False)
        
        fig = px.bar(
            summary, 
            x='Image_Type', 
            y=metric, 
            color='Gender', 
            color_discrete_map=gender_palette,
            title=f'{metric} ({agg_func.capitalize()}) by Image Type {plot_title_suffix}',
            height=500,
            barmode='group'
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0.8)',
            paper_bgcolor='rgba(0,0,0,0.8)',
            font_color='white',
            title_x=0.5
        )
        
        return fig
        
    except Exception as e:
        return go.Figure().add_annotation(text=f"Error creating plot: {str(e)}", showarrow=False)

def create_modern_scatter_plot(data, dur_col, count_col, plot_title_suffix):
    """Create a modern scatter plot."""
    if data is None or data.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    if dur_col not in data.columns or count_col not in data.columns:
        return go.Figure().add_annotation(
            text=f"Required columns not found: {dur_col}, {count_col}", 
            showarrow=False
        )
    
    try:
        valid_data = data.dropna(subset=[dur_col, count_col])
        
        if valid_data.empty:
            return go.Figure().add_annotation(text="No valid data points", showarrow=False)
        
        fig = px.scatter(
            valid_data, 
            x=dur_col, 
            y=count_col, 
            color='Gender', 
            symbol='Image_Type',
            title=f'Scatter: {count_col} vs {dur_col} {plot_title_suffix}',
            hover_data=['Participant_ID', 'AOI'] if 'Participant_ID' in valid_data.columns and 'AOI' in valid_data.columns else None,
            color_discrete_map=gender_palette,
            height=600
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0.8)',
            paper_bgcolor='rgba(0,0,0,0.8)',
            font_color='white',
            title_x=0.5
        )
        
        return fig
        
    except Exception as e:
        return go.Figure().add_annotation(text=f"Error creating scatter plot: {str(e)}", showarrow=False)

def create_comparison_dashboard(data, metric, metrics, plot_title_suffix):
    """Create a comprehensive comparison dashboard."""
    if data is None or data.empty or metric not in data.columns:
        return go.Figure().add_annotation(text="No data available for dashboard", showarrow=False)
    
    try:
        clean_data = data.dropna(subset=[metric])
        
        if clean_data.empty:
            return go.Figure().add_annotation(text="No valid data for dashboard", showarrow=False)
        
        # Create 4-panel dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{metric} by Image Type & Gender',
                f'{metric} Distribution',
                'Gender Comparison',
                'Summary Statistics'
            ),
            specs=[
                [{"type": "box"}, {"type": "histogram"}],
                [{"type": "violin"}, {"type": "table"}]
            ]
        )
        
        # Panel 1: Box plot
        if all(c in clean_data.columns for c in ['Image_Type', 'Gender']):
            for gender in clean_data['Gender'].unique():
                subset = clean_data[clean_data['Gender'] == gender]
                fig.add_trace(
                    go.Box(
                        y=subset[metric],
                        x=subset['Image_Type'],
                        name=gender,
                        marker_color=gender_palette.get(gender, '#888888'),
                        showlegend=False
                    ),
                    row=1, col=1
                )
        
        # Panel 2: Histogram
        for gender in clean_data['Gender'].unique():
            fig.add_trace(
                go.Histogram(
                    x=clean_data[clean_data['Gender'] == gender][metric],
                    name=gender,
                    marker_color=gender_palette.get(gender, '#888888'),
                    showlegend=False,
                    opacity=0.7
                ),
                row=1, col=2
            )
        
        # Panel 3: Violin plot
        for gender in clean_data['Gender'].unique():
            fig.add_trace(
                go.Violin(
                    y=clean_data[clean_data['Gender'] == gender][metric],
                    name=gender,
                    fillcolor=gender_palette.get(gender, '#888888'),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Panel 4: Summary table
        try:
            if all(c in clean_data.columns for c in ['Image_Type', 'Gender']):
                summary_stats = clean_data.groupby(['Image_Type', 'Gender'])[metric].agg([
                    'count', 'mean', 'std', 'min', 'max'
                ]).round(2).reset_index()
            else:
                summary_stats = clean_data.groupby('Gender')[metric].agg([
                    'count', 'mean', 'std', 'min', 'max'
                ]).round(2).reset_index()
            
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=[f'<b>{c.upper()}</b>' for c in summary_stats.columns],
                        font=dict(size=11)
                    ),
                    cells=dict(
                        values=[summary_stats[c] for c in summary_stats.columns],
                        font=dict(size=10)
                    )
                ),
                row=2, col=2
            )
        except Exception as table_error:
            print(f"Error creating table: {table_error}")
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0.8)',
            paper_bgcolor='rgba(0,0,0,0.8)',
            font_color='white',
            height=850,
            title_x=0.5,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        return go.Figure().add_annotation(text=f"Error creating dashboard: {str(e)}", showarrow=False)

# --- Load Data on Startup ---
print("Loading data...")
all_merged_long_dfs, final_combined_long_df, selected_metric_sheets = load_and_process_data()
print(f"Data loaded. Available questions: {list(all_merged_long_dfs.keys())}")

# --- Main Dashboard Functions ---
def update_charts(question, metric):
    """Update all charts based on user selection."""
    # Get the appropriate dataset
    if question == 'All Combined':
        df = final_combined_long_df
    else:
        df = all_merged_long_dfs.get(question, pd.DataFrame())
    
    # Determine aggregation function
    agg = 'mean' if 'Time to first Fixation' in metric else 'sum'
    
    # Create the three visualizations
    if question != 'All Combined':
        bar_chart = create_modern_bar_plot(df, metric, agg, f"({question})")
    else:
        bar_chart = create_combined_bar_plot(df, metric, agg, f"({question})")
    
    scatter_chart = create_modern_scatter_plot(df, 'Tot Fixation dur', 'Fixation count', f"({question})")
    
    dashboard_chart = create_comparison_dashboard(df, metric, selected_metric_sheets, f"({question})")
    
    return bar_chart, scatter_chart, dashboard_chart

# --- Create Gradio Interface ---
def create_gradio_interface():
    """Create the Gradio interface."""
    
    # Prepare options
    question_options = ['All Combined'] + list(all_merged_long_dfs.keys()) if all_merged_long_dfs else ['All Combined']
    
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="gray"
        ),
        title="👁️ Eye-Tracking Analytics Dashboard",
        css="""
        .gradio-container {
            background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
        }
        .panel {
            background: rgba(46, 46, 72, 0.8);
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
        }
        """
    ) as demo:
        
        # Header
        gr.Markdown(
            """
            # 👁️ Eye-Tracking Analytics Dashboard
            *Interactive analysis of eye-tracking data across different question sets*
            
            This dashboard provides comprehensive analysis of eye-tracking metrics including fixation duration, 
            fixation count, time to first fixation, and total visit duration across different Areas of Interest (AOIs).
            """,
            elem_classes=["panel"]
        )
        
        # Controls
        with gr.Row():
            question_select = gr.Dropdown(
                choices=question_options,
                value=question_options[0],
                label="📋 Select Question Set",
                info="Choose a specific question set or view all combined data"
            )
            
            metric_select = gr.Dropdown(
                choices=selected_metric_sheets,
                value=selected_metric_sheets[0] if selected_metric_sheets else "Tot Fixation dur",
                label="📊 Select Metric",
                info="Choose the eye-tracking metric to analyze"
            )
        
        # Main visualizations
        with gr.Tabs():
            with gr.Tab("📊 Bar Chart Analysis"):
                gr.Markdown("### Interactive Bar Chart showing metric distributions by gender and AOI")
                bar_plot = gr.Plot(label="Bar Chart")
            
            with gr.Tab("🔍 Correlation Analysis"):
                gr.Markdown("### Scatter plot showing relationship between fixation duration and count")
                scatter_plot = gr.Plot(label="Scatter Plot")
            
            with gr.Tab("📈 Multi-Dimensional Dashboard"):
                gr.Markdown("### Comprehensive analysis with multiple visualization types")
                dashboard_plot = gr.Plot(label="Dashboard")
        
        
        # Set up interactivity
        inputs = [question_select, metric_select]
        outputs = [bar_plot, scatter_plot, dashboard_plot]
        
        # Update charts when inputs change
        question_select.change(fn=update_charts, inputs=inputs, outputs=outputs)
        metric_select.change(fn=update_charts, inputs=inputs, outputs=outputs)
        
        # Initialize with default values
        demo.load(
            fn=update_charts,
            inputs=inputs,
            outputs=outputs
        )
    
    return demo

# --- Create and Export the App for Hugging Face ---
demo = create_gradio_interface()

# Hugging Face Spaces will automatically launch this
if __name__ == "__main__":
    demo.launch()