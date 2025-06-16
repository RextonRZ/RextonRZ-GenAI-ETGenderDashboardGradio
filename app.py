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
import re
import traceback

warnings.filterwarnings('ignore')

# --- Styling and Global Configuration ---
MODERN_COLORS = {
    'primary': '#00D4FF', 'secondary': '#FF6B6B', 'accent': '#4ECDC4', 'dark': '#1A1A2E',
    'light': '#16213E', 'success': '#00F5A0', 'warning': '#FFD93D', 'text': '#FFFFFF'
}
gender_palette = {'Male': MODERN_COLORS['primary'], 'Female': MODERN_COLORS['secondary']}

# --- DATA LOADING AND PROCESSING (Faithfully replicating the notebook) ---
def load_and_process_data():
    """Performs the entire notebook's data pipeline and returns final dataframes."""
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
            participant_file, sheet_name='GENAI', header=2, usecols=['Gender', 'Participant ID']
        )
        participant_df_global = participant_df_global.rename(
            columns={'Participant ID': 'Participant_ID'}
        ).dropna(subset=['Gender', 'Participant_ID']).drop_duplicates(subset='Participant_ID', keep='first')
        print("Participant data loaded successfully.")
        
        questions_config = {
            'Q1': {'file_path': os.path.join(base_path, 'Filtered_GenAI_Metrics_cleaned_Q1.xlsx'), 'aoi_columns': ['1 Eyebrow A', '1 Eyebrow B', '1 Eyes A', '1 Eyes B', '1 Hair A', '1 Hair B', '1 Nose A', '1 Nose B']},
            'Q2': {'file_path': os.path.join(base_path, 'Filtered_GenAI_Metrics_cleaned_Q2.xlsx'), 'aoi_columns': ['2 Body A', '2 Body B', '2 Face A', '2 Face B', '2 Hair A', '2 Hair B']},
            'Q3': {'file_path': os.path.join(base_path, 'Filtered_GenAI_Metrics_cleaned_Q3.xlsx'), 'aoi_columns': ['3 Back Mountain A', '3 Back Mountain B', '3 Front Mountain A', '3 Front Mountain B', '3 Midground A', '3 Midground B', '3 Plain A', '3 River B', '3 Sky A', '3 Sky B']},
            'Q4': {'file_path': os.path.join(base_path, 'Filtered_GenAI_Metrics_cleaned_Q4.xlsx'), 'aoi_columns': ['4 Chilli B', '4 Jalapeno B', '4 Mushroom A1', '4 Mushroom A2', '4 Mushroom B', '4 Olive A', '4 Pepperoni A', '4 Pepperoni B']},
            'Q5': {'file_path': os.path.join(base_path, 'Filtered_GenAI_Metrics_cleaned_Q5.xlsx'), 'aoi_columns': ['5 Sea A', '5 Sea B', '5 Sky A', '5 Sky B']},
            'Q6': {'file_path': os.path.join(base_path, 'Filtered_GenAI_Metrics_cleaned_Q6.xlsx'), 'aoi_columns': ['6 Background B1','6 Background B2','6 Flower A', '6 Flower B', '6 Inside A', '6 Inside B', '6 Leaf A', '6 Leaf B', '6 Sky A', '6 Sky B']}
        }
        
        selected_metric_sheets = ["Tot Fixation dur", "Fixation count", "Time to first Fixation", "Tot Visit dur"]
        master_sheet_for_balancing = 'Tot Fixation dur'

        # Stage 1: Load Raw Data Sheets
        all_data_sheets = {}
        for q_name, config in questions_config.items():
            if os.path.exists(config['file_path']):
                xls = pd.ExcelFile(config['file_path'])
                all_data_sheets[q_name] = {sheet: xls.parse(sheet) for sheet in xls.sheet_names if sheet in selected_metric_sheets}
            else:
                all_data_sheets[q_name] = {}
        
        # Stage 2: Clean and Merge Gender
        all_cleaned_metrics_dfs = {}
        for q_name, data_sheets_qN in all_data_sheets.items():
            cleaned_metrics_dfs_qN = {}
            for sheet_name, df_qN in data_sheets_qN.items():
                if 'Participant' in df_qN.columns:
                    df_qN = df_qN.rename(columns={'Participant': 'Participant_ID'})
                if 'Participant_ID' in df_qN.columns:
                    df_qN['Participant_ID'] = df_qN['Participant_ID'].apply(lambda x: f'P{int(str(x)[1:]):02d}' if isinstance(x, str) and str(x).startswith('P') and str(x)[1:].isdigit() else (f'P{int(x):02d}' if pd.notna(x) and isinstance(x, (int, float)) else x))
                    df_qN = df_qN.merge(participant_df_global, on='Participant_ID', how='left')
                    cleaned_metrics_dfs_qN[sheet_name] = df_qN.dropna(subset=['Participant_ID', 'Gender'])
            all_cleaned_metrics_dfs[q_name] = cleaned_metrics_dfs_qN
        print("Data cleaning and gender merge complete.")

        # Stage 3: SMOTE Balancing and Reconstruction
        all_balanced_unified_dfs = {}
        for q_name, config in questions_config.items():
            cleaned_metrics_qN = all_cleaned_metrics_dfs.get(q_name, {})
            if master_sheet_for_balancing not in cleaned_metrics_qN or cleaned_metrics_qN[master_sheet_for_balancing].empty:
                all_balanced_unified_dfs[q_name] = cleaned_metrics_qN
                continue

            df_master = cleaned_metrics_qN[master_sheet_for_balancing]
            aoi_cols = [col for col in config['aoi_columns'] if col in df_master.columns]
            
            df_repr = df_master.groupby('Participant_ID').first().reset_index()
            X = df_repr[aoi_cols].fillna(0)
            y = df_repr['Gender']
            
            unified_resampled_master_set = None
            if y.nunique() >= 2 and y.value_counts().min() >= 2:
                try:
                    smote = SMOTE(random_state=42, k_neighbors=y.value_counts().min() - 1)
                    X_res, y_res = smote.fit_resample(X, y)
                    df_resampled = pd.DataFrame(X_res, columns=aoi_cols)
                    df_resampled['Gender'] = y_res
                    df_resampled['Participant_ID'] = [f"Balanced_{q_name}_P{i:03d}" for i in range(len(df_resampled))]
                    unified_resampled_master_set = df_resampled
                except Exception:
                    unified_resampled_master_set = df_repr
            else:
                unified_resampled_master_set = df_repr

            current_q_reconstructed_dfs = {}
            for sheet_name in selected_metric_sheets:
                df_orig = cleaned_metrics_qN.get(sheet_name)
                if df_orig is None or df_orig.empty: continue
                
                if sheet_name == master_sheet_for_balancing:
                    current_q_reconstructed_dfs[sheet_name] = unified_resampled_master_set
                else:
                    sheet_aoi_cols = [c for c in config['aoi_columns'] if c in df_orig.columns]
                    reconstructed_rows = []
                    gender_means = df_orig.groupby('Gender')[sheet_aoi_cols].mean()
                    for _, master_row in unified_resampled_master_set.iterrows():
                        new_row = {'Participant_ID': master_row['Participant_ID'], 'Gender': master_row['Gender']}
                        new_row.update(gender_means.loc[master_row['Gender']].to_dict())
                        reconstructed_rows.append(new_row)
                    current_q_reconstructed_dfs[sheet_name] = pd.DataFrame(reconstructed_rows)
            all_balanced_unified_dfs[q_name] = current_q_reconstructed_dfs
        print("Data balancing and reconstruction complete.")

        # Stage 4: Melt and Combine
        all_merged_long_dfs = {}
        for q_name, config in questions_config.items():
            reconstructed_dfs = all_balanced_unified_dfs.get(q_name, {})
            if not reconstructed_dfs: continue
            
            list_of_long_dfs = []
            for sheet_name, df_sheet in reconstructed_dfs.items():
                aoi_cols_to_melt = [c for c in config['aoi_columns'] if c in df_sheet.columns]
                if aoi_cols_to_melt:
                    df_long = df_sheet.melt(id_vars=['Participant_ID', 'Gender'], value_vars=aoi_cols_to_melt, var_name='AOI', value_name=sheet_name)
                    list_of_long_dfs.append(df_long)
            
            if list_of_long_dfs:
                merged_df = reduce(lambda left, right: pd.merge(left, right, on=['Participant_ID', 'Gender', 'AOI'], how='outer'), list_of_long_dfs)
                
                # Add image type
                def get_image_type(aoi_name_str):
                    if isinstance(aoi_name_str, str):
                        if re.search(r'\sA\d*$', aoi_name_str.strip()): return 'Real'
                        if re.search(r'\sB\d*$', aoi_name_str.strip()): return 'AI'
                    return 'Unknown'
                merged_df['Image_Type'] = merged_df['AOI'].apply(get_image_type)
                
                all_merged_long_dfs[q_name] = merged_df

        # Stage 5: Final Combination
        final_combined_long_df = pd.concat([
            df.assign(Question=q) for q, df in all_merged_long_dfs.items() if not df.empty
        ], ignore_index=True)

        print(f"--- Data processing finished. Loaded {len(all_merged_long_dfs)} question sets ---")
        return all_merged_long_dfs, final_combined_long_df, selected_metric_sheets

    except Exception as e:
        print(f"ERROR during data processing: {e}\n{traceback.format_exc()}")
        print("Creating sample data for demonstration...")
        return create_sample_data()

def create_sample_data():
    """Creates sample dataframes if the main data loading fails."""
    questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']
    all_merged_long_dfs = {}
    all_q_dfs = []
    selected_metric_sheets = ["Tot Fixation dur", "Fixation count", "Time to first Fixation", "Tot Visit dur"]
    
    for q in questions:
        data = []
        for p in range(50):
            gender = 'Male' if p % 2 == 0 else 'Female'
            for aoi_type in ['Real', 'AI']:
                for aoi_num in range(5):
                    aoi_name = f"{q} AOI {aoi_num} {'A' if aoi_type == 'Real' else 'B'}"
                    data.append({
                        'Participant_ID': f'Sample_P{p}', 'Gender': gender, 'AOI': aoi_name, 'Image_Type': aoi_type,
                        'Tot Fixation dur': np.random.gamma(2, 0.7 if gender == 'Female' else 0.8),
                        'Fixation count': np.random.gamma(2.5, 1.1 if gender == 'Female' else 1.0),
                        'Time to first Fixation': np.random.gamma(1.5, 0.5),
                        'Tot Visit dur': np.random.gamma(3, 0.8 if gender == 'Female' else 0.9)
                    })
        df = pd.DataFrame(data)
        all_merged_long_dfs[q] = df
        all_q_dfs.append(df.assign(Question=q))
    
    final_combined_long_df = pd.concat(all_q_dfs, ignore_index=True)
    return all_merged_long_dfs, final_combined_long_df, selected_metric_sheets


# --- Plotting Functions (Replicated from Notebook) ---
def _create_4_panel_dashboard(data, selected_metric, plot_title_suffix):
    """Creates the 4-panel dashboard figure."""
    if data is None or data.empty or selected_metric not in data.columns:
        return go.Figure().add_annotation(text="No data for this view", showarrow=False)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(f'{selected_metric} by Image Type & Gender', f'{selected_metric} Violin Plot', f'"{selected_metric}" Distribution by Gender', 'Summary Statistics'),
        specs=[[{"type": "box"}, {"type": "violin"}], [{"type": "histogram"}, {"type": "table"}]]
    )
    
    # Panel 1: Grouped Box Plot
    for gender in ['Male', 'Female']:
        subset = data[data['Gender'] == gender]
        fig.add_trace(go.Box(
            y=subset[selected_metric], x=subset['Image_Type'], name=gender,
            marker_color=gender_palette.get(gender), legendgroup=gender, showlegend=True, boxpoints='outliers'
        ), row=1, col=1)
    fig.update_layout(boxmode='group', xaxis1_title='Image Type')
        
    # Panel 2: Split Violin Plot
    for gender in ['Male', 'Female']:
        for img_type in data['Image_Type'].unique():
            subset = data[(data['Image_Type'] == img_type) & (data['Gender'] == gender)]
            if not subset.empty:
                fig.add_trace(go.Violin(
                    y=subset[selected_metric], x0=str(img_type), name=gender,
                    side='negative' if gender == 'Male' else 'positive',
                    marker_color=gender_palette.get(gender), points=False,
                    legendgroup=gender, showlegend=False, meanline_visible=True
                ), row=1, col=2)
    fig.update_layout(violinmode='overlay', xaxis2_title='Image Type')
        
    # Panel 3: Overlapping Histogram
    for gender in ['Male', 'Female']:
        subset = data[data['Gender'] == gender]
        fig.add_trace(go.Histogram(
            x=subset[selected_metric], name=gender, marker_color=gender_palette.get(gender),
            legendgroup=gender, showlegend=False, opacity=0.7
        ), row=2, col=1)
    fig.update_layout(barmode='overlay')

    # Panel 4: Summary Table (with improved styling)
    summary_stats = data.groupby(['Image_Type', 'Gender'])[selected_metric].agg(['count', 'mean', 'std', 'min', 'max']).round(2).reset_index()
    fig.add_trace(go.Table(
        header=dict(
            values=[f'<b>{c.upper()}</b>' for c in summary_stats.columns],
            fill_color=MODERN_COLORS['primary'], font_color='white', align='center',
            font=dict(size=12)
        ),
        cells=dict(
            values=[summary_stats[c] for c in summary_stats.columns],
            fill_color='rgba(40,40,60,0.8)', font_color='white', align='center',
            font=dict(size=11)
        )
    ), row=2, col=2)

    # Final Layout
    fig.update_layout(
        height=850, plot_bgcolor='#000', paper_bgcolor='#000',
        font_color='white', title_text=f'"{selected_metric}" - Analysis Dashboard {plot_title_suffix}',
        title_x=0.5, title_font_size=20,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text=selected_metric, row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    return fig

def _create_correlation_heatmap(data, numeric_metrics, plot_title_suffix):
    """Creates the correlation heatmap figure."""
    genders_present = sorted([g for g in data['Gender'].unique() if pd.notna(g)])
    if not genders_present or len(numeric_metrics) < 2:
        return go.Figure().add_annotation(text="Not enough data for heatmap", showarrow=False)

    fig = make_subplots(
        rows=1, cols=len(genders_present),
        subplot_titles=[f"Metric Correlation ({gender})" for gender in genders_present]
    )

    for i, gender in enumerate(genders_present):
        col = i + 1
        subset_corr = data[data['Gender'] == gender][numeric_metrics]
        if not subset_corr.empty:
            corr_matrix = subset_corr.corr()
            fig.add_trace(go.Heatmap(
                z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                colorscale='RdBu_r', zmin=-1, zmax=1, text=corr_matrix.values,
                texttemplate="%{text:.2f}", textfont={"size":9},
                hovertemplate='Metric 1: %{y}<br>Metric 2: %{x}<br>Correlation: %{z:.2f}<extra></extra>'
            ), row=1, col=col)
    
    fig.update_layout(
        height=500, plot_bgcolor='#000', paper_bgcolor='#000',
        font_color='white', title_text=f"Correlation Heatmaps by Gender {plot_title_suffix}",
        title_x=0.5, title_font_size=20
    )
    return fig

def create_modern_bar_plot(data, metric, agg_func, plot_title_suffix):
    """Creates bar plot, using notebook's logic."""
    if data is None or data.empty or metric not in data.columns:
        return go.Figure().add_annotation(text="No data for bar plot", showarrow=False)
    
    aoi_gender_summary = data.groupby(['Gender', 'AOI', 'Image_Type'], as_index=False).agg({metric: agg_func}).sort_values(by=['AOI', 'Gender'])
    fig = px.bar(aoi_gender_summary, x='AOI', y=metric, color='Gender', color_discrete_map=gender_palette, title=f'{metric} ({agg_func.capitalize()}) per AOI {plot_title_suffix}', height=500, barmode='group')
    fig.update_layout(plot_bgcolor='#000', paper_bgcolor='#000', font_color='white', title_x=0.5, xaxis_tickangle=-45)
    return fig

def create_combined_bar_plot(data, metric, agg_func, plot_title_suffix):
    """Creates combined bar plot, using notebook's logic."""
    if data is None or data.empty or metric not in data.columns:
        return go.Figure().add_annotation(text="No data for bar plot", showarrow=False)
        
    summary = data.groupby(['Image_Type', 'Gender'], as_index=False).agg({metric: agg_func})
    fig = px.bar(summary, x='Image_Type', y=metric, color='Gender', color_discrete_map=gender_palette, title=f'{metric} ({agg_func.capitalize()}) by Image Type {plot_title_suffix}', height=500, barmode='group')
    fig.update_layout(plot_bgcolor='#000', paper_bgcolor='#000', font_color='white', title_x=0.5, xaxis_title='Image Type')
    return fig

def create_modern_scatter_plot(data, dur_col, count_col, plot_title_suffix):
    """Creates scatter plot, using notebook's logic."""
    if data is None or data.empty or dur_col not in data.columns or count_col not in data.columns:
        return go.Figure().add_annotation(text="No data for scatter plot", showarrow=False)

    valid_data = data.dropna(subset=[dur_col, count_col])
    if valid_data.empty: return go.Figure().add_annotation(text="No valid data points", showarrow=False)
    
    fig = px.scatter(valid_data, x=dur_col, y=count_col, color='Gender', symbol='Image_Type', title=f'Interactive Scatter: {count_col} vs {dur_col} {plot_title_suffix}', hover_data=['Participant_ID', 'AOI'], color_discrete_map=gender_palette, height=600)
    fig.update_layout(plot_bgcolor='#000', paper_bgcolor='#000', font_color='white', title_x=0.5, legend=dict(bgcolor='rgba(0,0,0,0)'))
    return fig

# --- Load Data on Startup ---
print("Loading and processing data. This may take a moment...")
all_merged_long_dfs, final_combined_long_df, selected_metric_sheets = load_and_process_data()

# --- Main Dashboard Function ---
def update_dashboard(question, metric):
    """Update all charts based on user selection and return a tuple of plots."""
    try:
        df_to_plot = final_combined_long_df if question == 'All Combined' else all_merged_long_dfs.get(question, pd.DataFrame())
        
        if df_to_plot.empty:
            # Return empty plots if no data
            empty_fig = go.Figure().add_annotation(text="No data available", showarrow=False)
            return empty_fig, empty_fig, empty_fig, empty_fig
        
        agg_func = 'mean' if 'Time to first Fixation' in metric else 'sum'
        plot_title_suffix = f"({question})"
        
        # Create plots
        if question != 'All Combined':
            bar_chart = create_modern_bar_plot(df_to_plot, metric, agg_func, plot_title_suffix)
        else:
            bar_chart = create_combined_bar_plot(df_to_plot, metric, agg_func, plot_title_suffix)
            
        scatter_chart = create_modern_scatter_plot(df_to_plot, 'Tot Fixation dur', 'Fixation count', plot_title_suffix)
        dashboard_chart = _create_4_panel_dashboard(df_to_plot, metric, plot_title_suffix)
        heatmap_chart = _create_correlation_heatmap(df_to_plot, selected_metric_sheets, plot_title_suffix)
        
        return dashboard_chart, heatmap_chart, bar_chart, scatter_chart
    except Exception as e:
        print(f"Error in update_dashboard: {e}")
        empty_fig = go.Figure().add_annotation(text=f"Error: {str(e)}", showarrow=False)
        return empty_fig, empty_fig, empty_fig, empty_fig

# --- Create Gradio Interface ---
def create_gradio_interface():
    """Create the Gradio interface with improved layout and styling."""
    question_options = ['All Combined'] + sorted(list(all_merged_long_dfs.keys()))
    
    # Create custom theme
    theme = gr.themes.Default(
        primary_hue="blue",
        secondary_hue="purple"
    )
    
    with gr.Blocks(theme=theme, title="Eye-Tracking Analytics Dashboard") as demo:
        # Header with custom HTML and CSS
        gr.HTML("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 20px;'>
            <h1 style='color: white; font-size: 2.5em; margin: 0; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
                🧠 Eye-Tracking Analytics Dashboard
            </h1>
            <p style='color: rgba(255,255,255,0.9); font-size: 1.2em; margin: 10px 0 0 0; font-weight: 300;'>
                Advanced Visual Analytics & Data Exploration Platform
            </p>
        </div>
        """)
        
        with gr.Row():
            question_select = gr.Dropdown(
                choices=question_options, 
                value=question_options[0] if question_options else "All Combined", 
                label="📋 Select Question Set"
            )
            metric_select = gr.Dropdown(
                choices=selected_metric_sheets, 
                value=selected_metric_sheets[0] if selected_metric_sheets else "Tot Fixation dur", 
                label="📊 Select Metric"
            )
        
        with gr.Accordion("📊 Interactive Bar Chart Analysis", open=True):
            bar_plot = gr.Plot(label="Bar Chart")
        
        with gr.Accordion("🔍 Correlation Scatter Analysis", open=False):
            scatter_plot = gr.Plot(label="Scatter Plot")
        
        with gr.Accordion("📈 Multi-Dimensional Analysis & Heatmaps", open=False):
            dashboard_plot = gr.Plot(label="Multi-Dimensional Dashboard")
            heatmap_plot = gr.Plot(label="Correlation Heatmap")
            
        # Wire up components
        inputs = [question_select, metric_select]
        outputs = [dashboard_plot, heatmap_plot, bar_plot, scatter_plot]
        
        # Event handlers
        question_select.change(fn=update_dashboard, inputs=inputs, outputs=outputs)
        metric_select.change(fn=update_dashboard, inputs=inputs, outputs=outputs)
        
        # Load initial data
        demo.load(fn=update_dashboard, inputs=inputs, outputs=outputs)
        
    return demo

# --- Run the App ---
if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch(debug=True, share=False)