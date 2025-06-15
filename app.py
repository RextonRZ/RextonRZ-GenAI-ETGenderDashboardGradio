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
        
        all_merged_long_dfs = {}
        
        # Process each question's data
        for q_name, config in questions_config.items():
            file_path = config['file_path']
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found. Skipping {q_name}")
                continue
                
            try:
                # Load the Excel file with all sheets
                excel_data = pd.ExcelFile(file_path)
                
                # Process each metric sheet
                merged_sheets = []
                for sheet_name in selected_metric_sheets:
                    if sheet_name in excel_data.sheet_names:
                        df = pd.read_excel(file_path, sheet_name=sheet_name)
                        df['Metric'] = sheet_name
                        merged_sheets.append(df)
                
                if merged_sheets:
                    combined_df = pd.concat(merged_sheets, ignore_index=True)
                    # Add question identifier
                    combined_df['Question'] = q_name
                    
                    # Merge with participant data
                    if 'Participant_ID' in combined_df.columns:
                        combined_df = combined_df.merge(
                            participant_df_global, 
                            on='Participant_ID', 
                            how='left'
                        )
                    
                    all_merged_long_dfs[q_name] = combined_df
                    
            except Exception as e:
                print(f"Error processing {q_name}: {e}")
                continue

        # Combine all data
        if all_merged_long_dfs:
            final_combined_long_df = pd.concat(all_merged_long_dfs.values(), ignore_index=True)
        else:
            print("No data loaded successfully. Creating sample data.")
            return create_sample_data()

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
                    for metric in selected_metric_sheets:
                        if metric == "Time to first Fixation":
                            value = np.random.exponential(500)  # milliseconds
                        elif metric == "Tot Fixation dur":
                            value = np.random.gamma(2, 200)  # milliseconds
                        elif metric == "Fixation count":
                            value = np.random.poisson(8)  # count
                        else:  # Tot Visit dur
                            value = np.random.gamma(3, 150)  # milliseconds
                        
                        sample_data.append({
                            'Participant_ID': f'P{participant:03d}',
                            'AOI': f'{q}_AOI_{aoi}',
                            'Image_Type': image_type,
                            'Question': q,
                            'Metric': metric,
                            'Value': value,
                            'Gender': np.random.choice(['Male', 'Female']),
                            metric: value  # Add metric as column name too
                        })
    
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
            template="plotly_dark",
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
            template="plotly_dark",
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
            template="plotly_dark",
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
            template="plotly_dark",
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
        
        # Statistics summary
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    f"""
                    ### 📋 Dataset Summary
                    - **Total Questions**: {len(all_merged_long_dfs)}
                    - **Available Metrics**: {len(selected_metric_sheets)}
                    - **Total Records**: {len(final_combined_long_df) if 'final_combined_long_df' in locals() else 'N/A'}
                    """,
                    elem_classes=["panel"]
                )
        
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
        
        # Footer
        gr.Markdown(
            """
            ---
            *Dashboard powered by Gradio & Plotly | Eye-tracking data analysis made interactive*
            """,
            elem_classes=["panel"]
        )
    
    return demo

# --- Create and Export the App for Hugging Face ---
demo = create_gradio_interface()

# Hugging Face Spaces will automatically launch this
if __name__ == "__main__":
    demo.launch()