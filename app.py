import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from functools import reduce
import traceback
from IPython.display import display, HTML # For better display of multiple dataframes
from plotly.subplots import make_subplots
import re

# ==============================================================================
# GRADIO DASHBOARD FOR HUGGING FACE DEPLOYMENT WITH LIGHT/DARK MODE TOGGLE
# ==============================================================================

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
from scipy.stats import shapiro, ttest_ind, mannwhitneyu

# Import ML libraries for the new section
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier

# --- Global Data Variables (Initialize them) ---
all_balanced_long_dfs = {}
final_combined_long_df = pd.DataFrame()
all_original_long_dfs = {}
all_cleaned_metrics_dfs = {}
selected_metric_sheets = []
participant_df_global = pd.DataFrame()
questions_config = {}
model = None # For the simple interactive predictor
le = LabelEncoder() # Global LabelEncoder
model_features = []
final_original_long_df = pd.DataFrame() # Explicitly make this one global too

warnings.filterwarnings('ignore')

# --- Styling and Global Configuration ---
MODERN_COLORS_DARK = {
    'primary': '#00D4FF', 'secondary': '#FF6B6B', 'accent': '#4ECDC4', 'dark': '#1A1A2E',
    'light': '#16213E', 'success': '#00F5A0', 'warning': '#FFD93D', 'text': '#FFFFFF'
}

MODERN_COLORS_LIGHT = {
    'primary': '#0066CC', 'secondary': '#E74C3C', 'accent': '#2ECC71', 'dark': '#F8F9FA',
    'light': '#FFFFFF', 'success': '#27AE60', 'warning': '#F39C12', 'text': '#2C3E50'
}

gender_palette_dark = {'Male': MODERN_COLORS_DARK['primary'], 'Female': MODERN_COLORS_DARK['secondary']}
gender_palette_light = {'Male': MODERN_COLORS_LIGHT['primary'], 'Female': MODERN_COLORS_LIGHT['secondary']}

# Global theme state - this is the single source of truth for the backend
current_theme = "dark"

# --- DATA LOADING AND PROCESSING (Faithfully replicating the notebook) ---
def load_and_process_data():
    """Performs the entire data pipeline."""
    global final_original_long_df 
    final_original_long_df = pd.DataFrame()
    print("--- Starting data loading and processing ---")
    try:
        # --- Part 1: Load Raw Data ---
        base_path = 'GenAIEyeTrackingCleanedDataset/'
        participant_file = os.path.join(base_path, 'ParticipantList.xlsx')

        if not os.path.exists(participant_file):
            print("Participant file not found. Falling back to sample data.")
            return create_sample_data()

        participant_df_global = pd.read_excel(
            participant_file, sheet_name='GENAI', header=2, usecols=['Gender', 'Participant ID']
        ).rename(columns={'Participant ID': 'Participant_ID'})
        participant_df_global.dropna(subset=['Gender', 'Participant_ID'], inplace=True)
        participant_df_global.drop_duplicates(subset='Participant_ID', keep='first', inplace=True)
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

        # --- Part 2: Clean and Merge All Raw Data ---
        all_cleaned_metrics_dfs = {}
        for q_name, config in questions_config.items():
            if not os.path.exists(config['file_path']): continue
            xls = pd.ExcelFile(config['file_path'])
            cleaned_metrics_dfs_qN = {}
            for sheet_name in selected_metric_sheets:
                if sheet_name in xls.sheet_names:
                    df = xls.parse(sheet_name)
                    if 'Participant' in df.columns:
                        df = df.rename(columns={'Participant': 'Participant_ID'})
                    if 'Participant_ID' in df.columns:
                        df['Participant_ID'] = df['Participant_ID'].apply(lambda x: f'P{int(str(x)[1:]):02d}' if isinstance(x, str) and str(x).startswith('P') and str(x)[1:].isdigit() else (f'P{int(x):02d}' if pd.notna(x) and isinstance(x, (int, float)) else x))
                        df = df.merge(participant_df_global, on='Participant_ID', how='left')
                        cleaned_metrics_dfs_qN[sheet_name] = df.dropna(subset=['Participant_ID', 'Gender'])
            all_cleaned_metrics_dfs[q_name] = cleaned_metrics_dfs_qN
        print("Data cleaning and gender merge complete.")

        # --- Part 3: Create the ORIGINAL Long DataFrame (from CLEANED data) ---
        all_original_long_dfs = {}
        for q_name, cleaned_sheets in all_cleaned_metrics_dfs.items():
            list_of_long_dfs_orig = []
            for sheet_name, df_cleaned in cleaned_sheets.items():
                aoi_cols_to_melt = [c for c in questions_config[q_name]['aoi_columns'] if c in df_cleaned.columns]
                if aoi_cols_to_melt:
                    df_long = df_cleaned.melt(id_vars=['Participant_ID', 'Gender'], value_vars=aoi_cols_to_melt, var_name='AOI', value_name=sheet_name)
                    list_of_long_dfs_orig.append(df_long)
            if list_of_long_dfs_orig:
                merged_df_orig = reduce(lambda left, right: pd.merge(left, right, on=['Participant_ID', 'Gender', 'AOI'], how='outer'), list_of_long_dfs_orig)
                def get_image_type(aoi_name_str):
                    if isinstance(aoi_name_str, str):
                        if re.search(r'\sA\d*$', aoi_name_str.strip()): return 'Real'
                        if re.search(r'\sB\d*$', aoi_name_str.strip()): return 'AI'
                    return 'Unknown'
                merged_df_orig['Image_Type'] = merged_df_orig['AOI'].apply(get_image_type)
                all_original_long_dfs[q_name] = merged_df_orig
        final_original_long_df = pd.concat([df.assign(Question=q) for q, df in all_original_long_dfs.items() if not df.empty], ignore_index=True)
        if final_original_long_df.empty:
            print("WARNING (load_and_process_data): final_original_long_df is EMPTY after Part 3 concatenation.")
        else:
            print("Original (unbalanced) combined dataframe (final_original_long_df) created successfully.")

        # --- Part 4: Create the BALANCED & RECONSTRUCTED Long DataFrame ---
        all_balanced_long_dfs = {}
        for q_name, cleaned_sheets in all_cleaned_metrics_dfs.items():
            df_master = cleaned_sheets.get(master_sheet_for_balancing)
            if df_master is None or df_master.empty: continue
            
            df_repr = df_master.groupby('Participant_ID').first().reset_index()
            X = df_repr[[c for c in questions_config[q_name]['aoi_columns'] if c in df_repr.columns]].fillna(0)
            y = df_repr['Gender']
            
            unified_resampled_master_set = df_repr
            if y.nunique() >= 2 and y.value_counts().min() >= 2:
                try:
                    smote = SMOTE(random_state=42, k_neighbors=min(5, y.value_counts().min() - 1))
                    X_res, y_res = smote.fit_resample(X, y)
                    df_resampled = pd.DataFrame(X_res, columns=X.columns)
                    df_resampled['Gender'] = y_res
                    df_resampled['Participant_ID'] = [f"Balanced_{q_name}_P{i:03d}" for i in range(len(df_resampled))]
                    unified_resampled_master_set = df_resampled
                except Exception as smote_error:
                    print(f"SMOTE failed for {q_name}: {smote_error}. Using original representative data.")

            reconstructed_wide_dfs = {}
            for sheet_name, df_orig in cleaned_sheets.items():
                if sheet_name == master_sheet_for_balancing:
                    reconstructed_wide_dfs[sheet_name] = unified_resampled_master_set
                else:
                    sheet_aoi_cols = [c for c in questions_config[q_name]['aoi_columns'] if c in df_orig.columns]
                    if not sheet_aoi_cols: continue
                    gender_means = df_orig.groupby('Gender')[sheet_aoi_cols].mean()
                    reconstructed_rows = []
                    for _, master_row in unified_resampled_master_set.iterrows():
                        new_row = {'Participant_ID': master_row['Participant_ID'], 'Gender': master_row['Gender']}
                        if master_row['Gender'] in gender_means.index:
                            new_row.update(gender_means.loc[master_row['Gender']].to_dict())
                        reconstructed_rows.append(new_row)
                    reconstructed_wide_dfs[sheet_name] = pd.DataFrame(reconstructed_rows)
            
            list_of_long_dfs = []
            for sheet_name, df_sheet in reconstructed_wide_dfs.items():
                aoi_cols_to_melt = [c for c in questions_config[q_name]['aoi_columns'] if c in df_sheet.columns]
                if aoi_cols_to_melt and 'Participant_ID' in df_sheet.columns:
                    df_long = df_sheet.melt(id_vars=['Participant_ID', 'Gender'], value_vars=aoi_cols_to_melt, var_name='AOI', value_name=sheet_name)
                    list_of_long_dfs.append(df_long)
            if list_of_long_dfs:
                all_balanced_long_dfs[q_name] = reduce(lambda l, r: pd.merge(l, r, on=['Participant_ID', 'Gender', 'AOI'], how='outer'), list_of_long_dfs)
                all_balanced_long_dfs[q_name]['Image_Type'] = all_balanced_long_dfs[q_name]['AOI'].apply(get_image_type)
        final_combined_long_df = pd.concat([df.assign(Question=q) for q, df in all_balanced_long_dfs.items() if not df.empty], ignore_index=True)
        print("Balanced & reconstructed dataframe created successfully.")


        # --- Part 5: Train Global Prediction Model (NEW SECTION) ---
        print("--- Training global prediction model ---")
        model_features = ["Tot Fixation dur", "Fixation count", "Time to first Fixation", "Tot Visit dur"]
        
        # === MODIFICATION FOR GLOBAL le ===
        global le # Declare le as global so it's accessible by other functions including Model 3
        le = LabelEncoder() # Initialize it

        if not final_original_long_df.empty: # More direct check
            participant_features = final_original_long_df.groupby(['Participant_ID', 'Gender'])[model_features].mean().reset_index().dropna()
        else:
            print("ERROR (load_and_process_data - Part 5): final_original_long_df is STILL empty. Cannot create participant_features for simple model.")
            participant_features = pd.DataFrame() 
        # Ensure participant_df_global is available from Part 1
        if 'participant_df_global' in globals() and not participant_df_global.empty and 'Gender' in participant_df_global.columns:
            # Fit 'le' on all unique gender values from the complete participant list
            try:
                unique_genders = participant_df_global['Gender'].astype(str).unique()
                if len(unique_genders) > 0:
                    le.fit(unique_genders) 
                    print(f"Global LabelEncoder 'le' fitted in load_and_process_data with classes: {le.classes_}")
                else:
                    print("WARNING (load_and_process_data): No unique gender values in participant_df_global to fit 'le'.")
            except Exception as e_fit_main:
                 print(f"ERROR (load_and_process_data): Failed to fit global 'le': {e_fit_main}. 'le' will be unfitted.")
        else:
            print("WARNING (load_and_process_data): participant_df_global not suitable for fitting 'le'. 'le' will be unfitted.")
        # === END MODIFICATION FOR GLOBAL le ===
        
        # Use the ORIGINAL data to create features, as this reflects real-world user behavior
        # Ensure final_original_long_df is correctly populated before this
        if 'final_original_long_df' in globals() and not final_original_long_df.empty:
            participant_features = final_original_long_df.groupby(['Participant_ID', 'Gender'])[model_features].mean().reset_index().dropna()
        else:
            print("ERROR (load_and_process_data): final_original_long_df is empty or not defined. Cannot create participant_features for simple model.")
            participant_features = pd.DataFrame() # Empty dataframe
    
        model = None # Initialize model to None
        
        if not participant_features.empty and 'Gender' in participant_features.columns:
            X = participant_features[model_features]
            y_original_gender_labels = participant_features['Gender']
            
            # Important: Check if there's enough data AND if le was fitted
            if y_original_gender_labels.nunique() >= 2 and len(participant_features) > 10 and hasattr(le, 'classes_') and le.classes_.size > 0:
                y_encoded = le.transform(y_original_gender_labels) # Use the globally prepared le
                
                smote_simple_model = SMOTE(random_state=42) # Instance for this model
                # Ensure k_neighbors is valid for SMOTE
                min_samples_smote = pd.Series(y_encoded).value_counts().min()
                k_neighbors = min(5, max(1, min_samples_smote - 1)) if min_samples_smote > 1 else 1
                
                if k_neighbors >= 1:
                    smote_simple_model.k_neighbors = k_neighbors
                    X_balanced, y_balanced = smote_simple_model.fit_resample(X, y_encoded)
                else:
                    print("Warning (load_and_process_data): Not enough samples for SMOTE in global predictor. Using original data.")
                    X_balanced, y_balanced = X.copy(), y_encoded.copy()
    
                # Train the model
                model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                model.fit(X_balanced, y_balanced)
                print("Global prediction model (simple) trained successfully.")
            else:
                print("Insufficient data or 'le' not properly fitted to train a global prediction model (simple). Predictor will be disabled.")
                # model remains None
                # model_features are already defined. 'le' is global.
        else:
            print("Participant features for simple model are empty. Simple predictor disabled.")
            # model remains None
    
        print(f"--- Data processing finished. ---")
        # 'le' is global now. The 'le' in the return statement refers to this global 'le'.
        return all_balanced_long_dfs, final_combined_long_df, all_original_long_dfs, all_cleaned_metrics_dfs, selected_metric_sheets, participant_df_global, questions_config, model, le, model_features

    except Exception as e:
        print(f"FATAL ERROR during data processing: {e}\n{traceback.format_exc()}")
        # When create_sample_data is called, it should also correctly set up global final_original_long_df
        return create_sample_data() 

def create_sample_data():
    """
    Creates comprehensive sample dataframes and sets up global variables 
    if the main data loading pipeline fails.
    This function is a fallback and aims to provide a functional, albeit simplified,
    dashboard experience.
    """
    print("--- Creating Sample Data as Fallback ---")

    # Declare all global variables that this function will define or modify
    global all_balanced_long_dfs, final_combined_long_df, all_original_long_dfs
    global all_cleaned_metrics_dfs, selected_metric_sheets, participant_df_global
    global questions_config, model, le, model_features, final_original_long_df

    # Initialize global variables to their default/empty states
    all_balanced_long_dfs = {}
    final_combined_long_df = pd.DataFrame()
    all_original_long_dfs = {}
    all_cleaned_metrics_dfs = {}
    selected_metric_sheets = ["Tot Fixation dur", "Fixation count", "Time to first Fixation", "Tot Visit dur"]
    participant_df_global = pd.DataFrame()
    # questions_config needs to be a sample version
    questions_config = { # Renamed from sample_questions_config to match global name
        'Q1_Sample': {'aoi_columns': ['Q1 AOI Real 1', 'Q1 AOI Real 2', 'Q1 AOI AI 1', 'Q1 AOI AI 2']},
        'Q2_Sample': {'aoi_columns': ['Q2 AOI Real 1', 'Q2 AOI Real 2', 'Q2 AOI AI 1', 'Q2 AOI AI 2']}
    }
    model = None # Simple predictor model
    le = LabelEncoder() # Global LabelEncoder
    model_features = ["Tot Fixation dur", "Fixation count", "Time to first Fixation", "Tot Visit dur"] # Features for simple model
    final_original_long_df = pd.DataFrame() # This is crucial

    # --- Generate sample data ---
    sample_questions = list(questions_config.keys()) # ['Q1_Sample', 'Q2_Sample']
    num_participants_sample = 20
    genders_sample = ['Male', 'Female']
    
    # Placeholder for all_original_long_dfs data
    temp_all_original_dfs_list = []

    for q_name in sample_questions:
        q_data_orig = []
        q_data_balanced = [] # For all_balanced_long_dfs (can be same as original for simplicity here)
        
        # For all_cleaned_metrics_dfs (wide format)
        # We'll focus on 'Tot Fixation dur' for the wide format sample
        q_cleaned_wide_data_rows = []

        for p_idx in range(num_participants_sample):
            participant_id = f'Sample_P{p_idx:02d}'
            gender = genders_sample[p_idx % len(genders_sample)]
            
            # Data for wide format (all_cleaned_metrics_dfs)
            row_for_wide_df = {'Participant_ID': participant_id, 'Gender': gender}

            for aoi_col_name in questions_config[q_name]['aoi_columns']:
                # Determine Image_Type from AOI name convention (simplified)
                image_type = 'Real' if 'Real' in aoi_col_name else 'AI' if 'AI' in aoi_col_name else 'Unknown'
                
                # Generate random metric values
                fix_dur = np.random.gamma(2, 0.8 if gender == 'Female' else 0.9) + (0.1 if image_type == 'AI' else 0)
                fix_count = np.random.gamma(2.5, 1.0 if gender == 'Female' else 1.1) + (1 if image_type == 'AI' else 0)
                time_first_fix = np.random.gamma(1.5, 0.5)
                visit_dur = np.random.gamma(3, 0.7 if gender == 'Female' else 0.8) + (0.2 if image_type == 'AI' else 0)

                # Append to long format list for this question
                record = {
                    'Participant_ID': participant_id,
                    'Gender': gender,
                    'AOI': aoi_col_name, # Use the actual column name as AOI for simplicity
                    'Image_Type': image_type,
                    'Question': q_name,
                    "Tot Fixation dur": fix_dur,
                    "Fixation count": fix_count,
                    "Time to first Fixation": time_first_fix,
                    "Tot Visit dur": visit_dur
                }
                q_data_orig.append(record)
                q_data_balanced.append(record) # For sample, balanced can be same as original

                # Add to row for wide df (using Tot Fixation dur as example)
                row_for_wide_df[aoi_col_name] = fix_dur
            
            q_cleaned_wide_data_rows.append(row_for_wide_df)

        # Create DataFrames for the current question
        if q_data_orig:
            df_orig_q = pd.DataFrame(q_data_orig)
            all_original_long_dfs[q_name] = df_orig_q
            temp_all_original_dfs_list.append(df_orig_q) # Collect for final_original_long_df

            df_balanced_q = pd.DataFrame(q_data_balanced)
            all_balanced_long_dfs[q_name] = df_balanced_q
        
        if q_cleaned_wide_data_rows:
            # For all_cleaned_metrics_dfs, it expects a dict of metric_name: dataframe
            all_cleaned_metrics_dfs[q_name] = {
                'Tot Fixation dur': pd.DataFrame(q_cleaned_wide_data_rows)
            }
            # If you need other metrics in wide format for sample, add them similarly

    # --- Populate the main global DataFrames ---
    if temp_all_original_dfs_list:
        final_original_long_df = pd.concat(temp_all_original_dfs_list, ignore_index=True)
    else:
        # Fallback if somehow no original data was generated (should not happen with this logic)
        final_original_long_df = pd.DataFrame(columns=['Participant_ID', 'Gender', 'AOI', 'Image_Type', 'Question'] + selected_metric_sheets)
    
    # For sample data, final_combined_long_df can be the same as final_original_long_df
    # or derived from all_balanced_long_dfs if they were different.
    # Here, using all_balanced_long_dfs to construct it.
    if all_balanced_long_dfs:
         final_combined_long_df = pd.concat(
             [df.assign(Question=q) for q, df in all_balanced_long_dfs.items() if not df.empty], 
             ignore_index=True
         )
    else:
        final_combined_long_df = pd.DataFrame(columns=['Participant_ID', 'Gender', 'AOI', 'Image_Type', 'Question'] + selected_metric_sheets)


    # Derive participant_df_global from the sample final_original_long_df
    if not final_original_long_df.empty:
        participant_df_global = final_original_long_df[['Participant_ID', 'Gender']].drop_duplicates().reset_index(drop=True)
    else:
        participant_df_global = pd.DataFrame(columns=['Participant_ID', 'Gender'])

    print(f"Sample Data - final_original_long_df shape: {final_original_long_df.shape}")
    if not final_original_long_df.empty: print(f"Sample Data - final_original_long_df head:\n{final_original_long_df.head().to_string()}")
    print(f"Sample Data - participant_df_global shape: {participant_df_global.shape}")
    if not participant_df_global.empty: print(f"Sample Data - participant_df_global head:\n{participant_df_global.head().to_string()}")


    # --- Fit the global LabelEncoder (le) ---
    if not participant_df_global.empty and 'Gender' in participant_df_global.columns:
        try:
            unique_genders_sample = participant_df_global['Gender'].astype(str).unique()
            if len(unique_genders_sample) > 0 and len(unique_genders_sample) >=2 : # Need at least 2 for meaningful encoding
                le.fit(unique_genders_sample)
                print(f"Sample Data: Global LabelEncoder 'le' fitted with classes: {le.classes_}")
            elif len(unique_genders_sample) < 2:
                print(f"WARNING (Sample Data): Not enough unique gender values ({len(unique_genders_sample)}) in sample participant_df_global to fit 'le' meaningfully.")
                # le remains unfitted or fitted with <2 classes, which can cause issues downstream
            else: # len is 0
                 print("WARNING (Sample Data): No unique gender values in sample participant_df_global to fit 'le'.")
        except Exception as e_fit_sample:
            print(f"ERROR (Sample Data): Could not fit LabelEncoder for sample data: {e_fit_sample}")
    else:
        print("WARNING (Sample Data): Sample participant_df_global is empty or missing 'Gender'. 'le' will be unfitted.")

    # Model for the simple predictor is None, model_features are set.
    # 'le' is the global instance, now hopefully fitted.

    print("--- Sample Data Creation Finished ---")
    
    # Return all the global-scope variables that load_and_process_data is expected to return
    return (
        all_balanced_long_dfs, 
        final_combined_long_df, 
        all_original_long_dfs, 
        all_cleaned_metrics_dfs, 
        selected_metric_sheets, 
        participant_df_global, 
        questions_config, # This is the sample questions_config
        model, # Which is None for sample data
        le,    # The global le instance
        model_features
    )


# --- THEME-AWARE PLOTTING FUNCTIONS ---
def get_theme_colors():
    """Get colors based on the global current_theme variable."""
    return (MODERN_COLORS_DARK, gender_palette_dark) if current_theme == "dark" else (MODERN_COLORS_LIGHT, gender_palette_light)

def get_plot_layout():
    """Get plot layout based on the global current_theme variable."""
    if current_theme == "dark":
        return {'plot_bgcolor': '#000000', 'paper_bgcolor': '#000000', 'font_color': 'white'}
    else:
        return {'plot_bgcolor': '#FFFFFF', 'paper_bgcolor': '#FFFFFF', 'font_color': '#2C3E50'}

def _create_4_panel_dashboard(data, selected_metric, plot_title_suffix):
    colors, gender_palette = get_theme_colors()
    layout = get_plot_layout()
    if data is None or data.empty or selected_metric not in data.columns:
        return go.Figure().add_annotation(text="No data for this view", showarrow=False).update_layout(**layout)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(f'{selected_metric} by Image Type & Gender', f'{selected_metric} Violin Plot', f'"{selected_metric}" Distribution by Gender', 'Summary Statistics'),
        specs=[[{"type": "box"}, {"type": "violin"}], [{"type": "histogram"}, {"type": "table"}]]
    )
    
    for gender in ['Male', 'Female']:
        subset = data[data['Gender'] == gender]
        fig.add_trace(go.Box(y=subset[selected_metric], x=subset['Image_Type'], name=gender, marker_color=gender_palette.get(gender), legendgroup=gender, showlegend=True, boxpoints='outliers'), row=1, col=1)
    fig.update_layout(boxmode='group', xaxis1_title='Image Type')
        
    for gender in ['Male', 'Female']:
        for img_type in data['Image_Type'].unique():
            subset = data[(data['Image_Type'] == img_type) & (data['Gender'] == gender)]
            if not subset.empty:
                fig.add_trace(go.Violin(y=subset[selected_metric], x0=str(img_type), name=gender, side='negative' if gender == 'Male' else 'positive', marker_color=gender_palette.get(gender), points=False, legendgroup=gender, showlegend=False, meanline_visible=True), row=1, col=2)
    fig.update_layout(violinmode='overlay', xaxis2_title='Image Type')
        
    for gender in ['Male', 'Female']:
        subset = data[data['Gender'] == gender]
        fig.add_trace(go.Histogram(x=subset[selected_metric], name=gender, marker_color=gender_palette.get(gender), legendgroup=gender, showlegend=False, opacity=0.7), row=2, col=1)
    fig.update_layout(barmode='overlay')

    summary_stats = data.groupby(['Image_Type', 'Gender'])[selected_metric].agg(['count', 'mean', 'std', 'min', 'max']).round(2).reset_index()
    header_color = colors['primary']
    cell_color = 'rgba(40,40,60,0.8)' if current_theme == "dark" else 'rgba(240, 240, 240, 0.9)'
    font_color = colors['text']
    header_font_color = 'white'

    fig.add_trace(go.Table(
        header=dict(values=[f'<b>{c.upper()}</b>' for c in summary_stats.columns], fill_color=header_color, font_color=header_font_color, align='center', font=dict(size=12)),
        cells=dict(values=[summary_stats[c] for c in summary_stats.columns], fill_color=cell_color, font_color=font_color, align='center', font=dict(size=11))
    ), row=2, col=2)

    fig.update_layout(height=850, **layout, title_text=f'"{selected_metric}" - Analysis Dashboard {plot_title_suffix}', title_x=0.5, title_font_size=20, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(title_text=selected_metric, row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    return fig

def _create_correlation_heatmap(data, numeric_metrics, plot_title_suffix):
    layout = get_plot_layout()
    genders_present = sorted([g for g in data['Gender'].unique() if pd.notna(g)])
    if not genders_present or len(numeric_metrics) < 2:
        return go.Figure().add_annotation(text="Not enough data for heatmap", showarrow=False).update_layout(**layout)

    fig = make_subplots(rows=1, cols=len(genders_present), subplot_titles=[f"Metric Correlation ({gender})" for gender in genders_present])
    for i, gender in enumerate(genders_present):
        col = i + 1
        subset_corr = data[data['Gender'] == gender][numeric_metrics]
        if not subset_corr.empty:
            corr_matrix = subset_corr.corr()
            fig.add_trace(go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, colorscale='RdBu_r', zmin=-1, zmax=1, text=corr_matrix.values, texttemplate="%{text:.2f}", textfont={"size":9}, hovertemplate='Metric 1: %{y}<br>Metric 2: %{x}<br>Correlation: %{z:.2f}<extra></extra>'), row=1, col=col)
    
    fig.update_layout(height=500, **layout, title_text=f"Correlation Heatmaps by Gender {plot_title_suffix}", title_x=0.5, title_font_size=20)
    return fig

def create_modern_bar_plot(data, metric, agg_func, plot_title_suffix, question=None):
    layout = get_plot_layout()
    _, gender_palette = get_theme_colors()
    
    if data is None or data.empty or metric not in data.columns:
        return go.Figure().add_annotation(text="No data for bar plot",
            showarrow=False).update_layout(**layout)
    
    aoi_gender_summary = data.groupby(['Gender', 'AOI', 'Image_Type'],
        as_index=False).agg({metric: agg_func})
    
    if aoi_gender_summary.empty:
        return go.Figure().add_annotation(text="No data for this filter",
            showarrow=False).update_layout(**layout)
    
    aoi_gender_summary['AOI_Labeled'] = aoi_gender_summary.apply(lambda row:
        f"{row['AOI']} ({row['Image_Type']})", axis=1)
    aoi_gender_summary = aoi_gender_summary.sort_values(by=['Image_Type', 'AOI'],
        ascending=[False, True])
    
    # Create subplot with image space - CHANGED: Better proportions and specs
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],  # Give more space to image
        specs=[[{"type": "bar"}, {"type": "xy"}]],  # Changed from "image" to "xy"
        subplot_titles=[f'{metric} ({agg_func.capitalize()}) per AOI {plot_title_suffix}', 'Question Image']
    )
    
    # Add bar chart
    for gender in ['Male', 'Female']:
        gender_data = aoi_gender_summary[aoi_gender_summary['Gender'] == gender]
        fig.add_trace(
            go.Bar(
                x=gender_data['AOI_Labeled'],
                y=gender_data[metric],
                name=gender,
                marker_color=gender_palette.get(gender),
                legendgroup=gender
            ),
            row=1, col=1
        )
    
    # Add question image if available - CHANGED: Better image handling
    if question and question != 'All Combined':
        image_url = f"https://huggingface.co/spaces/RextonRZ/GenAI-ETGenderDashboard/resolve/main/ImagePair/{question}.png"
        
        # Add the image using add_layout_image with better positioning
        fig.add_layout_image(
            dict(
                source=image_url,
                xref="x2", yref="y2",  # CHANGED: Use domain references
                x=0, y=0.95,
                sizex=1, sizey=1,
                sizing="contain",
                opacity=1,
                layer="below"  # CHANGED: Put image above background
            )
        )
    
    
    # Configure layout - CHANGED: Better height and spacing
    fig.update_layout(
        **layout,
        height=700,
        barmode='group',
        title_x=0.5,
        showlegend=True,
        margin=dict(l=50, r=50, t=100, b=100),
        dragmode='zoom',
    

        xaxis2=dict(
            range=[0, 1],
            constrain='domain',
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor='y2'
        ),
        yaxis2=dict(
            range=[0, 1],
            constrain='domain',
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleratio=1
        )
    )
    
    fig.update_xaxes(tickangle=-45, title_text="Area of Interest (Image Type)", row=1, col=1)
    fig.update_yaxes(title_text=metric, row=1, col=1)
    
    return fig
    
def create_combined_bar_plot(data, metric, agg_func, plot_title_suffix):
    layout = get_plot_layout()
    _, gender_palette = get_theme_colors()
    if data is None or data.empty or metric not in data.columns:
        return go.Figure().add_annotation(text="No data for bar plot", showarrow=False).update_layout(**layout)
        
    summary = data.groupby(['Image_Type', 'Gender'], as_index=False).agg({metric: agg_func})
    fig = px.bar(summary, x='Image_Type', y=metric, color='Gender', color_discrete_map=gender_palette, title=f'{metric} ({agg_func.capitalize()}) by Image Type {plot_title_suffix}', height=500, barmode='group')
    fig.update_layout(**layout, title_x=0.5, xaxis_title='Image Type')
    return fig

def create_modern_scatter_plot(data, dur_col, count_col, plot_title_suffix):
    layout = get_plot_layout()
    _, gender_palette = get_theme_colors()
    if data is None or data.empty or dur_col not in data.columns or count_col not in data.columns:
        return go.Figure().add_annotation(text="No data for scatter plot", showarrow=False).update_layout(**layout)

    valid_data = data.dropna(subset=[dur_col, count_col])
    if valid_data.empty: return go.Figure().add_annotation(text="No valid data points", showarrow=False).update_layout(**layout)
    
    fig = px.scatter(valid_data, x=dur_col, y=count_col, color='Gender', symbol='Image_Type', title=f'Interactive Scatter: {count_col} vs {dur_col} {plot_title_suffix}', hover_data=['Participant_ID', 'AOI'], color_discrete_map=gender_palette, height=600)
    fig.update_layout(**layout, title_x=0.5, legend=dict(bgcolor='rgba(0,0,0,0)'))
    return fig

# --- Load Data on Startup ---
print("Loading and processing data. This may take a moment...")
all_balanced_long_dfs, final_combined_long_df, all_original_long_dfs, all_cleaned_metrics_dfs, selected_metric_sheets, participant_df_global, questions_config, model, le, model_features = load_and_process_data()

# --- BACKEND LOGIC FUNCTIONS ---
def create_normality_test_table():
    """Creates a Plotly Table showing the normality test results."""
    global all_original_long_dfs
    layout = get_plot_layout()
    colors, _ = get_theme_colors()

    df = pd.concat([df_q.assign(Question=q) for q, df_q in all_original_long_dfs.items()], ignore_index=True) if all_original_long_dfs else pd.DataFrame()
    if df is None or df.empty:
        return go.Figure().add_annotation(text="Normality test data not available.", showarrow=False).update_layout(**layout)

    normality_info = []
    available_metrics = [
        col for col in ["Tot Fixation dur", "Fixation count", "Time to first Fixation", "Tot Visit dur"]
        if col in df.columns
    ]

    for q_name in sorted(df['Question'].unique()):
        for metric in available_metrics:
            q_metric_df = df[df['Question'] == q_name]
            male_data = q_metric_df[q_metric_df['Gender'] == 'Male'][metric].dropna()
            female_data = q_metric_df[q_metric_df['Gender'] == 'Female'][metric].dropna()

            is_normal_m, is_normal_f, test_used = None, None, 'Insufficient data'
            if len(male_data) >= 3 and len(female_data) >= 3:
                p_male = shapiro(male_data).pvalue if len(set(male_data)) > 1 else 1.0
                p_female = shapiro(female_data).pvalue if len(set(female_data)) > 1 else 1.0
                is_normal_m = p_male > 0.05
                is_normal_f = p_female > 0.05
                test_used = 't-test' if is_normal_m and is_normal_f else 'Mann-Whitney U'
            
            normality_info.append({
                'Question': q_name, 'Metric': metric,
                'Male Normal?': 'Yes' if is_normal_m else 'No' if is_normal_m is not None else 'N/A',
                'Female Normal?': 'Yes' if is_normal_f else 'No' if is_normal_f is not None else 'N/A',
                'Test Used': test_used
            })

    if not normality_info:
        return go.Figure().add_annotation(text="No normality data to display.", showarrow=False).update_layout(**layout)

    normality_df = pd.DataFrame(normality_info)
    
    header_color = colors['primary']
    cell_color = 'rgba(40,40,60,0.8)' if current_theme == "dark" else 'rgba(240, 240, 240, 0.9)'

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f'<b>{c.upper()}</b>' for c in normality_df.columns],
            fill_color=header_color,
            font=dict(color='white', size=12),
            align='left'
        ),
        cells=dict(
            values=[normality_df[c] for c in normality_df.columns],
            fill_color=cell_color,
            font=dict(color=colors['text'], size=11),
            align='left'
        ))
    ])
    fig.update_layout(height=600, **layout, title_text="Normality Check and Test Selection", title_x=0.5)
    return fig

def create_statistical_test_plot():
    """Creates a Plotly Table showing the p-values from statistical tests."""
    global all_original_long_dfs
    layout = get_plot_layout()
    colors, _ = get_theme_colors()

    df = pd.concat([df_q.assign(Question=q) for q, df_q in all_original_long_dfs.items()], ignore_index=True) if all_original_long_dfs else pd.DataFrame()
    if df is None or df.empty:
        return go.Figure().add_annotation(text="Statistical test data not available.", showarrow=False).update_layout(**layout)

    pval_rows_data = []
    available_metrics = [
        col for col in ["Tot Fixation dur", "Fixation count", "Time to first Fixation", "Tot Visit dur"]
        if col in df.columns
    ]

    for q_name in sorted(df['Question'].unique()):
        p_values_for_row = {'Question': q_name}
        for metric in available_metrics:
            q_metric_df = df[df['Question'] == q_name]
            male_data = q_metric_df[q_metric_df['Gender'] == 'Male'][metric].dropna()
            female_data = q_metric_df[q_metric_df['Gender'] == 'Female'][metric].dropna()

            test_used = 'Insufficient data'
            if len(male_data) >= 3 and len(female_data) >= 3:
                p_male = shapiro(male_data).pvalue if len(set(male_data)) > 1 else 1.0
                p_female = shapiro(female_data).pvalue if len(set(female_data)) > 1 else 1.0
                test_used = 't-test' if p_male > 0.05 and p_female > 0.05 else 'Mann-Whitney U'

            p_val = np.nan
            if test_used == 't-test':
                _, p_val = ttest_ind(male_data, female_data, equal_var=True)
            elif test_used == 'Mann-Whitney U':
                _, p_val = mannwhitneyu(male_data, female_data, alternative='two-sided')
            
            p_values_for_row[metric] = p_val
        
        pval_rows_data.append(p_values_for_row)

    if not pval_rows_data:
        return go.Figure().add_annotation(text="No p-value data to display.", showarrow=False).update_layout(**layout)

    pval_df = pd.DataFrame(pval_rows_data).set_index('Question')
    
    header_values = ['<b>Question</b>'] + [f'<b>{col}</b>' for col in pval_df.columns]
    cell_values = [pval_df.index.tolist()]
    cell_colors = []
    
    base_cell_color = 'rgba(40,40,60,0.8)' if current_theme == "dark" else 'rgba(240, 240, 240, 0.9)'
    significant_color = 'rgba(0, 245, 160, 0.3)'
    
    color_col_base = [base_cell_color] * len(pval_df)
    cell_colors.append(color_col_base)

    for col in pval_df.columns:
        formatted_values = [f"{v:.4f}" if pd.notna(v) else "N/A" for v in pval_df[col]]
        cell_values.append(formatted_values)
        color_col = [significant_color if pd.notna(v) and v < 0.05 else base_cell_color for v in pval_df[col]]
        cell_colors.append(color_col)
        
    fig = go.Figure(data=[go.Table(
        header=dict(values=header_values, fill_color=colors['primary'], font=dict(color='white', size=12), align='center'),
        cells=dict(values=cell_values, fill_color=cell_colors, font=dict(color=colors['text'], size=11), align='center')
    )])
    
    fig.update_layout(height=400, **layout, title_text="P-Value Summary from Hypothesis Tests (Gender Differences)", title_x=0.5)
    return fig

def create_participant_pie_chart(question):
    """Creates a pie chart of the gender distribution for the selected question set."""
    global all_original_long_dfs, participant_df_global
    layout = get_plot_layout()
    _, gender_palette = get_theme_colors()

    if question == 'All Combined':
        participants_to_show = participant_df_global
        title = "Overall Gender Distribution"
    else:
        df_q = all_original_long_dfs.get(question)
        if df_q is not None and not df_q.empty:
            participant_ids_in_q = df_q['Participant_ID'].unique()
            participants_to_show = participant_df_global[participant_df_global['Participant_ID'].isin(participant_ids_in_q)]
            title = f"Gender Distribution for {question}"
        else:
            participants_to_show = pd.DataFrame(columns=['Gender'])
            title = "No Data for Selection"

    if participants_to_show.empty:
        return go.Figure().add_annotation(text="No participant data for this selection.").update_layout(**layout)

    gender_counts = participants_to_show['Gender'].value_counts()
    fig = px.pie(values=gender_counts.values, names=gender_counts.index, title=title, color=gender_counts.index, color_discrete_map=gender_palette)
    fig.update_layout(**layout, title_x=0.5, showlegend=True, legend_title_text='Gender')
    return fig

def get_summary_stats(question):
    """Gets the count of participants for the summary boxes."""
    global all_original_long_dfs, participant_df_global
    if question == 'All Combined':
        participants_to_show = participant_df_global
    else:
        df_q = all_original_long_dfs.get(question)
        if df_q is not None and not df_q.empty:
            participant_ids_in_q = df_q['Participant_ID'].unique()
            participants_to_show = participant_df_global[participant_df_global['Participant_ID'].isin(participant_ids_in_q)]
        else:
            return 0, 0, 0

    total_p = participants_to_show['Participant_ID'].nunique()
    male_c = participants_to_show[participants_to_show['Gender'] == 'Male']['Participant_ID'].nunique()
    female_c = participants_to_show[participants_to_show['Gender'] == 'Female']['Participant_ID'].nunique()
    return total_p, male_c, female_c

def run_prediction(fix_dur, fix_count, ttf, visit_dur):
    """Runs the prediction model on user inputs from the sliders."""
    global model, le, model_features # Ensure we're using the global ones

    # Check if 'le' is fitted properly (has classes)
    le_is_ready = hasattr(le, 'classes_') and le.classes_.size > 0

    if model is None or not le_is_ready or not model_features:
        # If 'le' is not ready, we can't even show class labels.
        # Gradio Label can also take a simple string.
        # Or, provide dummy probabilities if le.classes_ exists.
        if le_is_ready:
            # Provide dummy low probabilities for known classes
            return {label: 0.0 for label in le.classes_} 
        else:
            # Critical error, 'le' itself is not ready
            return "Predictor Error: Model or LabelEncoder not available."
            # Alternatively, if you want to show the 'Error' key in the label:
            # return {"Error": "Model not available / LabelEncoder not ready"} 
            # But the value for "Error" should be a float if you want it displayed like a confidence.
            # Best to return a simple string for the label value in this case if num_top_classes is used.
            # If num_top_classes is NOT used, then `gr.Label` can just display the string "Error: Model not available...".
            # Since num_top_classes=2 is used, it expects a dict of label:confidence.
            # So, returning a string will make gr.Label show that string directly instead of trying to parse confidences.

    try:
        # Create a DataFrame in the same order the model was trained on
        input_data = pd.DataFrame([[fix_dur, fix_count, ttf, visit_dur]], columns=model_features)
        
        # Predict probabilities
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Format for Gradio's Label component
        return {le.classes_[i]: proba for i, proba in enumerate(prediction_proba)}
    except Exception as e:
        print(f"Error during prediction: {e}")
        if le_is_ready:
            return {label: 0.0 for label in le.classes_} # Dummy output on error
        else:
            return "Predictor Error: Exception during prediction."

def update_all_outputs(question, metric, image_types_to_show):
    """A single master function to update all UI components."""
    global final_combined_long_df, all_balanced_long_dfs, all_original_long_dfs, selected_metric_sheets
    
    try:
        df_balanced_reconstructed = final_combined_long_df if question == 'All Combined' else all_balanced_long_dfs.get(question, pd.DataFrame())
        df_original_unbalanced = pd.concat(list(all_original_long_dfs.values())) if question == 'All Combined' else all_original_long_dfs.get(question, pd.DataFrame())

        layout = get_plot_layout()
        if df_balanced_reconstructed.empty or df_original_unbalanced.empty:
            empty_fig = go.Figure().add_annotation(text="No data for this view", showarrow=False).update_layout(**layout)
            return 0, 0, 0, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

        plot_title_suffix = f"({question})"
        bar_chart_data = df_balanced_reconstructed[df_balanced_reconstructed['Image_Type'].isin(image_types_to_show or [])]
        bar_chart = create_modern_bar_plot(bar_chart_data, metric, 'mean', plot_title_suffix, question) if question != 'All Combined' else create_combined_bar_plot(bar_chart_data, metric, 'mean', plot_title_suffix)
        scatter_chart = create_modern_scatter_plot(df_original_unbalanced, 'Tot Fixation dur', 'Fixation count', plot_title_suffix)
        dashboard_chart = _create_4_panel_dashboard(df_original_unbalanced, metric, plot_title_suffix)
        heatmap_chart = _create_correlation_heatmap(df_original_unbalanced, selected_metric_sheets, plot_title_suffix)
        total_p, male_c, female_c = get_summary_stats(question)
        pie_chart = create_participant_pie_chart(question)
        
        return total_p, male_c, female_c, pie_chart, dashboard_chart, heatmap_chart, bar_chart, scatter_chart

    except Exception as e:
        print(f"Error in update_all_outputs: {e}\n{traceback.format_exc()}")
        layout = get_plot_layout()
        empty_fig = go.Figure().add_annotation(text=f"Plotting Error: {str(e)}", showarrow=False).update_layout(**layout)
        return 0, 0, 0, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

def handle_theme_toggle(theme_state, question, metric, image_types_to_show):
    """Updates theme, regenerates all outputs, and returns new state to the UI."""
    global current_theme
    new_theme = "light" if theme_state == "dark" else "dark"
    current_theme = new_theme
    
    dynamic_outputs = list(update_all_outputs(question, metric, image_types_to_show))
    # Regenerate static plots for the new theme
    normality_fig = create_normality_test_table()
    stats_fig = create_statistical_test_plot()

    new_button_text = "☀️" if new_theme == "light" else "🌙"
    
    return [new_theme] + dynamic_outputs + [normality_fig, stats_fig, new_button_text]

# --- ML Model 1: Backend Function ---
def run_predictive_power_analysis():
    """
    Trains a model to predict gender from aggregated metrics and returns the results.
    """
    global all_original_long_dfs
    df = pd.concat([df_q.assign(Question=q) for q, df_q in all_original_long_dfs.items()], ignore_index=True) if all_original_long_dfs else pd.DataFrame()
    if df is None or df.empty:
        return None, "Error: Data not loaded. Cannot run analysis."
    
    # 1. Prepare data
    participant_summary_df = df.groupby(['Participant_ID', 'Gender']).agg({
        "Tot Fixation dur": 'mean', "Fixation count": 'mean',
        "Time to first Fixation": 'mean', "Tot Visit dur": 'mean'
    }).reset_index().dropna()

    features = ["Tot Fixation dur", "Fixation count", "Time to first Fixation", "Tot Visit dur"]
    X = participant_summary_df[features]
    y = participant_summary_df['Gender']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )

    # 3. Baseline Model
    baseline_model = DummyClassifier(strategy="most_frequent")
    baseline_model.fit(X_train, y_train)
    baseline_accuracy = baseline_model.score(X_test, y_test)

    # 4. Actual Model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_accuracy = accuracy_score(y_test, y_pred)

    # 5. Create Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title(f'Confusion Matrix\nModel Accuracy: {model_accuracy:.2%}', fontsize=14)
    ax.set_ylabel('Actual Gender')
    ax.set_xlabel('Predicted Gender')
    plt.tight_layout()

    # 6. Create Conclusion HTML
    conclusion_html = f"""
    <div style='padding: 10px; border-radius: 5px; border: 1px solid #ccc;'>
    <h3>Grand, Unified Conclusion</h3>
    <p>Our investigation yielded a nuanced and insightful conclusion. Initial statistical hypothesis testing (Mann-Whitney U) successfully identified isolated, statistically significant differences in gaze behavior between genders for specific questions, leading us to reject our null hypothesis.</p>
    <p>However, the key question remained whether these isolated differences constituted a holistic, predictable pattern of behavior. To test this, a RandomForest machine learning model was trained to predict gender based on aggregated gaze metrics.</p>
    <p>The model achieved an accuracy of only <b>{model_accuracy:.2%}</b>, performing worse than a non-learning baseline of <b>{baseline_accuracy:.2%}</b>. This demonstrates that while statistically significant differences exist, they are not large or consistent enough across all contexts to be practically predictive.</p>
    <p><b>In conclusion, our mixed-method approach reveals that while we can statistically detect subtle gender-based differences in gaze, these differences do not translate into a robust, overarching behavioral 'signature'. This highlights a critical distinction between statistical significance and practical, predictive significance in human behavior research.</b></p>
    </div>
    """
    return fig, conclusion_html

# --- ML Model 2: Backend Function ---
def run_aoi_importance_analysis():
    """
    Performs cross-validated AOI importance analysis and returns the results.
    """
    global all_cleaned_metrics_dfs, questions_config
    if not all_cleaned_metrics_dfs or not questions_config:
        return None, "Error: Data not loaded. Cannot run analysis."
        
    overall_importance_results = []
    question_accuracies = {}
    
    # Setup a single figure for all importance plots
    q_keys = sorted(questions_config.keys())
    fig, axes = plt.subplots(len(q_keys), 1, figsize=(10, len(q_keys) * 5))
    if len(q_keys) == 1: axes = [axes] # Ensure axes is always iterable

    for i, q_name in enumerate(q_keys):
        ax = axes[i]
        df_wide = all_cleaned_metrics_dfs.get(q_name, {}).get('Tot Fixation dur')
        if df_wide is None or df_wide.empty:
            ax.text(0.5, 0.5, f"No data for {q_name}", ha='center', va='center')
            ax.set_title(f'AOI Importance for {q_name}')
            continue

        aoi_cols = [col for col in questions_config[q_name]['aoi_columns'] if col in df_wide.columns]
        if not aoi_cols:
            ax.text(0.5, 0.5, f"No AOI columns for {q_name}", ha='center', va='center')
            ax.set_title(f'AOI Importance for {q_name}')
            continue

        q_df = df_wide[['Gender'] + aoi_cols].copy().dropna(subset=['Gender'])
        q_df[aoi_cols] = q_df[aoi_cols].fillna(0)
        X, y = q_df[aoi_cols], q_df['Gender']
        
        if y.nunique() < 2: continue
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        importances_list, accuracies = [], []
        smote = SMOTE(random_state=42)

        for train_index, val_index in skf.split(X, y_encoded):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y_encoded[train_index], y_encoded[val_index]
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train_resampled, y_train_resampled)
            y_pred = model.predict(X_val)
            accuracies.append(accuracy_score(y_val, y_pred))
            importances_list.append(model.feature_importances_)

        avg_accuracy = np.mean(accuracies)
        avg_importances = np.mean(importances_list, axis=0)
        question_accuracies[q_name] = f"{avg_accuracy:.1%}"
        
        importance_df = pd.DataFrame({'AOI': aoi_cols, 'Importance': avg_importances}).sort_values(by='Importance', ascending=False)
        importance_df['Question'] = q_name
        overall_importance_results.append(importance_df)

        sns.barplot(x='Importance', y='AOI', data=importance_df, palette='viridis', ax=ax)
        ax.set_title(f'AOI Importance for {q_name}\n(CV Accuracy: {avg_accuracy:.2%})', fontsize=14)
        ax.set_xlabel('Average Importance Score')
    
    plt.tight_layout(h_pad=4.0)
    
    # Create final conclusion HTML
    if not overall_importance_results:
        return fig, "<p>Could not generate summary table.</p>"
        
    final_summary_df = pd.concat(overall_importance_results)
    top_aois = final_summary_df.groupby('Question').apply(lambda x: x.nlargest(3, 'Importance')).reset_index(drop=True)
    top_aois_pivot = top_aois.pivot_table(index='Question', columns=top_aois.groupby('Question').cumcount().add(1).rename('Rank'), values='AOI', aggfunc='first')
    top_aois_pivot.columns = ['Rank 1 AOI', 'Rank 2 AOI', 'Rank 3 AOI']
    
    # Convert pivot table to HTML for display
    summary_table_html = top_aois_pivot.to_html(classes="gr-table", border=0)

    accuracy_range = f"{min(question_accuracies.values())} to {max(question_accuracies.values())}"
    
    conclusion_html = f"""
    <div style='padding: 10px; border-radius: 5px; border: 1px solid #ccc;'>
    <h3>Summary: Top 3 Most Important Differentiating AOIs per Question</h3>
    {summary_table_html}
    <h4>Conclusion:</h4>
    <p>This granular, cross-validated analysis confirms that the nature of gender-based gaze differences is highly contextual. The model's ability to predict gender improves significantly when using specific AOI data (accuracies of <b>~{accuracy_range}</b> depending on the question) compared to using high-level averages.</p>
    <p>This demonstrates <b>why</b> the first model failed: the differentiating 'signal' is not in the overall metrics but is spread across different visual elements depending on the image content. This confirms that gender-based gaze differences are real but are tied to specific content, not a uniform behavioral pattern.</p>
    </div>
    """
    
    return fig, conclusion_html

def evaluate_model_gradio(name, model, X_test_data, y_test_data, label_encoder, results_dict, cmap='Blues'):
    """Evaluates a trained model and returns figure, report text."""
    report_text = f"\n--- Evaluating: {name} ---\n"
    y_pred = model.predict(X_test_data)
    accuracy = accuracy_score(y_test_data, y_pred)
    results_dict[name] = accuracy
    report_text += f"Accuracy: {accuracy:.4f}\n"
    report_text += "\nClassification Report:\n"
    
    try:
        target_names = label_encoder.classes_
        report_text += classification_report(y_test_data, y_pred, target_names=target_names) + "\n"
    except ValueError as e:
        report_text += f"Warning: Could not get target names for classification report. {e}\n"
        report_text += classification_report(y_test_data, y_pred) + "\n"

    cm = confusion_matrix(y_test_data, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4)) # Create fig and ax
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=label_encoder.classes_ if hasattr(label_encoder, 'classes_') else ['0', '1'],
                yticklabels=label_encoder.classes_ if hasattr(label_encoder, 'classes_') else ['0', '1'],
                ax=ax)
    ax.set_title(f'Confusion Matrix - {name}\nAccuracy: {accuracy:.2%}')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.tight_layout()
    # plt.close(fig) # Close to prevent display in notebook, Gradio will handle the fig object
    return fig, report_text

# --- ML Model 3: Advanced Feature Engineering & Tuned Comparison ---
def run_advanced_ml_pipeline():
    """
    Runs the advanced ML pipeline with CONDENSED Gradio output.
    SETTINGS ARE ALIGNED WITH THE PROVIDED COLAB SCRIPT.
    Returns:
        - all_cm_figs (list): List of confusion matrix figures.
        - condensed_reports_html (str): Condensed HTML log.
        - summary_accuracy_fig (plt.Figure): Figure of the accuracy summary bar chart.
        - final_conclusion_html (str): HTML string for the conclusion.
        - selected_features_list_html (str): HTML string for the list of selected features.
    """
    global final_original_long_df, le, participant_df_global 

    # --- Ensure 'le' is properly defined and fitted (same as before) ---
    le_instance_ready = False
    if 'le' in globals() and isinstance(le, LabelEncoder) and hasattr(le, 'classes_') and le.classes_.size > 0:
        print(f"INFO (Advanced ML): Using existing globally fitted 'le' with classes: {le.classes_}")
        le_instance_ready = True
    if not le_instance_ready: # Fallback logic
        print("INFO (Advanced ML): Global LabelEncoder 'le' was not fully ready. Attempting to initialize/fit from participant_df_global.")
        if 'le' not in globals() or not isinstance(le, LabelEncoder):
            le = LabelEncoder(); print("INFO (Advanced ML): Global 'le' was not defined, initialized a new one.")
        if 'participant_df_global' in globals() and not participant_df_global.empty and 'Gender' in participant_df_global.columns:
            try:
                unique_genders_fallback = participant_df_global['Gender'].astype(str).unique()
                if len(unique_genders_fallback) > 0:
                    le.fit(unique_genders_fallback)
                    print(f"INFO (Advanced ML): Fallback - Fitted global 'le' with classes: {le.classes_}")
                    if hasattr(le, 'classes_') and le.classes_.size > 0: le_instance_ready = True
                    else: print("ERROR (Advanced ML): Fallback - 'le.fit' did not result in usable classes.")
                else: return [], "<p>ERROR (Advanced ML): Fallback for 'le' failed: No gender data.</p>", None, "", ""
            except Exception as e_fit: return [], f"<p>ERROR (Advanced ML): Could not fit 'le' during fallback: {e_fit}.</p>", None, "", ""
        else: return [], "<p>ERROR (Advanced ML): 'le' crucial, participant_df_global not available for fallback.</p>", None, "", ""
    if not le_instance_ready: return [], "<p>ERROR (Advanced ML): Critical failure for LabelEncoder 'le'.</p>", None, "", ""
    if 'final_original_long_df' not in globals() or final_original_long_df.empty:
        return [], "<p>ERROR (Advanced ML): final_original_long_df not defined or empty.</p>", None, "", ""

    # Imports (same as before)
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler 
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectFromModel
    import lightgbm as lgb
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb
    # evaluate_model_gradio is defined elsewhere
    # sns, plt are used via evaluate_model_gradio or for the summary plot

    # --- Step 1: Advanced Feature Engineering & Initial Data Preparation ---
    # Console logs will still print detailed info from this step
    print("--- Advanced ML Step 1: Feature Engineering ---")
    print(f"Input final_original_long_df shape: {final_original_long_df.shape}")
    base_metrics = ["Tot Fixation dur", "Fixation count", "Time to first Fixation", "Tot Visit dur"]
    base_metrics = [metric for metric in base_metrics if metric in final_original_long_df.columns]
    if not base_metrics: return [], "<p>ERROR: No base_metrics found.</p>", None, "", ""
    
    participant_overall_agg = final_original_long_df.groupby(
        ['Participant_ID', 'Gender']
    )[base_metrics].agg(['mean', 'std', 'median', 'min', 'max', 'sum']).reset_index()
    participant_overall_agg.columns = ['_'.join(col).strip() if col[1] else col[0] for col in participant_overall_agg.columns.values]
    participant_overall_agg.rename(columns={'Participant_ID_': 'Participant_ID', 'Gender_': 'Gender'}, inplace=True)
    engineered_features_list = [participant_overall_agg]
    if 'Image_Type' in final_original_long_df.columns:
        participant_imagetype_agg = final_original_long_df.groupby(
            ['Participant_ID', 'Gender', 'Image_Type']
        )[base_metrics].agg(['mean', 'std']).unstack(level='Image_Type', fill_value=0)
        participant_imagetype_agg.columns = ['_'.join(map(str, col)).strip() for col in participant_imagetype_agg.columns.values]
        participant_imagetype_agg = participant_imagetype_agg.reset_index()
        for metric in base_metrics:
            for stat in ['mean', 'std']:
                real_col, ai_col = f'{metric}_{stat}_Real', f'{metric}_{stat}_AI'
                if real_col in participant_imagetype_agg.columns and ai_col in participant_imagetype_agg.columns:
                    participant_imagetype_agg[f'{metric}_{stat}_Diff_Real_AI'] = participant_imagetype_agg[real_col] - participant_imagetype_agg[ai_col]
        engineered_features_list.append(participant_imagetype_agg)
    final_participant_features = engineered_features_list[0]
    if len(engineered_features_list) > 1:
        for df_to_merge in engineered_features_list[1:]:
            cols_to_drop = ['Gender'] if 'Gender' in df_to_merge.columns else []
            final_participant_features = pd.merge(
                final_participant_features, df_to_merge.drop(columns=cols_to_drop),
                on='Participant_ID', how='left')
    final_participant_features = final_participant_features.fillna(0)
    print(f"Shape of final_participant_features: {final_participant_features.shape}")
    X_full_engineered = final_participant_features.drop(columns=['Participant_ID', 'Gender'])
    y_full_engineered = le.transform(final_participant_features['Gender'])
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X_full_engineered, y_full_engineered, test_size=0.3, random_state=42, stratify=y_full_engineered)
    min_samples_per_class_train = pd.Series(y_train_full).value_counts().min()
    k_neighbors_smote = min(5, max(1, min_samples_per_class_train - 1)) if min_samples_per_class_train > 1 else 1
    smote_instance = SMOTE(random_state=42, k_neighbors=k_neighbors_smote) 
    if X_train_full.shape[0] > 0 and len(np.unique(y_train_full)) > 1 and k_neighbors_smote >=1 :
        try: X_train_balanced_full, y_train_balanced_full = smote_instance.fit_resample(X_train_full, y_train_full)
        except ValueError: X_train_balanced_full, y_train_balanced_full = X_train_full.copy(), y_train_full.copy()
    else: X_train_balanced_full, y_train_balanced_full = X_train_full.copy(), y_train_full.copy()
    print(f"Shape of X_train_balanced_full: {X_train_balanced_full.shape}")

    # --- Step 2: Feature Selection & Preparation ---
    condensed_html_log_parts = [] # For selective HTML output
    selected_features_names = []
    selected_features_list_html = ""
    print("--- Advanced ML Step 2: Feature Selection ---")

    if X_train_balanced_full.empty:
        return [], "<p>ERROR: X_train_balanced_full empty for feature selection.</p>", None, "", ""

    rf_for_selection = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf_for_selection.fit(X_train_balanced_full, y_train_balanced_full)
    feature_names_full = X_full_engineered.columns
    importances = rf_for_selection.feature_importances_
    importance_df = pd.DataFrame({'feature': feature_names_full, 'importance': importances}).sort_values(by='importance', ascending=False)
    
    # CONDENSED OUTPUT: Only top 10 feature importances table
    condensed_html_log_parts.append("<h3>Top Feature Importances</h3>")
    condensed_html_log_parts.append(importance_df.head(10).to_html(classes="gr-table", border=0, float_format='{:.4f}'.format))
    print("\nTop 20 Feature Importances (Console Log):\n", importance_df.head(20)) # Console log for more details

    sfm = SelectFromModel(rf_for_selection, threshold='median', prefit=True) 
    X_train_balanced_selected_np = sfm.transform(X_train_balanced_full)
    X_test_selected_np = sfm.transform(X_test_full)
    selected_feature_indices = sfm.get_support(indices=True)
    selected_features_names = [feature_names_full[i] for i in selected_feature_indices]

    if not selected_features_names: 
        condensed_html_log_parts.append("<p><b>Warning: No features selected by SelectFromModel. Model training skipped.</b></p>")
        return [], "".join(condensed_html_log_parts), None, "No features selected, models not tuned.", "No features selected."

    X_train_balanced_selected = pd.DataFrame(X_train_balanced_selected_np, columns=selected_features_names)
    X_test_selected = pd.DataFrame(X_test_selected_np, columns=selected_features_names)
    selected_features_list_html = "<h4>Selected Features:</h4><ul>" + "".join([f"<li>{f}</li>" for f in selected_features_names]) + "</ul>"
    print(f"Number of selected features: {len(selected_features_names)}")

    X_train_scaled_selected, X_test_scaled_selected = pd.DataFrame(), pd.DataFrame()
    if not X_train_balanced_selected.empty and X_train_balanced_selected.shape[1] > 0 :
        scaler_selected = StandardScaler()
        X_train_scaled_selected_np = scaler_selected.fit_transform(X_train_balanced_selected)
        X_test_scaled_selected_np = scaler_selected.transform(X_test_selected)
        X_train_scaled_selected = pd.DataFrame(X_train_scaled_selected_np, columns=selected_features_names)
        X_test_scaled_selected = pd.DataFrame(X_test_scaled_selected_np, columns=selected_features_names)
    
    # --- Step 3: Model Training & Tuning ---
    print("--- Advanced ML Step 3: Model Tuning (Aligned with Colab) ---")
    model_accuracies_tuned_selected = {}
    all_cm_figs = [] # To store confusion matrix figures for Gradio

    if X_train_balanced_selected.empty or X_train_balanced_selected.shape[1] == 0:
        condensed_html_log_parts.append("<p>ERROR: No selected features for model tuning.</p>")
        return all_cm_figs, "".join(condensed_html_log_parts), None, "Models not tuned (no features).", selected_features_list_html

    # Logistic Regression
    if not X_train_scaled_selected.empty:
        print("\n--- Tuning Logistic Regression ---")
        lr_param_dist = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 500, 1000],'penalty': ['l1', 'l2'],'solver': ['liblinear', 'saga']}
        lr_rand_search = RandomizedSearchCV(LogisticRegression(random_state=42, max_iter=3000, class_weight='balanced'),
                                       lr_param_dist, n_iter=15, cv=5, scoring='accuracy', n_jobs=-1, random_state=42, verbose=0)
        lr_rand_search.fit(X_train_scaled_selected, y_train_balanced_full)
        print("Best LR params:", lr_rand_search.best_params_)
        fig, _ = evaluate_model_gradio("Log.Reg. (Sel. Feat. Tuned)", lr_rand_search.best_estimator_, X_test_scaled_selected, y_test_full, le, model_accuracies_tuned_selected, cmap='Purples')
        all_cm_figs.append(fig)
    else: all_cm_figs.append(None) # Placeholder if skipped

    # RandomForest
    print("\n--- Tuning RandomForest ---")
    rf_param_dist = {'n_estimators': [50, 100, 200, 300], 'max_depth': [None, 5, 10, 15, 20], 'min_samples_split': [2, 5, 10, 15], 'min_samples_leaf': [1, 2, 4, 6], 'max_features': ['sqrt', 'log2', 0.6, 0.8]}
    rf_rand_search = RandomizedSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'),
                                   rf_param_dist, n_iter=25, cv=5, scoring='accuracy', n_jobs=-1, random_state=42, verbose=0)
    rf_rand_search.fit(X_train_balanced_selected, y_train_balanced_full)
    print("Best RF params:", rf_rand_search.best_params_)
    fig, _ = evaluate_model_gradio("RF (Sel. Feat. Tuned)", rf_rand_search.best_estimator_, X_test_selected, y_test_full, le, model_accuracies_tuned_selected, cmap='Greens')
    all_cm_figs.append(fig)

    # LightGBM
    print("\n--- Tuning LightGBM ---")
    lgbm_param_dist = {'n_estimators': [50, 100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1, 0.2], 'num_leaves': [15, 20, 31, 40], 'max_depth': [-1, 5, 10, 15], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9], 'subsample': [0.6, 0.7, 0.8, 0.9]}
    lgbm_rand_search = RandomizedSearchCV(lgb.LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1),
                                     lgbm_param_dist, n_iter=25, cv=5, scoring='accuracy', n_jobs=-1, random_state=42, verbose=0)
    lgbm_rand_search.fit(X_train_balanced_selected, y_train_balanced_full)
    print("Best LightGBM params:", lgbm_rand_search.best_params_)
    fig, _ = evaluate_model_gradio("LGBM (Sel. Feat. Tuned)", lgbm_rand_search.best_estimator_, X_test_selected, y_test_full, le, model_accuracies_tuned_selected, cmap='Blues')
    all_cm_figs.append(fig)

    # SVM
    if not X_train_scaled_selected.empty:
        print("\n--- Tuning SVM ---")
        svm_param_dist = {'C': [0.1, 1, 10, 50, 100], 'gamma': ['scale', 'auto', 0.001, 0.01, 0.1], 'kernel': ['rbf', 'linear', 'poly']}
        svm_rand_search = RandomizedSearchCV(SVC(random_state=42, probability=True, class_weight='balanced'),
                                        svm_param_dist, n_iter=20, cv=5, scoring='accuracy', n_jobs=-1, random_state=42, verbose=0)
        svm_rand_search.fit(X_train_scaled_selected, y_train_balanced_full)
        print("Best SVM params:", svm_rand_search.best_params_)
        fig, _ = evaluate_model_gradio("SVM (Sel. Feat. Tuned)", svm_rand_search.best_estimator_, X_test_scaled_selected, y_test_full, le, model_accuracies_tuned_selected, cmap='Oranges')
        all_cm_figs.append(fig)
    else: all_cm_figs.append(None) # Placeholder if skipped

    # XGBoost
    print("\n--- Tuning XGBoost ---")
    xgb_param_dist = {'n_estimators': [50, 100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1, 0.2], 'max_depth': [3, 5, 7, 9], 'gamma': [0, 0.1, 0.2, 0.3], 'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}
    xgb_rand_search = RandomizedSearchCV(xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
                                    xgb_param_dist, n_iter=25, cv=5, scoring='accuracy', n_jobs=-1, random_state=42, verbose=0)
    xgb_rand_search.fit(X_train_balanced_selected, y_train_balanced_full)
    print("Best XGBoost params:", xgb_rand_search.best_params_)
    fig, _ = evaluate_model_gradio("XGBoost (Sel. Feat. Tuned)", xgb_rand_search.best_estimator_, X_test_selected, y_test_full, le, model_accuracies_tuned_selected, cmap='magma')
    all_cm_figs.append(fig)

    # --- Summary of Accuracies (This will be part of HTML output) ---
    summary_accuracy_fig = None
    condensed_html_log_parts.append("<hr/><h3>Summary of Model Accuracies (Tuned on Selected Features)</h3>")
    if model_accuracies_tuned_selected:
        accuracies_df_tuned_selected = pd.DataFrame(list(model_accuracies_tuned_selected.items()), columns=['Model', 'Accuracy'])
        accuracies_df_tuned_selected = accuracies_df_tuned_selected.sort_values(by='Accuracy', ascending=False)
        
        fig_summary, ax_summary = plt.subplots(figsize=(10, max(5, len(accuracies_df_tuned_selected) * 0.4)))
        sns.barplot(x='Accuracy', y='Model', data=accuracies_df_tuned_selected, palette='viridis', ax=ax_summary)
        ax_summary.set_title('Model Accuracy Comparison (Tuned on Selected Features)', fontsize=15)
        ax_summary.set_xlim(0, 1.1)
        for i_bar, acc_val in enumerate(accuracies_df_tuned_selected['Accuracy']):
          ax_summary.text(acc_val + 0.01, i_bar, f"{acc_val:.3%}", color='black', ha="left", va='center')
        plt.tight_layout()
        summary_accuracy_fig = fig_summary
        condensed_html_log_parts.append(accuracies_df_tuned_selected.to_html(classes="gr-table", border=0, float_format='{:.4f}'.format))
    else:
        condensed_html_log_parts.append("<p>No model accuracies to summarize.</p>")

    # --- Simplified Final Conclusion ---
    best_model_name = ""
    best_accuracy = 0
    if model_accuracies_tuned_selected:
        best_model_name = max(model_accuracies_tuned_selected, key=model_accuracies_tuned_selected.get)
        best_accuracy = model_accuracies_tuned_selected[best_model_name]

    final_conclusion_html = "<h3>Final Conclusion</h3>"
    if best_model_name:
        final_conclusion_html += f"<p>The best performing model using advanced feature engineering and tuning was <b>{best_model_name}</b> with an accuracy of <b>{best_accuracy:.2%}</b>.</p>"
        # You'll need a way to get Model 1's accuracy here if you want a direct comparison sentence.
        # For now, a general statement:
        final_conclusion_html += "<p>This approach attempts a more sophisticated feature set and hyperparameter optimization compared to the simpler overall averages used in Model 1. The resulting accuracy indicates the predictive power achieved with these advanced techniques for this dataset.</p>"
    else:
        final_conclusion_html += "<p>Advanced modeling did not yield a best performing model, or no features were selected. Please check console logs for details.</p>"
    
    # Ensure exactly 5 CM figures are returned, using None for skipped models
    # The order should match your Gradio plot components: LR, RF, LGBM, SVM, XGB
    # Current all_cm_figs has them in order, but SVM/LR might be None if skipped.
    # We need to ensure the list has 5 elements.
    # Assuming they are appended in order: LR, RF, LGBM, SVM, XGB
    # If LR skipped, all_cm_figs[0] would be for RF if not handled.
    # Let's reconstruct carefully based on which models ran:
    
    final_cm_outputs = [None] * 5 # LR, RF, LGBM, SVM, XGB
    cm_idx = 0
    if not X_train_scaled_selected.empty: # LR ran
        final_cm_outputs[0] = all_cm_figs[cm_idx] if cm_idx < len(all_cm_figs) else None
        cm_idx +=1
    
    final_cm_outputs[1] = all_cm_figs[cm_idx] if cm_idx < len(all_cm_figs) else None # RF
    cm_idx +=1
    final_cm_outputs[2] = all_cm_figs[cm_idx] if cm_idx < len(all_cm_figs) else None # LGBM
    cm_idx +=1

    if not X_train_scaled_selected.empty: # SVM ran
        final_cm_outputs[3] = all_cm_figs[cm_idx] if cm_idx < len(all_cm_figs) else None
        cm_idx +=1
    
    final_cm_outputs[4] = all_cm_figs[cm_idx] if cm_idx < len(all_cm_figs) else None # XGB
    
    return final_cm_outputs, "".join(condensed_html_log_parts), summary_accuracy_fig, final_conclusion_html, selected_features_list_html


# --- GRADIO INTERFACE ---
def create_gradio_interface():
    """Creates the Gradio UI with custom CSS and JS for theming."""
    global all_balanced_long_dfs, selected_metric_sheets
    question_options = ['All Combined'] + sorted(list(all_balanced_long_dfs.keys()))

    js_theme_handler = """
    function(theme_state) {
        const gradio_app = document.querySelector('gradio-app');
        if (gradio_app) {
            const root = gradio_app.shadowRoot || gradio_app;
            if (theme_state === 'light') {
                root.querySelector('.gradio-container').classList.add('light');
            } else {
                root.querySelector('.gradio-container').classList.remove('light');
            }
        }
        return theme_state;
    }
    """
    
    custom_css = """
    :root {
        --dark-bg: #000000; --dark-panel-bg: #16213E; --dark-input-bg: #1A1A2E; --dark-text: #FFFFFF; --dark-border: #00D4FF;
        --light-bg: #FFFFFF; --light-panel-bg: #FFFFFF; --light-input-bg: #F8F9FA; --light-text: #2C3E50; --light-border: #DEE2E6;
    }
    .gradio-container .gr-checkbox-group label span, .gradio-container .gr-check-radio span { color: var(--dark-text) !important; }
    .gradio-container.light .gr-checkbox-group label span, .gradio-container.light .gr-check-radio span { color: var(--light-text) !important; }
    .gr-table { width: 100%; border-collapse: collapse; }
    .gr-table th, .gr-table td { padding: 8px; border: 1px solid #ddd; text-align: left; }
    .gradio-container.light .gr-table { color: var(--light-text); } 
    .gradio-container:not(.light) .gr-table { color: var(--dark-text); }
    """

    with gr.Blocks(css=custom_css, title="Eye-Tracking Analytics Dashboard") as demo:
        theme_state = gr.Textbox("dark", visible=False)
        
        gr.HTML("""
        <div style='background: linear-gradient(135deg, #00D4FF 0%, #FF6B6B 100%); padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 20px;'>
            <h1 style='color: white; font-size: 2.0em; margin: 0; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
                🧠 Eye-Tracking Analytics Dashboard
            </h1>
            <p style='color: rgba(255,255,255,0.9); font-size: 1.2em; margin: 10px 0 0 0; font-weight: 300;'>
                Visual Attention Differences Between AI-Generated and Real Images Based on Gender
            </p>
        </div>
        """)
        
        with gr.Accordion("ℹ️ Dataset Overview & Demographics", open=True):
            with gr.Row():
                with gr.Column(scale=2):
                    total_participants_box = gr.Number(label="Total Unique Participants", interactive=False)
                    male_count_box = gr.Number(label="Male Participants", interactive=False)
                    female_count_box = gr.Number(label="Female Participants", interactive=False)
                with gr.Column(scale=1):
                    gender_pie_chart = gr.Plot()
        
        with gr.Row():
            with gr.Column(scale=10):
                with gr.Row():
                    question_select = gr.Dropdown(choices=question_options, value=question_options[0], label="📋 Select Question Set")
                    metric_select = gr.Dropdown(choices=selected_metric_sheets, value=selected_metric_sheets[0], label="📊 Select Metric")
            with gr.Column(scale=1, min_width=80):
                theme_toggle_btn = gr.Button("🌙", elem_classes="theme-btn")
        
        with gr.Tabs():
            with gr.TabItem("📊 Interactive Bar Chart Analysis"):
                image_type_filter = gr.CheckboxGroup(choices=["Real", "AI"], value=["Real", "AI"], label="Filter by Image Type", interactive=True)
                bar_plot = gr.Plot(show_label=False)
            with gr.TabItem("🔍 Scatter Analysis"):
                scatter_plot = gr.Plot(show_label=False)
            with gr.TabItem("🔥 Correlation Heatmap"):
                heatmap_plot = gr.Plot(show_label=False)
            with gr.TabItem("📈 Comprehensive Analysis"):
                dashboard_plot = gr.Plot(show_label=False)
        
        with gr.Accordion("🧪 Statistical Test Results", open=False):
            with gr.Tabs():
                with gr.TabItem("Normality & Test Selection"):
                    gr.Markdown("For each metric and question, data is checked for normal distribution using the Shapiro-Wilk test. If both male and female data are normal, a t-test is used for comparison; otherwise, the non-parametric Mann-Whitney U test is used.")
                    normality_test_table = gr.Plot()
                with gr.TabItem("P-Value Summary"):
                    gr.Markdown("P-values from the selected statistical tests comparing male and female gaze behavior. Green cells indicate a statistically significant difference (p < 0.05).")
                    statistical_test_plot = gr.Plot()

        with gr.Accordion("🤖 Predictive Model", open=False):
            with gr.Tabs():
                with gr.TabItem("Simple RF Model"):
                    gr.Markdown("""
                    This predictor uses **Model 1** that is trained on the overall average gaze behavior of all participants to predict gender. 
                    Use the sliders to input hypothetical average metrics and see the model's prediction.
                    """)
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2):
                            gr.Markdown("### Input Hypothetical Averages")
                            slider_fix_dur = gr.Slider(0, 5, value=1.05, label="Avg. Total Fixation Duration (s)", interactive=True)
                            slider_fix_count = gr.Slider(0, 20, value=7, step=1, label="Avg. Fixation Count", interactive=True)
                            slider_ttf = gr.Slider(0, 100, value=95, label="Avg. Time to First Fixation (ms)", interactive=True)
                            slider_visit_dur = gr.Slider(0, 10, value=2.1, label="Avg. Total Visit Duration (s)", interactive=True)
                            predict_btn = gr.Button("Predict Gender Likelihood")
                        with gr.Column(scale=1):
                            gr.Markdown("### Prediction Results")
                            prediction_label = gr.Label(label="Predicted Gender Likelihood", num_top_classes=2)
        
        # --- NEW MACHINE LEARNING ACCORDION ---
        with gr.Accordion("👨🏻‍💻 Machine Learning Investigation", open=False):
            with gr.Tabs():
                with gr.TabItem("Model 1: Are Gaze Differences Practically Predictive?"):
                    gr.Markdown("""
                    ### Methodology
                    While initial statistical tests revealed significant differences in specific gaze metrics, this does not guarantee a coherent, predictive pattern. To test the practical significance, we built a machine learning model to see if a participant's **gender** could be predicted from their **overall average gaze behavior** across all questions.
                    """)
                    run_model1_btn = gr.Button("Run Predictive Power Analysis")
                    model1_plot = gr.Plot(show_label=False)
                    model1_conclusion = gr.HTML()

                with gr.TabItem("Model 2: Which AOIs Drive Gaze Differences?"):
                    gr.Markdown("""
                    ### Methodology
                    The first model showed that high-level averages are not predictive. This suggests differences are tied to specific visual elements. To investigate this, we use a more granular model that uses the **fixation duration on each individual Area of Interest (AOI)** as features to predict gender, providing a stable ranking of which AOIs are most important in each context.
                    """)
                    run_model2_btn = gr.Button("Run AOI Importance Analysis")
                    model2_plot = gr.Plot(show_label=False)
                    model2_conclusion = gr.HTML()

                with gr.TabItem("Model 3: Advanced Prediction & Algorithm Comparison"):
                    gr.Markdown("""
                    ### Methodology
                    This analysis takes a comprehensive approach to predict gender using eye-tracking data. It involves:
                    1.  **Advanced Feature Engineering:** Creating detailed statistical features from base metrics (overall, per image type, differences).
                    2.  **Data Balancing (SMOTE):** Addressing class imbalance in the training data for the engineered features.
                    3.  **Feature Selection:** Using RandomForest importances (SelectFromModel with 'median' threshold) to identify the most relevant features from the engineered set.
                    4.  **Hyperparameter Tuning & Model Comparison:** Evaluating several algorithms (Logistic Regression, RandomForest, LightGBM, SVM, XGBoost) after tuning them with RandomizedSearchCV on the selected features.
                    The goal is to determine the best possible predictive performance with these sophisticated techniques.
                    **Note:** This process can take some time to run due to the comprehensive computations.
                    """)
                    run_model3_btn = gr.Button("Run Advanced ML Analysis (may take time)")
                    
                    model3_selected_features_html = gr.HTML(label="Selected Features by Model 3") # Label for clarity

                    model3_reports_html = gr.HTML(label="Detailed Logs & Model Reports from Model 3") # Label
                    
                    with gr.Row():
                        model3_cm_plot1 = gr.Plot(label="CM LogReg")
                        model3_cm_plot2 = gr.Plot(label="CM RF")
                        model3_cm_plot3 = gr.Plot(label="CM LGBM")
                    with gr.Row():
                        model3_cm_plot4 = gr.Plot(label="CM SVM")
                        model3_cm_plot5 = gr.Plot(label="CM XGB")
                    
                    model3_accuracy_summary_plot = gr.Plot(label="Overall Accuracy Comparison from Model 3") # Label
                    
                    model3_conclusion_html = gr.HTML(label="Final Conclusion from Model 3")

        # --- Event Handlers ---
        inputs = [question_select, metric_select, image_type_filter]
        prediction_inputs = [slider_fix_dur, slider_fix_count, slider_ttf, slider_visit_dur]
        predict_btn.click(fn=run_prediction, inputs=prediction_inputs, outputs=prediction_label)
        dynamic_ui_outputs = [
            total_participants_box, male_count_box, female_count_box, gender_pie_chart,
            dashboard_plot, heatmap_plot, bar_plot, scatter_plot,
        ]
        static_ui_outputs = [normality_test_table, statistical_test_plot]

        # Handlers for main dashboard
        for control in inputs:
            control.change(fn=update_all_outputs, inputs=inputs, outputs=dynamic_ui_outputs)
        
        theme_toggle_btn.click(
            fn=handle_theme_toggle,
            inputs=[theme_state] + inputs,
            outputs=[theme_state] + dynamic_ui_outputs + static_ui_outputs + [theme_toggle_btn]
        )
        
        theme_state.change(js=js_theme_handler, inputs=[theme_state], outputs=None)
        
        # Handlers for ML models
        run_model1_btn.click(fn=run_predictive_power_analysis, inputs=None, outputs=[model1_plot, model1_conclusion])
        run_model2_btn.click(fn=run_aoi_importance_analysis, inputs=None, outputs=[model2_plot, model2_conclusion])

        model3_output_components = [
            model3_cm_plot1, model3_cm_plot2, model3_cm_plot3, model3_cm_plot4, model3_cm_plot5,
            model3_reports_html, 
            model3_accuracy_summary_plot, 
            model3_conclusion_html,
            model3_selected_features_html
        ]

        def model3_wrapper_function():
            # This wrapper unpacks the list of CM figures. Max 5 CM figures expected.
            cm_figs, reports_text, acc_summary_fig, conclusion_html, sel_features_html = run_advanced_ml_pipeline()
            
            # Prepare the CM plot outputs. If fewer than 5 figs, pass None.
            output_cm_figs = [None] * 5 
            if isinstance(cm_figs, list): # Ensure cm_figs is a list
                for i in range(min(len(cm_figs), 5)):
                    output_cm_figs[i] = cm_figs[i]
            
            # Ensure all returned values are in the correct order for the output_components list
            return (
                # These must match the order in model3_output_components
                output_cm_figs[0], output_cm_figs[1], output_cm_figs[2], output_cm_figs[3], output_cm_figs[4],
                reports_text, 
                acc_summary_fig, 
                conclusion_html,
                sel_features_html
            )

        run_model3_btn.click(fn=model3_wrapper_function, inputs=None, outputs=model3_output_components)
        
        def initial_load_wrapper():
            dynamic_outputs = update_all_outputs(
                question=question_options[0],
                metric=selected_metric_sheets[0],
                image_types_to_show=["Real", "AI"]
            )
            normality_fig = create_normality_test_table()
            stats_fig = create_statistical_test_plot()
            return list(dynamic_outputs) + [normality_fig, stats_fig]
            
        demo.load(fn=initial_load_wrapper, inputs=None, outputs=dynamic_ui_outputs + static_ui_outputs)
        
    return demo

# --- Run the App ---
if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch(debug=True, share=False)
