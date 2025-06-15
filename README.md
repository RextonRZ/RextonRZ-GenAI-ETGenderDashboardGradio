---
title: Eye Tracking Analytics Dashboard
emoji: 👁️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 👁️ Eye-Tracking Analytics Dashboard

An interactive web application for analyzing eye-tracking data across different question sets and metrics. This dashboard provides comprehensive visualization and analysis tools for understanding gaze patterns, fixation behaviors, and attention distribution.

## 🚀 Features

### Interactive Visualizations
- **📊 Bar Chart Analysis**: Compare metrics across different Areas of Interest (AOIs) by gender
- **🔍 Correlation Analysis**: Scatter plots showing relationships between fixation duration and count  
- **📈 Multi-Dimensional Dashboard**: Comprehensive analysis with box plots, histograms, violin plots, and summary statistics

### Metrics Supported
- **Total Fixation Duration**: Time spent looking at specific areas
- **Fixation Count**: Number of individual fixations
- **Time to First Fixation**: How quickly participants notice areas
- **Total Visit Duration**: Overall time spent in regions

### Data Organization
- **Question Sets Q1-Q6**: Organized by different experimental conditions
- **Gender-based Analysis**: Compare patterns between male and female participants
- **Image Type Comparison**: Analyze differences between A/B image variants
- **AOI-level Insights**: Detailed breakdowns by Areas of Interest

## 🎯 Use Cases

- **Research Analysis**: Academic studies on visual attention and perception
- **UX Research**: Understanding user interface engagement patterns
- **Marketing Research**: Analyzing advertisement effectiveness and visual hierarchy
- **Educational Studies**: Investigating learning patterns and visual processing

## 📊 Data Structure

The dashboard processes eye-tracking data with the following structure:
- **Participants**: Individual subject data with demographic information
- **Questions**: Different experimental scenarios (Q1-Q6)
- **AOIs**: Specific regions of interest within each stimulus
- **Metrics**: Quantitative measures of eye movement behavior

## 🛠️ Technical Stack

- **Frontend**: Gradio 4.44.0 for interactive web interface
- **Visualization**: Plotly for dynamic, interactive charts
- **Data Processing**: Pandas for data manipulation and analysis
- **Computation**: NumPy for numerical operations
- **File Handling**: OpenPyXL for Excel file processing

## 📈 Getting Started

1. **Select Question Set**: Choose from Q1-Q6 or view all combined data
2. **Choose Metric**: Pick the eye-tracking measure you want to analyze
3. **Explore Visualizations**: Navigate through different chart types in the tabs
4. **Interactive Analysis**: Hover over charts for detailed data points

## 📋 Sample Data

If no data files are found, the dashboard automatically generates sample data for demonstration purposes, allowing you to explore all features and functionality.

## 🔧 Development

This dashboard is built with modern web technologies and deployed on Hugging Face Spaces for easy access and sharing. The interface is responsive and works across different devices and screen sizes.

## 📄 License

MIT License - Feel free to use and modify for your research and projects.

---

*Built with ❤️ for the eye-tracking research community*