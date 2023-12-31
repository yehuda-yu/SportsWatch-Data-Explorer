# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 21:39:06 2023

@author: Yehuda Yungstein
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend("Agg")
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from scipy import stats
import functions
import plotly.express as px
import plotly.graph_objs as go

# Define the color palette
color_palette = ["#21b0fe", "#fed700", "#fe218b"]
sns.set_theme(style="white", font_scale=1.5, palette=color_palette)

# Set the font to Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Arial'] + plt.rcParams['font.serif']

# Create a Streamlit app
# Set page icon
st.set_page_config(page_icon="🥇")
st.title("Garmin Running Visualization and Insights")

# Section: File Upload
st.subheader("1. Upload Data")
use_sample_data = st.checkbox("Use Example Data")
uploaded_file = None
if not use_sample_data:
    uploaded_file = st.file_uploader("Upload a CSV file with your running data", type="csv")

# Section: Data Analysis and Visualization
if use_sample_data or uploaded_file is not None:
    # Read CSV file into a DataFrame
    if use_sample_data:
        df = pd.read_csv("sample_running_data.csv")  # Use the provided sample file
    else:
        df = pd.read_csv(uploaded_file)

    # Clean the data using functions
    df = functions.clean_running_data(df)

    # Display the cleaned DataFrame
    st.subheader("2. Data Overview")
    st.dataframe(df)
    with st.expander("Explain the meaning and units of each parameter in the data"):
        functions.show_parameters_table()

    # Section: Visualizations
    st.markdown("---")
    st.subheader('3. Visualizations')
    
   # Define the parameter columns for line charts
    columns = df[['Distance', 'Calories', 'Avg HR', 'Max HR', 'Aerobic TE', 'Avg Run Cadence',
       'Max Run Cadence', 'Avg Pace', 'Best Pace', 'Total Ascent',
       'Total Descent', 'Avg Stride Length', 'Avg Vertical Ratio',
       'Avg Vertical Oscillation', 'Avg Ground Contact Time',
       'Training Stress Score®', 'Avg Power', 'Max Power', 'Min Temp',
       'Surface Interval', 'Max Temp', 'Elapsed Time', 'Min Elevation',
       'Max Elevation']].columns
    parameter = st.selectbox("Select a parameter to visualize", columns)

    # Display line charts for monthly averages of selected parameters
    fig = functions.get_chart(df, parameter)
    st.plotly_chart(fig)
    
    
    st.markdown("""### Select your own visualization""")
    # Select column name
    column_name = st.selectbox("Select a parameter to visualize", df.columns,4)

    # Define interval options with user-friendly labels
    interval_options = {
        'Weekly': 'W',
        'Monthly': 'M',
        '3-Months': '3M',
        'Semi-Annual': '6M',
        'Annual': 'A'
    }

    # Select interval using user-friendly labels
    interval_label = st.selectbox("Select an interval of time for visualization", list(interval_options.keys()))

    # Retrieve corresponding interval value from the options dictionary
    interval = interval_options[interval_label]

    # Select calculation type
    calculation = st.selectbox("Select a calculation for visualization", ['average', 'sum', 'maximum', 'minimum'])

    # Plot the interval statistics if the button is clicked
    if st.button("Plot Visualizations"):
        st.subheader("Interval Statistics")
        functions.plot_interval_statistics(df, column_name, interval, calculation)

        st.subheader("Distribution with Average")
        functions.plot_distribution_with_average(df, column_name)

        st.subheader("Day vs. Night Performance")
        functions.compare_day_night_performance(df, column_name)

# Section: Report
st.markdown("---")
st.subheader("4. Report - Analysis & Recommendations")
if use_sample_data or uploaded_file is not None:
    
    # Heart Rate Analysis Expander
    with st.expander("Check Heart Rates"):
        st.write("Enter your details below:")
        age = st.slider("Age", min_value=20, max_value=70, step=5)
        gender = st.selectbox("Gender", ['Male', 'Female'])

        # Calculate heart rate statistics for the last year and last month
        average_last_year = round(df[df['Date'].dt.year == df['Date'].dt.year.max()]['Max HR'].mean(), 1)
        average_last_month = round(df[df['Date'].dt.month == df['Date'].dt.month.max()]['Max HR'].mean(), 1)
        max_last_year = round(df[df['Date'].dt.year == df['Date'].dt.year.max()]['Max HR'].mean(), 1)
        max_last_month = round(df[df['Date'].dt.month == df['Date'].dt.month.max()]['Max HR'].mean(), 1)
    
        # Check button for heart rate analysis
        if st.button("Check Heart Rates"):
            st.subheader("Heart Rate Analysis Results")
            st.write("Your last year heart rates:")
            functions.check_heart_rate_normal(average_last_year, max_last_year, age, gender)

            st.write("Your last month heart rates:")
            functions.check_heart_rate_normal(average_last_month, max_last_month, age, gender)

    # Duration Distribution Expander
    with st.expander("Check Stride and Cadence"):
        st.subheader("Stride and Cadence")
        functions.analyse_Cadence_Stride(df)

    # Duration Distribution Expander
    with st.expander("Check Duration"):
        st.subheader("Duration")
        functions.analyze_activity_duration(df,'Time')
        
    # Duration Distribution Expander
    with st.expander("Check Temperature Effect"):
        # check performance in different temperatures
        functions.analyze_temperature_impact(df, 'Max Temp', 'Min Temp', 'Avg Pace', 'Avg HR', 'Max HR')
        # Check correlations
        st.subheader("Correlation Heatmap off Temperature")
        st.write("display a correlation heatmap for the selected temperature column and highly correlated columns.")
        
        #plot heatmap of correlations 
        functions.plot_correlation_heatmap(df, ['Max Temp', 'Min Temp'])    
  
    # Records Expander
    with st.expander("Check Your Records"):
        st.subheader("Records")
        functions.identify_personal_records(df)


# Credits
st.markdown("**Developed by**: Yehuda Yungstein")

# Set symbols and links 
st.markdown(
    """
    <style>
        div[data-testid="column"]:nth-of-type(1)
        {
            text-align: right;
        } 

        div[data-testid="column"]:nth-of-type(2)
        {
            text-align: left;
        } 
    </style>
    """,unsafe_allow_html=True
)

col1, col2, = st.columns(2,gap = "small")

with col1:
      # URL of the image
      image_url = "https://img.icons8.com/small/256/new-post.png"
      # Mail URL
      mail_url = "mailto:yehudayu@gmail.com"
      st.write("<a href='" + mail_url + "'><img src='" + image_url + "' width='50' height='50'></a>", unsafe_allow_html=True)


with col2:
      # URL of the image
      image_url = "https://img.icons8.com/small/256/linkedin.png"
      # LinkedIn URL
      linkedin_url = "https://www.linkedin.com/in/yehuda-yungstein/"
      st.write("<a href='" + linkedin_url + "'><img src='" + image_url + "' width='50' height='50'></a>", unsafe_allow_html=True)
