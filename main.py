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
st.title("Garmin Running Visualization and Insights")

# Section: File Upload
st.subheader("1. Upload Data")
use_sample_data = st.checkbox("Use Sample Data")
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
    
    # ... (rest of the code remains the same)
    
    # Records Expander
    with st.expander("Check Your Records"):
        st.subheader("Records")
        functions.identify_personal_records(df)
