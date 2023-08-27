# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 21:27:15 2023

@author: User
"""
import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objs as go
# Define the color palette
color_palette = ["#21b0fe","#fed700","#fe218b"]
sns.set_theme(style="white",font_scale = 1.5,palette = color_palette)
# Set the font to Times new roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Arial'] + plt.rcParams['font.serif']
plt.switch_backend("Agg")

def clean_running_data(df):
    """
    Clean and preprocess running data from a DataFrame by convert to int/float/datetime formats.

    Args:
        df (pandas.DataFrame): Input DataFrame containing the running data.

    Returns:
        pandas.DataFrame: Cleaned DataFrame with the running data.
    """
    try:
          # Replace '--' with NaN
          df.replace('--', np.nan, inplace=True)
        
          # Drop rows with NaN values only if entire column is not NaN
          for col in df.columns:
              if df[col].notna().any():
                  df = df.dropna(subset=[col])
        
          # Columns to convert to int/float
          cols_to_int = ['Calories', 'Avg Run Cadence', 'Max Run Cadence', 'Total Ascent', 'Total Descent', 'Min Elevation', 'Max Elevation']
          cols_to_float = ['Aerobic TE']
        
          # Remove commas from numeric values
          df[cols_to_int] = df[cols_to_int].replace(',', '', regex=True)
          df[cols_to_float] = df[cols_to_float].replace(',', '', regex=True)
        
          # Convert columns to appropriate data types
          df[cols_to_int] = df[cols_to_int].astype('Int64')
          df[cols_to_float] = df[cols_to_float].astype(float)
        
          # Define time formats for parsing
          time_formats = {
              'Date': '%Y-%m-%d %H:%M:%S',
              'Time': '%H:%M:%S',
              'Avg Pace': '%M:%S',
              'Best Pace': '%M:%S',
              'Moving Time': '%H:%M:%S',
              'Best Lap Time': '%M:%S.%f',
              'Elapsed Time': '%H:%M:%S'
          }
        
          # Convert time-related columns to datetime data type
          for column, time_format in time_formats.items():
              df[column] = pd.to_datetime(df[column], format=time_format, errors='coerce')
        
          # Drop nan after all proccess
          df = df.dropna()
        
          # Convert time-related columns to desired display format
          df['Time'] = df['Time'].dt.strftime('%H:%M:%S')
          df['Avg Pace'] = df['Avg Pace'].dt.strftime('%M:%S')
          df['Best Pace'] = df['Best Pace'].dt.strftime('%M:%S')
          df['Best Lap Time'] = df['Best Lap Time'].dt.strftime('%M:%S.%f')
          df['Moving Time'] = df['Moving Time'].dt.strftime('%H:%M:%S')
          df['Elapsed Time'] = df['Elapsed Time'].dt.strftime('%H:%M:%S')
          df['Hour'] = df['Date'].dt.round('H').dt.hour
          # Create a new column 'DayOfWeek' containing the day of the week
          df['DayOfWeek'] = df['Date'].dt.day_name()
        
              # Create "season" column
          df['season'] = df['Date'].dt.month.map({12: 'winter', 1: 'winter', 2: 'winter',
                                                  3: 'spring', 4: 'spring', 5: 'spring',
                                                  6: 'summer', 7: 'summer', 8: 'summer',
                                                  9: 'autumn', 10: 'autumn', 11: 'autumn'})
        
        
          # Convert Avg Pace and Best Pace columns to minute.percent format
          df[['Minutes', 'Seconds']] = df['Avg Pace'].str.split(':', expand=True).astype(int)
          df['Avg Pace'] = df['Minutes'] + df['Seconds'] / 60
          df[['Minutes', 'Seconds']] = df['Best Pace'].str.split(':', expand=True).astype(int)
          df['Best Pace'] = df['Minutes'] + df['Seconds'] / 60
        
          # drop the columns
          df.drop(columns=['Minutes', 'Seconds'], inplace=True)

          return df

    except Exception as e:
        st.error(f"An error occurred while cleaning the running data: {e}")
        return None

def show_parameters_table():    
    """
    Show the the meaning and units of each parameter
    """
    table =     st.markdown('''
                | Parameter | Meaning | Units |
                |---|---|---|
                | Activity Type | The type of activity, such as running, cycling, or swimming | - |
                | Date | The date of the activity | - |
                | Favorite | Whether the activity is marked as a favorite | Boolean |
                | Title | The title of the activity | - |
                | Distance | The distance traveled in the activity | meters |
                | Calories | The number of calories burned in the activity | calories |
                | Time | The total time of the activity | seconds |
                | Avg HR | The average heart rate during the activity | beats per minute |
                | Max HR | The maximum heart rate during the activity | beats per minute |
                | Aerobic TE | The aerobic training effect of the activity | - |
                | Avg Run Cadence | The average cadence (steps per minute) during the activity | steps per minute |
                | Max Run Cadence | The maximum cadence during the activity | steps per minute |
                | Avg Pace | The average pace (minutes per kilometer) during the activity | minutes per kilometer |
                | Best Pace | The best pace (minutes per kilometer) during the activity | minutes per kilometer |
                | Total Ascent | The total ascent in meters during the activity | meters |
                | Total Descent | The total descent in meters during the activity | meters |
                | Avg Stride Length | The average stride length in meters during the activity | meters |
                | Avg Vertical Ratio | The average vertical ratio during the activity | - |
                | Avg Vertical Oscillation | The average vertical oscillation during the activity | meters |
                | Avg Ground Contact Time | The average ground contact time in milliseconds during the activity | milliseconds |
                | Training Stress Score® | The training stress score of the activity | - |
                | Avg Power | The average power output during the activity | watts |
                | Max Power | The maximum power output during the activity | watts |
                | Grit | The grit of the activity | - |
                | Flow | The flow of the activity | - |
                | Avg. Swolf | The average swolf score during the activity | - |
                | Avg Stroke Rate | The average stroke rate during the activity | strokes per minute |
                | Total Reps | The total number of repetitions during the activity | - |
                | Dive Time | The total dive time during the activity | seconds |
                | Min Temp | The minimum temperature during the activity | degrees Celsius |
                | Surface Interval | The total surface interval time during the activity | seconds |
                | Decompression | The total decompression time during the activity | seconds |
                | Best Lap Time | The best lap time during the activity | seconds |
                | Number of Laps | The number of laps during the activity | - |
                | Max Temp | The maximum temperature during the activity | degrees Celsius |
                | Moving Time | The time spent moving during the activity | seconds |
                | Elapsed Time | The total elapsed time of the activity | seconds |
                | Min Elevation | The minimum elevation during the activity | meters |
                | Max Elevation | The maximum elevation during the activity | meters |
                ''')
    return table

@st.experimental_memo
def get_chart(df,parameter):
    """
    Create simple line plot with Plotly of several parameters
    """

    # Define line plot
    fig = px.line(df, x="Date", y=parameter,
                      hover_data={"Date": "|%B %d, %Y"},
                      color_discrete_sequence=['#21b0fe'],
                      title=F'Time Series of {parameter}')
    
    fig.update_xaxes(rangeslider_visible=True)

    return fig

def monthly_average_line_charts(df, date_column, parameter_columns):
    """
    Display monthly average line charts for the specified parameters over time.

    Args:
        df (pandas.DataFrame): Input DataFrame containing the data.
        date_column (str): Name of the column containing the dates.
        parameter_columns (list): List of column names for which monthly averages will be plotted.
    """
    try:
        # Set the date_column as the DataFrame index
        df1 = df[['Date','Distance', 'Calories', 'Max HR', 'Avg HR', 'Avg Pace', 'Aerobic TE']].set_index(date_column)
        
        # Resample the DataFrame at a monthly frequency and calculate the monthly average for each parameter
        df_monthly_avg = df1.resample('M').mean()
        df_monthly_avg['Date_Col'] = df_monthly_avg.index.values
        

        # Plot with plotly
        tabo, tab1, tab2, tab3, tab4, tab5 = st.tabs(parameter_columns)
        with tab0:
            st.plotly_chart(get_chart(df1,parameter_columns[0]))
        with tab1:
            st.plotly_chart(get_chart(df1,parameter_columns[1]))
        with tab2:
            st.plotly_chart(get_chart(df1,parameter_columns[2]))        
        with tab3:
            st.plotly_chart(get_chart(df1,parameter_columns[3]))
        with tab4:
            st.plotly_chart(get_chart(df1,parameter_columns[4]))
        with tab5:
            st.plotly_chart(get_chart(df1,parameter_columns[5]))                

    except Exception as e:
        st.error("An error occurred while processing the data:")
        st.error(str(e))


def plot_interval_statistics(df, column_name, interval, calculation):
    """
    Plot the specified statistic (average, sum, maximum, or minimum) of a column from a DataFrame
    based on the chosen time interval as a bar graph.

    Args:
        df (pandas.DataFrame): Input DataFrame containing the data.
        column_name (str): Name of the column to calculate the statistic and plot.
        interval (str): Time interval for calculating the statistic.
                        Options: 'W' for weekly, 'M' for monthly, '6M' for semi-annual, 'A' for annual, etc.
        calculation (str): Statistic to calculate. Options: 'average', 'sum', 'maximum', 'minimum'.
    """
    try:
        # Select the appropriate aggregation function based on the calculation
        if calculation == 'average':
            aggregation_func = 'mean'
        elif calculation == 'sum':
            aggregation_func = 'sum'
        elif calculation == 'maximum':
            aggregation_func = 'max'
        elif calculation == 'minimum':
            aggregation_func = 'min'
        else:
            raise ValueError("Invalid calculation. Choose 'average', 'sum', 'maximum', or 'minimum'.")

        # Calculate the specified statistic based on the chosen interval
        statistic_by_interval = df.groupby(df['Date'].dt.to_period(interval))[column_name].agg(aggregation_func)

        # Convert index to strings for plotting
        statistic_by_interval.index = statistic_by_interval.index.astype(str)

        # Calculate the running average of five samples
        running_average = statistic_by_interval.rolling(window=5, min_periods=1).mean()

        # Create the bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(statistic_by_interval.index, statistic_by_interval.values, edgecolor='white', label='Original Data')
        ax.plot(statistic_by_interval.index, running_average, color='red', label='Running Average (5 samples)',lw=3)

        # Set plot labels and title
        ax.set_xlabel('Time Interval')
        ax.set_ylabel(calculation.capitalize() + ' ' + column_name)
        ax.set_title(calculation.capitalize() + ' ' + column_name + ' by ' + interval)

        # Configure x-axis tick labels
        ax.xaxis.set_major_locator(MaxNLocator(nbins='auto', min_n_ticks=5, integer=True))
        plt.xticks(rotation=45, ha='right')

        # Display the legend
        ax.legend()

        # Display the plot in Streamlit
        st.pyplot(fig)
   
    except Exception as e:
       st.error(f"An error occurred while plotting interval statistics: {e}")

def plot_distribution_with_average(df, column_name):
    """
    Plot a bar plot of the distribution of a column from a DataFrame
    with a black dashed line indicating the average value.

    Args:
        df (pandas.DataFrame): Input DataFrame containing the data.
        column_name (str): Name of the column to plot the distribution.
    """
    try:
        # Calculate the average value
        average_value = df[column_name].mean()
    
        # Create the bar plot of the distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df[column_name].dropna(), bins=20, edgecolor='black')
    
        # Add a black dashed line indicating the average value
        ax.axvline(average_value, color='black', linestyle='--', linewidth=1.5)
    
        # Set plot labels and title
        ax.set_xlabel(column_name)
        ax.set_ylabel('Count')
        ax.set_title('Distribution of ' + column_name)
    
        # Add a legend
        ax.legend([f'Average={round(average_value,2)}'])
    
        # Display the plot in Streamlit
        st.pyplot(fig)
    
    except Exception as e:
        st.error(f"An error occurred while plotting distribution with average: {e}")
    
def compare_day_night_performance(df, parameter, night_threshold="19:00"):
    """
    Compare the performance of a parameter between day and night using a t-test
    and generate boxplots to visualize the parameter distribution.

    Args:
        df (pandas.DataFrame): Input DataFrame containing the data.
        parameter (str): Name of the parameter to compare.
        night_threshold (str): Time threshold to determine day/night categorization (format: 'HH:MM').
    """
    try:
        # Convert 'Date' column to datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Categorize data into day and night based on the threshold
        df['Day/Night'] = df['Date'].apply(lambda x: 'Night' if x.time() >= pd.to_datetime(night_threshold).time() else 'Day')
        day_data = df[df['Day/Night'] == 'Day'][parameter]
        night_data = df[df['Day/Night'] == 'Night'][parameter]

        # Perform t-test
        t_statistic, p_value = stats.ttest_ind(day_data, night_data)

        # plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = sns.boxplot(x=df['Day/Night'], y=df[parameter], showfliers=False)

        # Set plot labels and title
        ax.set_xlabel('Day/Night')
        ax.set_ylabel(parameter)
        ax.set_title('Comparison of ' + parameter + ' between Day and Night')

        # Add t-test results to the plot
        plt.text(0.5, 0.9, f"P-Value: {p_value:.4f}", ha='center', va='center', transform=plt.gca().transAxes)
        # Add mean values to the plot
        day_mean = day_data.mean()
        night_mean = night_data.mean()
        ax.text(0.1, day_mean, f"{day_mean:.2f}", ha='right', va='center', fontweight='bold')
        ax.text(0.9, night_mean, f"{night_mean:.2f}", ha='left', va='center', fontweight='bold')

        # Configure x-axis tick labels
        ax.xaxis.set_major_locator(MaxNLocator(nbins='auto', min_n_ticks=5, integer=True))
        plt.xticks(rotation=45, ha='right')

        # Display the plot in Streamlit
        st.pyplot(fig)

        # Display the comparison result
        if p_value < 0.05:
            st.write("There is a significant difference between day and night performance.")
        else:
            st.write("There is no significant difference between day and night performance.")

    except Exception as e:
        st.error(f"An error occurred while comparing day-night performance: {e}")

def check_heart_rate_normal(average_rate, max_rate, age, gender):
        
    try:      
        data = {'Age': [20, 20, 30, 30, 35, 35, 40, 40, 45, 45, 50, 50, 55, 55, 60, 60, 65, 65, 70, 70],
            'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
            'Normal Heart Rate (bpm)': ['60-100', '60-100', '60-100', '60-100', '60-100', '60-100', '60-100', '60-100', '60-100', '60-100', '60-100', '60-100', '60-100', '60-100', '60-100', '60-100', '60-100', '60-100', '60-100', '60-100'],
            'Average Heart Rate (bpm)': ['100-170', '100-170', '95-162', '95-162', '93-157', '93-157', '90-153', '90-153', '88-149', '88-149', '85-145', '85-145', '83-140', '83-140', '80-136', '80-136', '78-132', '78-132', '75-128', '75-128'],
            'Maximum Heart Rate (bpm)': ['200', '200', '190', '190', '185', '185', '180', '180', '175', '175', '170', '170', '165', '165', '160', '160', '155', '155', '150', '150']}
     
        df = pd.DataFrame(data)
        df = df[['Age','Gender','Average Heart Rate (bpm)','Maximum Heart Rate (bpm)']]
        # Filter the DataFrame based on age and gender
        filtered_df = df[(df['Age'] == age) & (df['Gender'] == gender)]
        
        if filtered_df.empty:
            st.write("No normal values found for the given age and gender.")
            return
        
        # Get the normal range values
        normal_average_range = filtered_df['Average Heart Rate (bpm)'].iloc[0]
        normal_max_range = filtered_df['Maximum Heart Rate (bpm)'].iloc[0]
        
        # Extract the minimum and maximum values from the string range
        normal_average_min, normal_average_max = map(int, normal_average_range.split('-'))
        
        # Create a dictionary for the results
        results = {
            'Parameter': [ 'Average Heart Rate', 'Maximum Heart Rate'],
            'Actual': [ average_rate, max_rate],
            'Recommended': [ normal_average_range, normal_max_range]
        }
    
        # Display the results in a table
        st.table(pd.DataFrame(results))
    except Exception as e:
        st.error(f"An error occurred while checking heart rate normal: {e}")
       

def analyse_Cadence_Stride(df):
    try:
        # Calculate running speed
        df['Running Speed (m/min)'] = df['Avg Run Cadence'] * df['Avg Stride Length']

        # Statistics Summary
        st.write("Statistics Summary:")
        statistics_summary = df[['Avg Run Cadence', 'Avg Stride Length']].agg(['mean', 'std',]).round(2)
        st.table(statistics_summary.T)

        # Efficiency and Endurance Recommendations
        st.subheader("Efficiency and Endurance Recommendations:")
        if df['Avg Run Cadence'].mean() > 180:
            st.write("- Focus on reducing cadence slightly for better running economy.")
        else:
            st.write("- Maintain or increase cadence around 180 steps per minute for better running economy and reduced injury risk.")

        # Injury Risk Analysis
        st.markdown('''**Injury Risk Analysis**:''')
        if df['Avg Stride Length'].std() > 0.05:
            st.write("- Work on maintaining a more consistent stride length to reduce injury risk associated with variations in form.")
        else:
            st.write("- Continue maintaining a consistent stride length to reduce injury risk.")

        # Pacing Strategy Recommendations
        st.markdown('''**Pacing Strategy Recommendations**:''')
        st.write("- During uphill sections, maintain a higher cadence with shorter strides.")
        st.write("- During downhill sections, utilize longer strides with slightly reduced cadence.")

        # Section 3: Individual Variability and References
        st.markdown('''**Note**: Optimal cadence and stride length can vary based on individual factors such as height, leg length, and running style.''')
        st.write("References:")
        st.write("1. [Finding Your Optimal Running Cadence](https://www.trainingpeaks.com/blog/finding-your-perfect-run-cadence/)")
        st.write("2. [How to Find Your Ideal Running Cadence](https://www.outsideonline.com/health/running/running-cadence-new-research/)")
        st.write("3. [What Is A Good Running Cadence? - Marathon Handbook](https://marathonhandbook.com/running-cadence/)")
        st.write("4. [What is a good cadence for running – and how to find it? - Advnture](https://www.advnture.com/features/good-cadence-for-running)")

    except Exception as e:
        st.error("An error occurred. Please check your data and try again.")
        st.error(f"Error details: {e}")
        
def analyze_activity_duration(df, duration_column):
    """
    Analyze the distribution of activity durations to identify patterns in workout length and frequency.

    Args:
        df (pandas.DataFrame): Input DataFrame containing the data.
        duration_column (str): Name of the column representing activity durations in the format 'hh:mm:ss'.
    """
    try:
        # Create a new DataFrame to store the converted durations in total minutes
        df_minutes = df.copy()
        df_minutes[duration_column] = pd.to_timedelta(df_minutes[duration_column])
        df_minutes[duration_column] = df_minutes[duration_column].dt.total_seconds() / 60

        # Calculate the mean and standard deviation
        mean_duration = df_minutes[duration_column].mean()

        # Create a histogram using Plotly
        fig = px.histogram(df_minutes, x=duration_column, nbins=20,
                           labels={'x': 'Activity Duration (minutes)', 'y': 'Frequency'},
                           title='Distribution of Activity Durations',)
        
        # Add a vertical line for the mean duration
        fig.add_vline(x=mean_duration, line_dash="dash", line_color="#fe218b",
                      annotation_text=f"Mean: {mean_duration:.2f} min", annotation_position="top")

        # Set the color of the histogram
        fig.update_traces(marker_color='#21b0fe')
        # Set the layout of the figure
        fig.update_layout(showlegend=False)

        # Display the histogram using st.plotly_chart
        st.plotly_chart(fig)
        
        # Calculate summary statistics using df_minutes
        avg_duration = df_minutes[duration_column].mean()
        median_duration = df_minutes[duration_column].median()
        max_duration = df_minutes[duration_column].max()
        min_duration = df_minutes[duration_column].min()

        # Display summary statistics
        st.subheader('Summary Statistics')
        st.write(f"Average Duration: {avg_duration:.2f} minutes")
        st.write(f"Median Duration: {median_duration:.2f} minutes")
        st.write(f"Maximum Duration: {max_duration:.2f} minutes")
        st.write(f"Minimum Duration: {min_duration:.2f} minutes")

    except Exception as e:
        st.error("An error occurred while processing the data:")
        st.error(str(e)) 
     
def analyze_temperature_impact(df, max_temp_column, min_temp_column, pace_column, heart_rate_column, max_heart_rate_column):
    """
    Analyze how temperature affects performance, comparing metrics like pace, heart rate, and power output
    in different temperature ranges.

    Args:
        df (pandas.DataFrame): Input DataFrame containing the data.
        max_temp_column (str): Name of the column representing maximum temperature (°C).
        min_temp_column (str): Name of the column representing minimum temperature (°C).
        pace_column (str): Name of the column representing pace (minutes per kilometer).
        heart_rate_column (str): Name of the column representing heart rate (beats per minute).
        power_column (str): Name of the column representing power output (watts).
    """
    try:
        # Define temperature ranges
        cold_range = (0, 10)
        moderate_range = (10, 20)
        hot_range = (20, df[max_temp_column].max())

        # Create subsets based on maximum and minimum temperature ranges
        cold_subset = df[(df[max_temp_column] >= cold_range[0]) & (df[min_temp_column] < cold_range[1])]
        moderate_subset = df[(df[max_temp_column] >= moderate_range[0]) & (df[min_temp_column] < moderate_range[1])]
        hot_subset = df[df[max_temp_column] >= hot_range[0]]

        # Calculate average metrics for each temperature range
        cold_pace_avg = cold_subset[pace_column].mean()
        cold_heart_rate_avg = cold_subset[heart_rate_column].mean()
        cold_heart_rate_max = cold_subset[max_heart_rate_column].mean()

        moderate_pace_avg = moderate_subset[pace_column].mean()
        moderate_heart_rate_avg = moderate_subset[heart_rate_column].mean()
        moderate_heart_rate_max = moderate_subset[max_heart_rate_column].mean()

        hot_pace_avg = hot_subset[pace_column].mean()
        hot_heart_rate_avg = hot_subset[heart_rate_column].mean()
        hot_heart_rate_max = hot_subset[max_heart_rate_column].mean()

        # Display the results in a Streamlit table
        st.subheader("Temperature Impact on Performance")
        st.write("This analysis compares performance metrics (pace and heart rate) across different temperature ranges.")

        # Prepare the data for the table
        table_data = {
            'Temperature Range': ['Cold: 0°C - 10°C', 'Moderate: 10°C - 20°C', 'Hot: 20°C and above'],
            'Average Pace (minutes per kilometer)': [cold_pace_avg, moderate_pace_avg, hot_pace_avg],
            'Average Mean Heart Rate (beats per minute)': [cold_heart_rate_avg, moderate_heart_rate_avg, hot_heart_rate_avg],
            'Average Max Heart Rate (beats per minute)': [cold_heart_rate_max, moderate_heart_rate_max, hot_heart_rate_max]
        }

        # Display the table
        st.table(pd.DataFrame(table_data))

    except Exception as e:
        st.error("An error occurred while processing the data:")
        st.error(str(e))

def plot_correlation_heatmap(df, temperature_columns):
    """
    Calculate Spearman correlations and plot a heatmap of correlations > 0.5 for the temperature columns and numeric columns.

    Args:
        df (pandas.DataFrame): Input DataFrame containing the data.
        temperature_columns (list): List of temperature column names.
    """
    try:
        # Filter numeric columns
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

        # Calculate Spearman correlations
        correlations = df[numeric_columns].corr(method='spearman')

        # Filter correlations > 0.5 for the temperature columns
        high_correlations = correlations[temperature_columns].abs().max() > 0.5

        # Create a DataFrame containing selected columns and the columns with high correlations
        selected_columns = temperature_columns + high_correlations.index.tolist()
        selected_corr = correlations[selected_columns]

        # Drop duplicate columns
        selected_corr = selected_corr.loc[:, ~selected_corr.columns.duplicated()]

        # Filter correlations with absolute value > 0.5
        selected_corr = selected_corr[(selected_corr.abs() > 0.5).any(axis=1)]

        # Check if only temperature columns are selected
        if selected_corr.shape[1] == len(temperature_columns):
            st.write("There is no significant correlation between performance and temperature.")
        else:
            # Plot heatmap if there are significant correlations
            if not selected_corr.empty:
                fig, ax = plt.subplots(figsize=(15, 15))
                sns.heatmap(selected_corr, annot=True, cmap='coolwarm_r', vmin=-1, vmax=1, fmt=".2f", ax=ax)
                ax.set_title(f"Spearman Correlation Heatmap for Temperature (Correlations > 0.5)")

                # Display the plot in Streamlit
                st.pyplot(fig)
            else:
                st.write("There is no significant correlation between performance and temperature.")

    except Exception as e:
        st.error("An error occurred while processing the data:")
        st.error(str(e))
        
def highlight_personal_records(df,val, record_column):
    """
    Highlight personal records in the DataFrame.

    Args:
        val: Cell value in the DataFrame.
        record_column (str): Name of the column to check for personal records.

    Returns:
        str: CSS style for highlighting the cell.
    """
    if val == df[record_column].max():
        return 'background-color: yellow'
    else:
        return ''

def identify_personal_records(df):
    """
    Identify and highlight the user's personal records for metrics.

    Args:
        df (pandas.DataFrame): Input DataFrame containing the running data.

    Returns:
        None
    """
    try:
        # Find personal records for metrics
        fastest_pace = df['Avg Pace'].min()
        max_heart_rate = df['Max HR'].max()
        longest_distance = df['Distance'].max()
        highest_power_output = df['Max Power'].max()
        total_ascent = df['Total Ascent'].max()  # Replace 'Elevation' with 'Total Ascent'

        # Create a copy of the DataFrame with personal records highlighted
        df_highlighted = df.style.applymap(lambda x: highlight_personal_records(df,x, 'Avg Pace'), subset=['Avg Pace']) \
                               .applymap(lambda x: highlight_personal_records(df,x, 'Max HR'), subset=['Max HR']) \
                               .applymap(lambda x: highlight_personal_records(df,x, 'Distance'), subset=['Distance']) \
                               .applymap(lambda x: highlight_personal_records(df,x, 'Max Power'), subset=['Max Power']) \
                               .applymap(lambda x: highlight_personal_records(df,x, 'Total Ascent'), subset=['Total Ascent'])  # Highlight 'Total Ascent'

        # Display the user's personal records in a Streamlit table
        st.subheader("Performance Records - Personal Bests")
        st.write("Here are your personal records for various metrics:")

        # Prepare the data for the table
        table_data = {
            'Metric': ['Fastest Pace', 'Maximum Heart Rate', 'Longest Distance', 'Highest Power Output', 'Total Ascent'],  # Add 'Total Ascent' to the table
            'Personal Record': [fastest_pace, max_heart_rate, longest_distance, highest_power_output, total_ascent]  # Add 'total_ascent' to the list
        }

        # Display the table with personal records highlighted
        st.table(pd.DataFrame(table_data))

        # Display the full DataFrame with personal records highlighted
        st.subheader("Check Your Other Records by Clicking on the Column")
        st.dataframe(df_highlighted)

    except Exception as e:
        st.error("An error occurred while processing the data:")
        st.error(str(e))


# Draft

def find_highest_performance_hour(df, parameter):
    """
    Find the hour when the distance from the overall average is the highest and return the hour and value.
    Also, plot a bar plot of the parameter and hours.

    Args:
        df (pandas.DataFrame): Input DataFrame containing the running data.
        parameter (str): Name of the parameter to analyze.

    Returns:
        highest_distance_hour (int): Hour when the distance from the overall average is the highest.
        highest_distance_value (float): Average value of the parameter for the hour.
        lowest_distance_hour (int): Hour when the distance from the overall average is the lowest.
        lowest_distance_value (float): Average value of the parameter for the hour.
    """
    try:
        # Create a new DataFrame with the average of the parameter according to each hour
        hourly_avg_df = df.groupby('Hour')[parameter].mean().reset_index()
        
        # Calculate the overall average of the parameter
        overall_avg = df[parameter].mean()
        
        # Calculate the distance from the overall average for each hour
        hourly_avg_df['Distance from Overall Average'] = abs(hourly_avg_df[parameter] - overall_avg)
        
        # Find the hour with the highest distance from the overall average
        highest_distance_hour = hourly_avg_df.loc[hourly_avg_df['Distance from Overall Average'].idxmax(), 'Hour']
        highest_distance_value = hourly_avg_df.loc[hourly_avg_df['Distance from Overall Average'].idxmax(), parameter]
        
        # Find the hour with the lowest distance from the overall average
        lowest_distance_hour = hourly_avg_df.loc[hourly_avg_df['Distance from Overall Average'].idxmin(), 'Hour']
        lowest_distance_value = hourly_avg_df.loc[hourly_avg_df['Distance from Overall Average'].idxmin(), parameter]
        
        # Display the results
        st.write(f"The hour when the distance from the overall average is the highest is: {highest_distance_hour}")
        st.write(f"The average value for this hour is: {highest_distance_value} while the overall average is: {overall_avg}")
        st.write(f"The hour when the distance from the overall average is the lowest is: {lowest_distance_hour}")
        st.write(f"The average value for this hour is: {lowest_distance_value} while the overall average is: {overall_avg}")
        
        # Plot a bar plot of the parameter and hours
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(hourly_avg_df['Hour'], hourly_avg_df[parameter])
        ax.set_xlabel('Hour')
        ax.set_ylabel(parameter)
        ax.set_title('Average ' + parameter + ' by Hour')
        
        # Display the plot in Streamlit
        st.pyplot(fig)
        
        return highest_distance_hour, highest_distance_value, lowest_distance_hour, lowest_distance_value    
    
    except Exception as e:
        st.error(f"An error occurred while finding the highest performance hour: {e}")
        return None, None, None, None
