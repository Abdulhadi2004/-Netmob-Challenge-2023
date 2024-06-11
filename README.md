# CSCI-322 Project: Data Analysis for Lille and Nancy

## Project Overview

This project involves data analysis for two cities, Lille and Nancy, across four applications: Amazon Web Services, Apple Web Services, Google Web Services, and Microsoft Web Services. The analysis includes data visualization, traffic pattern analysis, and usage comparisons.

## Team Members

1. Abdulhay Ibrahim - 211000768
2. Mohamed Ashraf Mohamed - 211001445
3. Khaled Zaki - 211001783
4. Mohamed Adel - 202001764
5. Mohamed Osman - 211001922

## Table of Contents

1. [Import Libraries and Define City Dimensions](#import-libraries-and-define-city-dimensions)
2. [Reading CSV and GeoJSON Files](#reading-csv-and-geojson-files)
3. [Displaying Data Information](#displaying-data-information)
4. [Creating 3D Arrays for Traffic Data](#creating-3d-arrays-for-traffic-data)
5. [Daily Usage Analysis](#daily-usage-analysis)
6. [Total Usage Calculation](#total-usage-calculation)
7. [Heatmaps](#heatmaps)
8. [Weekdays vs. Weekends Analysis](#weekdays-vs-weekends-analysis)
9. [AM vs. PM Traffic Analysis](#am-vs-pm-traffic-analysis)
10. [Correlation Matrices](#correlation-matrices)
11. [Traffic Time by Hour](#traffic-time-by-hour)
12. [Daily Traffic Maps](#daily-traffic-maps)

## Import Libraries and Define City Dimensions

The following libraries are used in this project:

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import json
import matplotlib.cm as cm
import matplotlib.colors as colrs
import geopandas as gpd
from descartes.patch import PolygonPatch
from shapely.geometry import shape as Shape
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

```

City dimensions are defined as follows:

```python
city_dims = {
    'Bordeaux': (334, 342),
    'Clermont-Ferrand': (208, 268),
    'Dijon': (195, 234),
    'Grenoble': (409, 251),
    'Lille': (330, 342),
    'Lyon': (426, 287),
    'Mans': (228, 246),
    'Marseille': (211, 210),
    'Metz': (226, 269),
    'Montpellier': (334, 327),
    'Nancy': (151, 165),
    'Nantes': (277, 425),
    'Nice': (150, 214),
    'Orleans': (282, 256),
    'Paris': (409, 346),
    'Rennes': (423, 370),
    'Saint-Etienne': (305, 501),
    'Strasbourg': (296, 258),
    'Toulouse': (280, 347),
    'Tours': (251, 270)
}
```

# Reading CSV and GeoJSON Files

CSV and GeoJSON files are read into Pandas DataFrames and JSON objects respectively. Below is an example for reading Lille Amazon data:

```python
column_names = ['tile_id', '00:00', '00:15', '00:30', '00:45', '01:00', '01:15', '01:30', '01:45',
                '02:00', '02:15', '02:30', '02:45', '03:00', '03:15', '03:30', '03:45',
                '04:00', '04:15', '04:30', '04:45', '05:00', '05:15', '05:30', '05:45',
                '06:00', '06:15', '06:30', '06:45', '07:00', '07:15', '07:30', '07:45',
                '08:00', '08:15', '08:30', '08:45', '09:00', '09:15', '09:30', '09:45',
                '10:00', '10:15', '10:30', '10:45', '11:00', '11:15', '11:30', '11:45',
                '12:00', '12:15', '12:30', '12:45', '13:00', '13:15', '13:30', '13:45',
                '14:00', '14:15', '14:30', '14:45', '15:00', '15:15', '15:30', '15:45',
                '16:00', '16:15', '16:30', '16:45', '17:00', '17:15', '17:30', '17:45',
                '18:00', '18:15', '18:30', '18:45', '19:00', '19:15', '19:30', '19:45',
                '20:00', '20:15', '20:30', '20:45', '21:00', '21:15', '21:30', '21:45',
                '22:00', '22:15', '22:30', '22:45', '23:00', '23:15', '23:30', '23:45', 'Day']

lille_amazon = pd.read_csv('Lille_Amazon.csv', sep=',', names=column_names, skiprows=[0])

```

# Displaying Data Information

You can display basic information about the DataFrame using:

```python
lille_amazon.info()
```

# Creating 3D Arrays for Traffic Data

Traffic data is stored in 3D arrays where the first dimension is time, and the second and third dimensions are spatial dimensions:

```python
times = ['00:00', '00:15', '00:30', '00:45', '01:00', '01:15', '01:30', '01:45',
         '02:00', '02:15', '02:30', '02:45', '03:00', '03:15', '03:30', '03:45',
         '04:00', '04:15', '04:30', '04:45', '05:00', '05:15', '05:30', '05:45',
         '06:00', '06:15', '06:30', '06:45', '07:00', '07:15', '07:30', '07:45',
         '08:00', '08:15', '08:30', '08:45', '09:00', '09:15', '09:30', '09:45',
         '10:00', '10:15', '10:30', '10:45', '11:00', '11:15', '11:30', '11:45',
         '12:00', '12:15', '12:30', '12:45', '13:00', '13:15', '13:30', '13:45',
         '14:00', '14:15', '14:30', '14:45', '15:00', '15:15', '15:30', '15:45',
         '16:00', '16:15', '16:30', '16:45', '17:00', '17:15', '17:30', '17:45',
         '18:00', '18:15', '18:30', '18:45', '19:00', '19:15', '19:30', '19:45',
         '20:00', '20:15', '20:30', '20:45', '21:00', '21:15', '21:30', '21:45',
         '22:00', '22:15', '22:30', '22:45', '23:00', '23:15', '23:30', '23:45']

n_rows_lille, n_cols_lille = city_dims['Lille']
n_rows_nancy, n_cols_nancy = city_dims['Nancy']

lille_amazon_traffic = np.zeros((len(times), n_rows_lille, n_cols_lille))
nancy_amazon_traffic = np.zeros((len(times), n_rows_nancy, n_cols_nancy))

def fill_traffic_data(df, traffic_data, n_cols):
    for _, row in df.iterrows():
        tile_id = row['tile_id']
        row_index = int(tile_id // n_cols)
        col_index = int(tile_id % n_cols)
        traffic_values = np.array(row[times])
        traffic_data[:, row_index, col_index] = traffic_values

fill_traffic_data(lille_amazon, lille_amazon_traffic, n_cols_lille)

```

# Daily Usage Analysis

Daily usage is plotted to analyze the usage trends:

```python
def plot_daily_usage(data, city_name, app_name):
    daily_usage = data.groupby('Day').sum().sum(axis=1)
    plt.figure(figsize=(12, 6))
    plt.plot(daily_usage.index, daily_usage.values, label=f'{app_name} in {city_name}')
    plt.xlabel('Day')
    plt.ylabel('Total Usage')
    plt.title(f'Daily Usage of {app_name} in {city_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_daily_usage(lille_amazon, 'Lille', 'Amazon')

```

# Total Usage Calculation

Total usage is calculated and compared between Lille and Nancy:

```python
datasets_lille = [lille_amazon, lille_apple, lille_google, lille_microsoft]
datasets_nancy = [nancy_amazon, nancy_apple, nancy_google, nancy_microsoft]
applications = ['Amazon', 'Apple', 'Google', 'Microsoft']

def total_usage(data):
    return data.iloc[:, 1:-1].sum().sum()

total_usage_lille = [total_usage(data) for data in datasets_lille]
total_usage_nancy = [total_usage(data) for data in datasets_nancy]

fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.35
index = np.arange(len(applications))

bar1 = ax.bar(index, total_usage_lille, bar_width, label='Lille')
bar2 = ax.bar(index + bar_width, total_usage_nancy, bar_width, label='Nancy')

ax.set_xlabel('Applications')
ax.set_ylabel('Total Usage')
ax.set_title('Total Usage of Applications in Lille and Nancy')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(applications)
ax.legend()

plt.show()

```

# Heatmaps

Heatmaps are created to visualize the traffic data:

```python
def plot_heatmap(data, city_name, app_name):
    heatmap_data = data.groupby('Day').sum().drop(columns=['tile_id'])
    sns.heatmap(heatmap_data, cmap='viridis')
    plt.title(f'Heatmap of {app_name} Traffic in {city_name}')
    plt.xlabel('Time of Day')
    plt.ylabel('Day')
    plt.show()

plot_heatmap(lille_amazon, 'Lille', 'Amazon')

```

# Weekdays vs. Weekends Analysis

Traffic data is analyzed for weekdays vs. weekends:

```python
def label_weekday_or_weekend(day_str):
    date_obj = datetime.datetime.strptime(day_str, '%Y%m%d')
    if date_obj.weekday() < 5:
        return 'Weekday'
    else:
        return 'Weekend'

for df in datasets_lille + datasets_nancy:
    df['Day_Type'] = df['Day'].apply(label_weekday_or_weekend)

def aggregate_traffic_by_day_type(data):
    weekday_data = data[data['Day_Type'] == 'Weekday'].iloc[:, 1:-2].sum().sum()
    weekend_data = data[data['Day_Type'] == 'Weekend'].iloc[:, 1:-2].sum().sum()
    return weekday_data, weekend_data

weekday_weekend_lille = [aggregate_traffic_by_day_type(df) for df in datasets_lille]
weekday_weekend_nancy = [aggregate_traffic_by_day_type(df) for df in datasets_nancy]

labels = ['Amazon', 'Apple', 'Google', 'Microsoft']
x = np.arange(len(labels))  # label locations
width = 0.35  # width of the bars

fig, ax = plt.subplots(figsize=(12, 8))
weekday_lille, weekend_lille = zip(*weekday_weekend_lille)
weekday_nancy, weekend_nancy = zip(*weekday_weekend_nancy)

bar1 = ax.bar(x - width/2, weekday_lille, width, label='Weekday - Lille')
bar2 = ax.bar(x + width/2, weekend_lille, width, label='Weekend - Lille')
bar3 = ax.bar(x - width/2 + len(labels) + 1, weekday_nancy, width, label='Weekday - Nancy')
bar4 = ax.bar(x + width/2 + len(labels) + 1, weekend_nancy, width, label='Weekend - Nancy')

ax.set_xlabel('Applications')
ax.set_ylabel('Total Traffic')
ax.set_title('Total Traffic on Weekdays and Weekends')
ax.set_xticks(np.concatenate((x - width/2, x + width/2 + len(labels) + 1)))
ax.set_xticklabels(labels * 2)
ax.legend()

plt.show()

```

# AM vs. PM Traffic Analysis

Traffic data is analyzed for AM vs. PM periods:

```python
# Define AM and PM periods
am_hours = [f"{str(hour).zfill(2)}:{str(minute).zfill(2)}" for hour in range(12) for minute in range(0, 60, 15)]
pm_hours = [f"{str(hour).zfill(2)}:{str(minute).zfill(2)}" for hour in range(12, 24) for minute in range(0, 60, 15)]

def aggregate_am_pm_traffic_total(data):
    am_traffic_total = data[am_hours].sum().sum()
    pm_traffic_total = data[pm_hours].sum().sum()
    return am_traffic_total, pm_traffic_total

am_pm_lille_amazon = aggregate_am_pm_traffic_total(lille_amazon)
am_pm_nancy_amazon = aggregate_am_pm_traffic_total(nancy_amazon)

# Function to plot pie charts for AM and PM traffic
def plot_pie_chart(am_pm_data, city_name, app_name):
    labels = ['AM', 'PM']
    sizes = am_pm_data
    colors = ['#ff9999', '#66b3ff']
    explode = (0.1, 0)  # explode 1st slice

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', shadow=True, startangle=90)
    ax.axis('equal')
    plt.title(f'{app_name} Traffic in {city_name} (AM vs PM)')
    plt.show()

plot_pie_chart(am_pm_lille_amazon, 'Lille', 'Amazon')

```

# Correlation Matrices

Correlation matrices are calculated and visualized:

```python
def calculate_correlation(data_list):
    combined_data = pd.concat([df.iloc[:, 1:-2].sum() for df in data_list], axis=1)
    combined_data.columns = labels
    return combined_data.corr()

correlation_lille = calculate_correlation(datasets_lille)
correlation_nancy = calculate_correlation(datasets_nancy)

fig, ax = plt.subplots(1, 2, figsize=(16, 8))

sns.heatmap(correlation_lille, annot=True, cmap='coolwarm', ax=ax[0])
ax[0].set_title('Correlation Matrix - Lille')

sns.heatmap(correlation_nancy, annot=True, cmap='coolwarm', ax=ax[1])
ax[1].set_title('Correlation Matrix - Nancy')

plt.show()

```

# Traffic Time by Hour

Total traffic time series by hour is plotted:

```python
def plot_total_traffic_time_series_by_hour(city_traffic, app_name):
    hourly_traffic = [np.sum(city_traffic[hour*4:(hour+1)*4], axis=0) for hour in range(24)]
    median_traffic = [np.median(hour_traffic[hour_traffic > 0]) for hour_traffic in hourly_traffic]
    mean_traffic = [np.mean(hour_traffic[hour_traffic > 0]) for hour_traffic in hourly_traffic]
    hours = [f'{hour}:00' for hour in range(24)]

    fig = plt.figure(figsize=(12, 6))
    plt.plot(hours, median_traffic, linewidth=1, color='tab:orange', label='Median Traffic (valid tiles)')
    plt.plot(hours, mean_traffic, linewidth=1, color='tab:green', label='Mean Traffic (valid tiles)')
    plt.xlabel('Time')
    plt.ylabel(f'{app_name} Traffic DN')
    plt.xticks(rotation=45)
    plt.grid(axis='x', alpha=0.25)
    plt.xlim(hours[0], hours[-1])
    plt.legend(loc='upper right', ncol=1, bbox_to_anchor=(1, 1.1), fancybox=False, frameon=False)
    plt.show()

plot_total_traffic_time_series_by_hour(lille_amazon_traffic, 'Amazon - Lille')

```

# Daily Traffic Maps

Daily traffic maps are plotted for Google Web Services:

```python
cmap_traffic = plt.colormaps.get_cmap('Spectral_r').copy()
cmap_traffic.set_under('w', 0)
norm_traffic = colrs.LogNorm(vmin=1e0, vmax=1e7)

def ensure_numeric(df):
    for col in df.columns[1:-2]:  # Skip tile_id and Day columns
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

lille_google = ensure_numeric(lille_google)
nancy_google = ensure_numeric(nancy_google)

def plot_daily_traffic_maps(city_traffic, city_name, app_name, dates, n_rows, n_cols):
    for day in dates:
        daily_traffic = city_traffic[city_traffic['Day'] == day]
        daily_traffic_3d = np.zeros((len(times), n_rows, n_cols))

        for row in daily_traffic.itertuples():
            tile_id = row.tile_id
            row_index = int(tile_id // n_cols)
            col_index = int(tile_id % n_cols)
            daily_traffic_3d[:, row_index, col_index] = np.array(row[2:-2])

        fig, axs = plt.subplots(4, 6, figsize=(60, 40))
        axs = axs.flatten()

        for hour in range(24):
            ax = axs[hour]
            city_traffic_time = daily_traffic_3d[hour * 4]
            im = ax.imshow(city_traffic_time, origin='lower', cmap=cmap_traffic, norm=norm_traffic)
            ax.set_title(f'{str(hour).zfill(2)}:00', fontsize=30)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')

        ax = fig.add_axes([0.95, 0.25, 0.02, .5])
        sm = plt.cm.ScalarMappable(cmap=cmap_traffic, norm=norm_traffic)
        sm.set_array([])
        clb = plt.colorbar(sm, cax=ax, orientation='vertical')
        clb.set_label('Traffic DN', rotation=90, fontsize=40, labelpad=50)
        clb.ax.tick_params(labelsize=30)
        clb.ax.xaxis.set_ticks_position('default')

        fig.suptitle(f'{app_name} Traffic in {city_name} on {day}', fontsize=40)
        plt.show()

plot_daily_traffic_maps(lille_google, 'Lille', 'Google', dates_lille, n_rows_lille, n_cols_lille)
plot_daily_traffic_maps(nancy_google, 'Nancy', 'Google', dates_nancy, n_rows_nancy, n_cols_nancy)

```
