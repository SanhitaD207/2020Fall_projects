#!/usr/bin/env python
# coding: utf-8

# Team Members - Adarsh Agarwal (adarsha2), Sanhita Dhamdhere (sanhita2)

# References:
# - https://www.kite.com/python/answers/how-to-select-rows-by-multiple-label-conditions-with-pandas-loc-in-python
# - https://stackoverflow.com/questions/50375985/pandas-add-column-with-value-based-on-condition-based-on-other-columns
# - https://stackoverflow.com/questions/19384532/get-statistics-for-each-group-such-as-count-mean-etc-using-pandas-groupby
# - https://stackoverflow.com/questions/44111307/python-pandas-count-rows-based-on-column
# - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html
# - https://datatofish.com/line-chart-python-matplotlib/
# - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html
# - https://github.com/rahulrohri/final_project_2020Sp
# - https://studio.mapbox.com/
# - https://plotly.com/python/scattermapbox/


import pandas as pd
import numpy as np
from datetime import datetime, time
import matplotlib.pyplot as plt
import plotly.graph_objects as go


NYC_collision_crashes_file = "Motor_Vehicle_Collisions_-_Crashes.csv"
NYC_collision_persons_file = "Motor_Vehicle_Collisions_-_Person.csv"


def load_collision_data(crashes_file, persons_file):
    """
    This function is used to load the 2 data frames of motor vehicle collisions - crashes and persons data.
    The files need to be present in the same directory as this current PY file,
        or the complete file path needs to be sent as the argument to this function.

    :param crashes_file: File name for the crashes dataset
    :param persons_file: File name for the persons dataset
    :return: crashes and persons dataframes
    """
    if not crashes_file or not persons_file:
        raise Exception('Please provide a valid file name')

    crashes = pd.read_csv(crashes_file, low_memory=False)
    persons = pd.read_csv(persons_file, low_memory=False)

    return crashes, persons


def get_night_crashes(crashes):
    """
    This function is used to fetch all crashes that happen between 12 am to 5 am each day.
    :param crashes: Crashes dataset
    :return: Crashes between 12 am to 5 am
    """

    crashes['CRASH TIME'] = crashes['CRASH TIME'].apply(lambda x: datetime.strptime(x, "%H:%M").time())

    night_crash_data = crashes[(crashes['CRASH TIME'] < time(5, 0, 0))]

    return night_crash_data


def check_for_unsafe_speed(night_crash_data):
    night_crash_data = night_crash_data.assign(hasUnsafeSpeed=False)
    night_crash_data.loc[((night_crash_data['CONTRIBUTING FACTOR VEHICLE 1'] == 'Unsafe Speed') |
                          (night_crash_data['CONTRIBUTING FACTOR VEHICLE 2'] == 'Unsafe Speed') |
                          (night_crash_data['CONTRIBUTING FACTOR VEHICLE 3'] == 'Unsafe Speed') |
                          (night_crash_data['CONTRIBUTING FACTOR VEHICLE 4'] == 'Unsafe Speed') |
                          (night_crash_data['CONTRIBUTING FACTOR VEHICLE 5'] == 'Unsafe Speed')),
                         'hasUnsafeSpeed'] = True

    return night_crash_data


def calculate_percentage_of_speedy_collisions(night_crash_data):
    unsafe_speed_metrics = night_crash_data['hasUnsafeSpeed'].value_counts().to_frame()
    percentage_unsafe_speed_collisions = unsafe_speed_metrics.iloc[1] * 100 / night_crash_data.shape[0]

    return percentage_unsafe_speed_collisions


def calculate_invalid_collision_percentage(night_crash_data):
    unwanted_contributing_factors = ['1', '80', 'Unspecified']

    night_crash_data['isUnspecified'] = np.where((((
                                                       night_crash_data['CONTRIBUTING FACTOR VEHICLE 1'].isin(unwanted_contributing_factors)) |
                                                   (night_crash_data['CONTRIBUTING FACTOR VEHICLE 1'].isnull())) &
                                                  ((night_crash_data['CONTRIBUTING FACTOR VEHICLE 2'].isin(unwanted_contributing_factors)) |
                                                   (night_crash_data['CONTRIBUTING FACTOR VEHICLE 2'].isnull())) &
                                                  ((night_crash_data['CONTRIBUTING FACTOR VEHICLE 3'].isin(unwanted_contributing_factors)) |
                                                   (night_crash_data['CONTRIBUTING FACTOR VEHICLE 3'].isnull())) &
                                                  ((night_crash_data['CONTRIBUTING FACTOR VEHICLE 4'].isin(unwanted_contributing_factors)) |
                                                   (night_crash_data['CONTRIBUTING FACTOR VEHICLE 4'].isnull())) &
                                                  ((night_crash_data['CONTRIBUTING FACTOR VEHICLE 5'].isin(unwanted_contributing_factors)) |
                                                   (night_crash_data['CONTRIBUTING FACTOR VEHICLE 5'].isnull()))), True, False)

    invalid_night_crash_data_metrics = night_crash_data['isUnspecified'].value_counts().to_frame()

    percentage_invalid_collision_data = invalid_night_crash_data_metrics.iloc[1] * 100 / night_crash_data.shape[0]

    return percentage_invalid_collision_data


def get_merged_crashes_persons(crashes, persons):
    crashes_persons = pd.merge(crashes, persons, left_on='COLLISION_ID', right_on='COLLISION_ID', how='inner')
    crashes_persons.loc[:, 'CRASH_YEAR'] = crashes_persons['CRASH_DATE'].astype(np.str_).apply(lambda x: x.split('/')[-1])
    del crashes_persons['CRASH_DATE']
    del crashes_persons['CRASH_TIME']
    del crashes_persons['UNIQUE_ID']
    return crashes_persons


def get_crashes_persons_age_grouping_data(crashes_persons, columns):
    crashes_persons_age_grouping = crashes_persons[crashes_persons['POSITION_IN_VEHICLE'] == 'Driver'][columns]
    crashes_persons_age_grouping.loc[:, 'age16-25'] = np.where((crashes_persons_age_grouping['PERSON_AGE'] > 15) &
                                                               (crashes_persons_age_grouping['PERSON_AGE'] < 26), True, False)
    return crashes_persons_age_grouping


def calculate_percentage_drivers_between_ages_16_25(crashes_persons_age_grouping):
    df = crashes_persons_age_grouping['age16-25'].value_counts().to_frame()
    pct = df.iloc[1] * 100 / crashes_persons_age_grouping.shape[0]
    return pct


def get_nyc_population_data():
    # Source - https://worldpopulationreview.com/us-cities/new-york-city-ny-population

    nyc_population_data = {'Year': [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
                           'Population': [8348030, 8398740, 8437390, 8468180, 8475980, 8438270, 8398750, 8361040, 8323340],
                           }

    nyc_population = pd.DataFrame(nyc_population_data, columns=['Year', 'Population'])

    nyc_area = 300.4  # (in sq miles) Source - https://worldpopulationreview.com/us-cities/new-york-city-ny-population

    nyc_population['Population_Density'] = nyc_population['Population'].apply(lambda x: x / nyc_area)

    return nyc_population


def get_total_crashes_per_year(crashes):
    crashes_data = crashes.copy()
    crashes_total = crashes_data.groupby(['CRASH_YEAR'], sort=False).size().reset_index(name='Total_Crashes')
    crashes_total['CRASH_YEAR'] = crashes_total['CRASH_YEAR'].astype('int64')

    crashes_total = crashes_total.sort_values(by=['CRASH_YEAR'])

    return crashes_total


def calculate_crashes_per_capita(crashes_total, nyc_population):
    crashes_population = pd.merge(crashes_total, nyc_population, left_on='CRASH_YEAR', right_on='Year', how='inner')

    crashes_population.loc[:, 'Crashes_per_capita'] = crashes_population['Total_Crashes'] / crashes_population['Population']

    return crashes_population


def plot_crashes_per_capita_vs_year(crashes_population):
    plt.plot(crashes_population['Year'], crashes_population['Crashes_per_capita'], color='red', marker='o')
    plt.title('Crashes_per_Capita Vs Year for NYC')
    plt.xlabel('Year')
    plt.ylabel('Crashes_per_capita')
    plt.show()


def plot_crashes_per_capita_vs_population_density(crashes_population):
    plt.plot(crashes_population['Crashes_per_capita'], crashes_population['Population_Density'], color='red', marker='o')
    plt.title('Crashes_per_Capita Vs Population for NYC')
    plt.xlabel('Population Density')
    plt.ylabel('Crashes_per_capita')
    plt.show()


def prepare_crashes_data_for_maps(crashes, column_name, separator='/'):
    crashes.loc[:, 'CRASH_MONTH'] = crashes[column_name].astype(np.str_).apply(lambda x: int(x.split(separator)[0]))
    return crashes


def plot_crash_locations(mapbox_access_token, crashes, year, month, borough=''):
    if not 0 < month < 13:
        raise Exception('Please provide a valid month number (between 1 and 12)')
    if year < 2013:
        raise Exception('Year values above 2013 only')

    if not borough:
        df = crashes[(crashes['CRASH_YEAR'] == year) &
                     (crashes['CRASH_MONTH'] == month)]
    else:
        df = crashes[(crashes['BOROUGH'] == borough.upper()) & (crashes['CRASH_YEAR'] == year) &
                     (crashes['CRASH_MONTH'] == month)]

    lat = df['LATITUDE']
    lon = df['LONGITUDE']
    df_text = df['CONTRIBUTING FACTOR VEHICLE 1']

    fig = go.Figure(go.Scattermapbox(
        lat=lat.tolist(),
        lon=lon.tolist(),
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=5
        ),
        text=df_text
    ))

    fig.update_layout(
        hovermode='closest',
        width=960,
        height=600,
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=40.7,
                lon=-74
            ),
            pitch=0,
            zoom=8
        )
    )
    return fig


mapbox_access_token = 'pk.eyJ1IjoiYWdhcndhbGFkYXJzaCIsImEiOiJja2h5ZGYyd3UwZTN3MnFwYzM1YW9qNnFvIn0.SasVV15822weUxlZ3G0P8Q'
crashes, persons = load_collision_data(NYC_collision_crashes_file, NYC_collision_persons_file)
crashes.loc[:, 'CRASH_YEAR'] = crashes['CRASH DATE'].astype(np.str_).apply(lambda x: int(x.split('/')[-1]))

"""
Hypothesis: Of all collisions occurring late in the night (between 12 am - 5 am), the majority are caused due to overspeeding.
"""
night_crash_data = get_night_crashes(crashes)
night_crash_unsafe_speed_data = check_for_unsafe_speed(night_crash_data)
percentage_unsafe_speed_collisions = calculate_percentage_of_speedy_collisions(night_crash_unsafe_speed_data)
percentage_invalid_collision_data = calculate_invalid_collision_percentage(night_crash_data)
print(percentage_invalid_collision_data)

"""
Hypothesis: Of all crashes, a majority number is caused by persons between the age of 16-25.
"""
crashes_persons = get_merged_crashes_persons(crashes, persons)

# dropping all rows where there is no vehicle ID present
crashes_persons.drop(crashes_persons.loc[crashes_persons['VEHICLE_ID'].isna()].index, inplace=True)
columns = ['COLLISION_ID', 'VEHICLE_ID', 'PERSON_TYPE', 'POSITION_IN_VEHICLE', 'PERSON_AGE']
crashes_persons_age_grouping = get_crashes_persons_age_grouping_data(crashes_persons, columns)
crashes_persons_age_grouping.loc[:, 'ageBelow16'] = np.where(crashes_persons_age_grouping['PERSON_AGE'] < 16, True, False)

# Since there are only 8528 such rows where the age is below 16, we will be dropping those from the analysis
crashes_persons_age_grouping.drop(
    crashes_persons_age_grouping.loc[crashes_persons_age_grouping['PERSON_AGE'] < 16].index,
    inplace=True)
print(calculate_percentage_drivers_between_ages_16_25(crashes_persons_age_grouping))

"""
Hypothesis: The number of collisions increased with an increase in population
Source - https://worldpopulationreview.com/us-cities/new-york-city-ny-population
"""
NYC_Population = get_nyc_population_data()
crashes_total = get_total_crashes_per_year(crashes)
crashes_population = calculate_crashes_per_capita(crashes_total, NYC_Population)
plot_crashes_per_capita_vs_year(crashes_population)
plot_crashes_per_capita_vs_population_density(crashes_population)

crashes_population_subset = crashes_population.drop([0, 8], 0)
plot_crashes_per_capita_vs_year(crashes_population_subset)
plot_crashes_per_capita_vs_population_density(crashes_population_subset)

"""
Hypothesis: Crash locations are not random. The collisions are bound to specific areas due to a badly planned network of roads/traffic signs.
"""
crashes = prepare_crashes_data_for_maps(crashes, 'CRASH_DATE')
nyc_map_fig = plot_crash_locations(mapbox_access_token, crashes, 2014, 5)
nyc_map_fig.show()