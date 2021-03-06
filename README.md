# Analysis of NYC Motor Vehicle Collisions

## Introduction

The NYC open data initiative provides a vast number of data sources for analyses. New York City is densely populated
with a huge local as well as tourist populations. A substantial portion of this population also owns motor vehicles of
different kinds. There are numerous motor vehicle collisions that are reported, and we decided to analyse different
causes and metrics around these collisions' data.

The dataset contains the details about the crashes reported in NYC from 2012-2020.

## Hypotheses

We came up with a few scenarios that we wanted to verify with the help of the data:

#### Hypothesis - I

During the night (12 am - 5 am) the streets are usually empty, and some traffic lights are also turned off. Also, there
are lesser pedestrians around. So it could be possible that majority of the collisions occurring at night are caused
because people driving at night tend to drive with unsafe speed.

##### Results

Based on the calculations, we can see that only about 2.7% of the total number of collisions occurring late in the night
are caused due to an unsafe vehicular speed. We did notice, around 34% of the night crash data had unspecified/invalid
contributing factors for collision, a part of which could have contributed to the Unsafe speed subset. However, it is
safe to say that the hypothesis has been proved false.

#### Hypothesis - II

More crashes are caused by the young inexperienced drivers (assuming 16-25 years of age) as compared to the more
experienced drivers (assuming ages 26-99)

##### Results

![plot](https://github.com/SanhitaD207/2020Fall_projects/blob/main/images/NYC_crashes_per_year_age_group.png?raw=true)
![plot](https://github.com/SanhitaD207/2020Fall_projects/blob/main/images/NYC_normalized_crashes_per_100000people.png?raw=true)

As seen in the normalized values above, for the year 2016, we can see for every 100,000 people in the age group 16-25,
there were around 3245 people involved in collisions, whereas in the age group 26-99 the number is higher (4395). This
number is consistently higher for the 26-99 age group, but the numbers for the 16-25 age group are almost comparable.

#### Hypothesis - III

The number of collisions increased with an increase in population density because more people will be driving on the
roads.

##### Results

![plot](https://github.com/SanhitaD207/2020Fall_projects/blob/main/images/NYC_crashes_per_capita_vs_year.png?raw=true)
![plot](https://github.com/SanhitaD207/2020Fall_projects/blob/main/images/NYC_pop_density_vs_year.png?raw=true)

From the plots above, we can see that from 2013-2016 the population density was rising and so was the crashes per
capita. However, from 2016-2018 the population density was decreasing, but the crashes per capita kept increasing.
Hence, we can conclude that crashes per capita is not positively correlated with the population density, there could be
other factors contributing to the number of crashes.

## Files

- NYCMotorVehicleCollisions.py - Python file containing all the code with doctests
- NYCMotorVehicleCollisions.ipynb - Jupyter Notebook file (Please view the notebook on
  this [URL](https://nbviewer.jupyter.org/github/SanhitaD207/2020Fall_projects/blob/main/NYCMotorVehicleCollisions.ipynb)
  instead. This is because the plotly charts contain javascript code which is not rendered on when viewing the notebook
  on GitHub). This URL points to the NBViewer offered by the Jupyter org.

- population_by_age_2010.csv - CSV file containing data about the 2010 NYC population demographics
- zipcode.geojson - Geojson file containing the shape data for each zipcode of NYC
- doctest_dummy_files - Folder containing dummy CSV files for the doctests in the codebase
- images - Contains PNG files for images for the readme

## Data Sources

- https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Person/f55k-p6yu
- https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95
- https://worldpopulationreview.com/us-cities/new-york-city-ny-population
- https://www1.nyc.gov/assets/planning/download/office/planning-level/nyc-population/census2010/totpop_singage_sex2010_boro.xlsx
- https://data.beta.nyc/dataset/nyc-zip-code-tabulation-areas/resource/6df127b1-6d04-4bb7-b983-07402a2c3f90

## References

- _Stack Overflow_ :
    - https://stackoverflow.com/questions/50375985/pandas-add-column-with-value-based-on-condition-based-on-other-columns
    - https://stackoverflow.com/questions/19384532/get-statistics-for-each-group-such-as-count-mean-etc-using-pandas-groupby
    - https://stackoverflow.com/questions/44111307/python-pandas-count-rows-based-on-column
    - https://stackoverflow.com/questions/47502891/removing-group-header-after-pandas-aggregation
    - https://stackoverflow.com/questions/14529838/apply-multiple-functions-to-multiple-groupby-columns

- _Others_ :
    - https://www.kite.com/python/answers/how-to-select-rows-by-multiple-label-conditions-with-pandas-loc-in-python
    - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html
    - https://datatofish.com/line-chart-python-matplotlib/
    - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html
    - https://github.com/rahulrohri/final_project_2020Sp
    - https://studio.mapbox.com/
    - https://plotly.com/python/scattermapbox/
    - https://data.beta.nyc/dataset/nyc-zip-code-tabulation-areas/resource/6df127b1-6d04-4bb7-b983-07402a2c3f90
    - https://plotly.com/python/mapbox-county-choropleth/\
    




