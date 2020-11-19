import pandas as pd
from datetime import datetime, time


class NYCMotorVehicleCollisions:

    def __init__(self):
        self.crashes = {}
        self.persons = {}
        self.vehicles = {}

    def _load_dataframes(self):
        self.crashes = pd.read_csv('Motor_Vehicle_Collisions_-_Crashes.csv', low_memory=False)
        # self.persons = pd.read_csv('Motor_Vehicle_Collisions_-_Person.csv', low_memory=False)
        # self.vehicles = pd.read_csv('Motor_Vehicle_Collisions_-_Vehicles.csv', low_memory=False)

        # print('Crashes df shape ', self.crashes)
        # print('Persons df shape ', self.persons)
        # print('Vehicles df shape ', self.vehicles.shape)
        self.night_crashes_analysis()

    def night_crashes_analysis(self):
        print('Crashes df shape ', self.crashes)
        time_data = self.crashes['CRASH TIME']
        print(time_data)

        self.crashes['CRASH TIME'] = self.crashes['CRASH TIME'].apply(lambda x: datetime.strptime(x, "%H:%M").time())

        night_crashes_analysis = self.crashes[(self.crashes['CRASH TIME'] < time(5, 0, 0))]

        print('Crashes df shape ', self.crashes)
        print(night_crashes_analysis)


collision_data = NYCMotorVehicleCollisions()
collision_data._load_dataframes()
