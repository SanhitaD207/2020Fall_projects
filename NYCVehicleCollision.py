import pandas as pd


class NYCMotorVehicleCollisions:

    def __init__(self):
        self.crashes = {}
        self.persons = {}
        self.vehicles = {}


    def _load_dataframes(self):
        self.crashes = pd.read_csv('Motor_Vehicle_Collisions_-_Crashes.csv', low_memory=False)
        self.persons = pd.read_csv('Motor_Vehicle_Collisions_-_Person.csv', low_memory=False)
        self.vehicles = pd.read_csv('Motor_Vehicle_Collisions_-_Vehicles.csv', low_memory=False)

        print('Crashes df shape ', self.crashes.shape)
        print('Persons df shape ', self.persons.shape)
        print('Vehicles df shape ', self.vehicles.shape)
