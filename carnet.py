from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

class CarModel:
    def __init__(self):
        self.model = BayesianNetwork(
            [
                ("Battery", "Radio"),
                ("Battery", "Ignition"),
                ("Ignition", "Starts"),
                ("Gas", "Starts"),
                ("Starts", "Moves"),
                ("KeyPresent", "Starts"),
            ]
        )
        self._define_parameters()

    def _define_parameters(self):
        self.cpd_battery = TabularCPD(
            variable="Battery",
            variable_card=2,
            values=[[0.70], [0.30]],
            state_names={"Battery": ['Works', "Doesn't work"]},
        )

        self.cpd_gas = TabularCPD(
            variable="Gas",
            variable_card=2,
            values=[[0.40], [0.60]],
            state_names={"Gas": ['Full', "Empty"]},
        )

        self.cpd_keypresent = TabularCPD(
            variable="KeyPresent",
            variable_card=2,
            values=[[0.7], [0.3]],
            state_names={"KeyPresent": ['yes', 'no']},
        )

        self.cpd_radio = TabularCPD(
            variable="Radio",
            variable_card=2,
            values=[[0.75, 0.01], [0.25, 0.99]],
            evidence=["Battery"],
            evidence_card=[2],
            state_names={"Radio": ["turns on", "Doesn't turn on"],
                         "Battery": ['Works', "Doesn't work"]}
        )

        self.cpd_ignition = TabularCPD(
            variable="Ignition",
            variable_card=2,
            values=[[0.75, 0.01], [0.25, 0.99]],
            evidence=["Battery"],
            evidence_card=[2],
            state_names={"Ignition": ["Works", "Doesn't work"],
                         "Battery": ['Works', "Doesn't work"]}
        )

        self.cpd_starts = TabularCPD(
            variable="Starts",
            variable_card=2,
            values=[
                [0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
            ],
            evidence=["Ignition", "Gas", "KeyPresent"],
            evidence_card=[2, 2, 2],
            state_names={
                "Starts": ['yes', 'no'],
                "Ignition": ["Works", "Doesn't work"],
                "Gas": ['Full', "Empty"],
                "KeyPresent": ['yes', 'no'],
            },
        )

        self.cpd_moves = TabularCPD(
            variable="Moves",
            variable_card=2,
            values=[[0.8, 0.01], [0.2, 0.99]],
            evidence=["Starts"],
            evidence_card=[2],
            state_names={"Moves": ["yes", "no"],
                         "Starts": ['yes', 'no']}
        )

        self.model.add_cpds(
            self.cpd_starts, self.cpd_ignition, self.cpd_gas, self.cpd_radio,
            self.cpd_battery, self.cpd_moves, self.cpd_keypresent
        )

    def perform_inference(self):
        car_infer = VariableElimination(self.model)

        print("Query 1:")
        result = car_infer.query(variables=['Battery'], evidence={'Moves': 'no'})
        print(result)

        print("\nQuery 2:")
        result = car_infer.query(variables=['Starts'], evidence={'Radio': "Doesn't turn on"})
        print(result)

        print("\nQuery 3:")
        result1 = car_infer.query(variables=['Radio'], evidence={'Battery': 'Works'})
        result2 = car_infer.query(variables=['Radio'], evidence={'Battery': 'Works', 'Gas': 'Full'})
        print("P(Radio | Battery='Works'):\n", result1)
        print("\nP(Radio | Battery='Works', Gas='Full'):\n", result2)

        print("\nQuery 4:")
        result1 = car_infer.query(variables=['Ignition'], evidence={'Moves': 'no'})
        result2 = car_infer.query(variables=['Ignition'], evidence={'Moves': 'no', 'Gas': 'Empty'})
        print("P(Ignition | Moves='no'):\n", result1)
        print("\nP(Ignition | Moves='no', Gas='Empty'):\n", result2)

        print("\nQuery 5:")
        result = car_infer.query(variables=['Starts'], evidence={'Radio': 'turns on', 'Gas': 'Full'})
        print(result)

        print("\nQuery 6:")
        result = car_infer.query(variables=['KeyPresent'], evidence={'Moves': 'no'})
        print("P(KeyPresent | Moves='no'):\n", result)

def main():
    car_model = CarModel()
    car_model.perform_inference()

if __name__ == "__main__":
    main()






