from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

class AlarmModel:
    def __init__(self):
        self.model = BayesianNetwork(
            [
                ("Burglary", "Alarm"),
                ("Earthquake", "Alarm"),
                ("Alarm", "JohnCalls"),
                ("Alarm", "MaryCalls"),
            ]
        )
        self._define_parameters()

    def _define_parameters(self):
        self.cpd_burglary = TabularCPD(
            variable="Burglary",
            variable_card=2,
            values=[[0.999], [0.001]],
            state_names={"Burglary": ['no', 'yes']},
        )

        self.cpd_earthquake = TabularCPD(
            variable="Earthquake",
            variable_card=2,
            values=[[0.998], [0.002]],
            state_names={"Earthquake": ["no", "yes"]},
        )

        self.cpd_alarm = TabularCPD(
            variable="Alarm",
            variable_card=2,
            values=[[0.999, 0.71, 0.06, 0.05], [0.001, 0.29, 0.94, 0.95]],
            evidence=["Burglary", "Earthquake"],
            evidence_card=[2, 2],
            state_names={"Burglary": ['no', 'yes'], "Earthquake": ['no', 'yes'], 'Alarm': ['yes', 'no']},
        )

        self.cpd_johncalls = TabularCPD(
            variable="JohnCalls",
            variable_card=2,
            values=[[0.95, 0.1], [0.05, 0.9]],
            evidence=["Alarm"],
            evidence_card=[2],
            state_names={"Alarm": ['yes', 'no'], "JohnCalls": ['yes', 'no']},
        )

        self.cpd_marycalls = TabularCPD(
            variable="MaryCalls",
            variable_card=2,
            values=[[0.1, 0.7], [0.9, 0.3]],
            evidence=["Alarm"],
            evidence_card=[2],
            state_names={"Alarm": ['yes', 'no'], "MaryCalls": ['yes', 'no']},
        )

        self.model.add_cpds(
            self.cpd_burglary,
            self.cpd_earthquake,
            self.cpd_alarm,
            self.cpd_johncalls,
            self.cpd_marycalls
        )

    def perform_inference(self):
        alarm_infer = VariableElimination(self.model)

        print("Query 1:")
        result = alarm_infer.query(variables=["JohnCalls"], evidence={"Earthquake": "yes"})
        print(result)

        print("\nQuery 2:")
        result = alarm_infer.query(variables=["JohnCalls", "Earthquake"], evidence={"Burglary": "yes", "MaryCalls": "yes"})
        print(result)

        self._additional_queries(alarm_infer)

    def _additional_queries(self, infer):
        print("\nQuery 3:")
        result = infer.query(variables=['MaryCalls'], evidence={'JohnCalls': 'yes'})
        print(result)

        print("\nQuery 4:")
        result = infer.query(variables=['JohnCalls', 'MaryCalls'], evidence={'Alarm': 'yes'})
        print(result)

        print("\nQuery 5:")
        result = infer.query(variables=['Alarm'], evidence={'MaryCalls': 'yes'})
        print(result)

def main():
    alarm_model = AlarmModel()
    alarm_model.perform_inference()

if __name__ == "__main__":
    main()
