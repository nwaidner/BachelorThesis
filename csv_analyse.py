import pandas as pd


class ClassificationResult:
    def __init__(self, scenario, random_state, classifier_name, accuracy, balanced_accuracy, f1, runtime):
        self.scenario = scenario
        self.random_state = random_state
        self.classifier_name = classifier_name
        self.accuracy = accuracy
        self.balanced_accuracy = balanced_accuracy
        self.f1 = f1
        self.runtime = runtime

    def __str__(self):
        return f"Scenario: {self.scenario}, Random State: {self.random_state}, Classifier: {self.classifier_name}, " \
               f"Accuracy: {self.accuracy}, Balanced Accuracy: {self.balanced_accuracy}, F1: {self.f1}, " \
               f"Runtime: {self.runtime}"


csv_file_path = '/Attic/metrics_FINAL_6.csv'
with open(csv_file_path, 'r') as file:
    csv_data = file.readlines()

results = []
for line in csv_data:
    values = line.strip().split(',')
    scenario, random_state, classifier_name, accuracy, balanced_accuracy, f1, runtime = values
    result = ClassificationResult(scenario, int(random_state), classifier_name, float(accuracy),
                                  float(balanced_accuracy), float(f1), float(runtime))
    results.append(result)

scenario_seed_tables = {}

for result in results:
    key = (result.scenario, result.random_state)

    if key not in scenario_seed_tables:
        scenario_seed_tables[key] = pd.DataFrame(
            columns=['Classifier', 'Accuracy', 'Balanced Accuracy', 'F1', 'Runtime'])

    new_row = pd.DataFrame({
        'Classifier': [result.classifier_name],
        'Accuracy': [result.accuracy],
        'Balanced Accuracy': [result.balanced_accuracy],
        'F1': [result.f1],
        'Runtime': [result.runtime]
    })
    scenario_seed_tables[key] = pd.concat([scenario_seed_tables[key], new_row], ignore_index=True)

for key, table in scenario_seed_tables.items():
    csv_filename = f"table_scenario_{key[0]}_seed_{key[1]}.csv"
    table.to_csv(csv_filename, index=False)
    print(f"Table for Scenario: {key[0]}, Seed: {key[1]} saved as {csv_filename}")
