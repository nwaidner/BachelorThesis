from sklearn.ensemble import RandomForestClassifier

def rf_classifier(X_train, X_test, y_train, y_test, task_name, random_state):

    # Train/ Test
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    name = rf_classifier.__name__


    return y_test, y_pred, name, task_name, random_state








'''

print(classification_report(y_test, y_pred))


# Feature Importance
feature_importance = clf.feature_importances_

# Get column names from X_train
columns = X.columns

for feature, column in sorted(zip(feature_importance, columns), reverse=True)[:20]:
    print(f"{column}: {feature:.4f}")'''


'''# Feature Importance
feature_importance = clf.feature_importances_

# Get column names from X_train
columns = X.columns

# Sort and get the top 10 features
top_10_features = sorted(zip(feature_importance, columns), reverse=True)[:10]

# Separate the values into two lists
top_10_values = [value for value, _ in top_10_features]
top_10_column_names = [column for _, column in top_10_features]

selected_columns_data = unit1_merged[['Zeitstempel', 'Defect'] + top_10_column_names]

class_colors = {
    'sensordefect 1': 'blue',
    'sensordefect 2': 'purple',
    'sensordefect 3': 'green',
    'sensordefect 4': 'brown',
    'kein defekt': 'yellow'
}

# Setze den Zeitstempel als Index

# Erstelle ein Scatter-Plot für jedes Top Feature
for feature in top_10_column_names:
    plt.figure(figsize=(20, 6))

    for class_label, color in class_colors.items():
        class_data = selected_columns_data[selected_columns_data['Defect'] == class_label]

        alpha_value = 0.1 if class_label == 'kein defekt' else 1.0

        plt.scatter(class_data.index, class_data[feature], label=class_label, color=color, s=2, alpha=alpha_value)

    plt.title(f'Scatter-Plot für Feature: {feature}')
    plt.xlabel('Datenpunkt')
    plt.ylabel('Feature-Value')


    # Nur drei Datumswerte anzeigen
    x_ticks = plt.gca().get_xticks()
    plt.xticks(x_ticks[::len(x_ticks)//3])

    plt.legend()
    plt.show()
'''