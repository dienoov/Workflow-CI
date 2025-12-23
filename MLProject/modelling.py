import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow

if __name__ == '__main__':
    df = pd.read_csv('abalone_preprocessing.csv')
    X = df.drop(columns=['rings'])
    y = df['rings']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    mlflow.set_experiment('modelling')

    model = RandomForestClassifier(random_state=42)

    model.fit(X_train, y_train)

    mlflow.log_metric('accuracy', model.score(X_test, y_test))

    mlflow.sklearn.log_model(
        model,
        name='model',
    )
