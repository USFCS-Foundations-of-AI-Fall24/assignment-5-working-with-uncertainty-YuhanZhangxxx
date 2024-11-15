import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

def decision_tree_cross_validation():
    wine = load_wine()
    X, y = wine.data, wine.target
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        scores.append(score)

    print(f"Cross-validation scores: {scores}")

def random_forest_grid_search():
    wine = load_wine()
    X, y = wine.data, wine.target
    n_estimators_list = [10, 25, 50]
    criteria = ['gini', 'entropy']
    results = []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for criterion in criteria:
        for n_estimators in n_estimators_list:
            scores = []

            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                clf = RandomForestClassifier(n_estimators=n_estimators,
                                             criterion=criterion,
                                             random_state=42)
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                scores.append(score)

            avg_score = sum(scores) / len(scores)
            results.append({
                'Criterion': criterion,
                'Estimators': n_estimators,
                'Scores': scores,
                'Average Score': avg_score
            })

    df_results = pd.DataFrame(results)
    df_results['Average Score'] = df_results['Average Score'].round(4)

    table = df_results[['Criterion', 'Estimators', 'Average Score']]
    print(table)

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off')
    ax.table(cellText=table.values, colLabels=table.columns, loc='center')
    plt.tight_layout()
    plt.savefig('random_forest_results.png', dpi=300)
    plt.show()

def hyperparameter_search_comparison():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    models = {
        "Random Forest": RandomForestClassifier(random_state=0),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=0, early_stopping=False)
    }

    param_grids = {
        "Random Forest": {"n_estimators": [5, 10, 15, 20]},
        "Hist Gradient Boosting": {"max_iter": [25, 50, 75, 100]}
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    all_results = pd.DataFrame()

    for name, model in models.items():
        grid_search = GridSearchCV(estimator=model,
                                   param_grid=param_grids[name],
                                   cv=cv,
                                   return_train_score=True)
        grid_search.fit(X, y)

        cv_results_df = pd.DataFrame(grid_search.cv_results_)
        cv_results_df['model'] = name
        all_results = pd.concat([all_results, cv_results_df], ignore_index=True)

    rf_results = all_results[all_results['model'] == 'Random Forest']
    rf_results.loc[:, 'n_estimators'] = rf_results['param_n_estimators'].astype(int)
    fig_rf = px.scatter(rf_results, x='n_estimators', y='mean_test_score',
                        title='Random Forest Performance',
                        labels={'n_estimators': 'Number of Estimators', 'mean_test_score': 'Mean Test Score'})
    fig_rf.show()

    hgb_results = all_results[all_results['model'] == 'Hist Gradient Boosting']
    hgb_results.loc[:, 'max_iter'] = hgb_results['param_max_iter'].astype(int)
    fig_hgb = px.scatter(hgb_results, x='max_iter', y='mean_test_score',
                         title='Histogram Gradient Boosting Performance',
                         labels={'max_iter': 'Number of Iterations', 'mean_test_score': 'Mean Test Score'})
    fig_hgb.show()

if __name__ == "__main__":
    decision_tree_cross_validation()
    random_forest_grid_search()
    hyperparameter_search_comparison()
