import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score 
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer


from sklearn.preprocessing import MinMaxScaler

from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus

pd.set_option('display.float_format', lambda x: '%.4f' % x)

metrics_leaderboard = pd.DataFrame(columns=[
    'Model', 
    'Train Acc', 'Test Acc', 
    'Train Prec', 'Test Prec', 
    'Train Rec', 'Test Rec', 
    'Train F1', 'Test F1',
    'CV F1 Mean', 'CV F1 Std'
])

def log_and_display_metrics(
        model_name, 
        model, 
        X_train, 
        y_train, 
        X_test, 
        y_test, 
        pos_label='Female', 
        cv=5
        ):
    global metrics_leaderboard
    
    f1_scorer = make_scorer(f1_score, pos_label=pos_label, zero_division=0)
    
    cv_scores = cross_val_score(model, X_train, y_train, scoring=f1_scorer, cv=cv)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    new_row = pd.DataFrame({
        'Model': [model_name],
        'Train Acc': [round(accuracy_score(y_train, y_train_pred), 4)],
        'Test Acc': [round(accuracy_score(y_test, y_test_pred), 4)],
        'Train Prec': [round(precision_score(y_train, y_train_pred, pos_label=pos_label, zero_division=0), 4)],
        'Test Prec': [round(precision_score(y_test, y_test_pred, pos_label=pos_label, zero_division=0), 4)],
        'Train Rec': [round(recall_score(y_train, y_train_pred, pos_label=pos_label, zero_division=0), 4)],
        'Test Rec': [round(recall_score(y_test, y_test_pred, pos_label=pos_label, zero_division=0), 4)],
        'Train F1': [round(f1_score(y_train, y_train_pred, pos_label=pos_label, zero_division=0), 4)],
        'Test F1': [round(f1_score(y_test, y_test_pred, pos_label=pos_label, zero_division=0), 4)],
        'CV F1 Mean': [round(cv_scores.mean(), 4)],
        'CV F1 Std': [round(cv_scores.std(), 4)]
    })
    
    metrics_leaderboard = pd.concat([metrics_leaderboard, new_row], ignore_index=True)
    display(metrics_leaderboard)


def plot_the_test_results(X_test, y_test, y_predict):
    POS_LABEL = "Female"
    tp_mask = (y_test == POS_LABEL) & (y_predict == POS_LABEL)
    fp_mask = (y_test != POS_LABEL) & (y_predict == POS_LABEL)
    tn_mask = (y_test != POS_LABEL) & (y_predict != POS_LABEL)
    fn_mask = (y_test == POS_LABEL) & (y_predict != POS_LABEL)

    X_test_array = np.array(X_test)
    
    tp_data = X_test_array[tp_mask]
    fp_data = X_test_array[fp_mask]
    tn_data = X_test_array[tn_mask]
    fn_data = X_test_array[fn_mask]

    feature_names = X.columns[:2]
    
    f, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(tn_data[:, 0], tn_data[:, 1], color='green', marker='o', s=60, label='TN: Correct Male', alpha=0.6)
    ax.scatter(tp_data[:, 0], tp_data[:, 1], color='blue', marker='o', s=60, label='TP: Correct Female', alpha=0.6)
    ax.scatter(fn_data[:, 0], fn_data[:, 1], color='red', marker='X', s=100, label='FN: Missed Female', alpha=0.6)
    ax.scatter(fp_data[:, 0], fp_data[:, 1], color='orange', marker='X', s=80, label='FP: Incorrect Female', alpha=0.6)

    ax.set_title(f"Model Classification Performance")
    ax.set_xlabel(f"{feature_names[0]}")
    ax.set_ylabel(f"{feature_names[1]}")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title="Prediction Status", loc='best')
    
    plt.show()

def plot_confusion_matrices(model_name, y_train_true, y_train_pred, y_test_true, y_test_pred, labels=['Male', 'Female']):

    conf_mx_train = confusion_matrix(y_train_true, y_train_pred, labels=labels)
    conf_mx_test = confusion_matrix(y_test_true, y_test_pred, labels=labels)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    plt.subplots_adjust(wspace=0.3)
    
    sns.heatmap(conf_mx_train, annot=True, fmt="d", cbar=False, 
                xticklabels=labels, yticklabels=labels, ax=ax1)
    ax1.set_title(f'{model_name} - Training Data')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position('top')
    
    sns.heatmap(conf_mx_test, annot=True, fmt="d", cbar=False, cmap="Oranges", 
                xticklabels=labels, yticklabels=labels, ax=ax2)
    ax2.set_title(f'{model_name} - Testing Data')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.xaxis.tick_top()
    ax2.xaxis.set_label_position('top')
    
    plt.show()