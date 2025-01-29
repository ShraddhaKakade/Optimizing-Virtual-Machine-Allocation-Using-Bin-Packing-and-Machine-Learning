import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import sqlite3
import os


def load_data_from_sqlite(sqlite_file):
    """
    Load VM and VMType data from SQLite file.
    
    Args:
        sqlite_file (str): Path to the SQLite file.
    
    Returns:
        tuple: DataFrames for VM and VMType tables.
    """
    conn = sqlite3.connect(sqlite_file)
    vm_df = pd.read_sql_query("SELECT * FROM vm", conn)
    vm_type_df = pd.read_sql_query("SELECT * FROM vmType", conn)
    conn.close()
    return vm_df, vm_type_df


def train_ml_model_from_excel(file_path):
    """
    Train a machine learning model using data from the generated Excel file.
    
    Args:
        file_path (str): Path to the Excel file containing policy results.
    
    Returns:
        model: Trained Random Forest Regressor model.
    """
    # Load data from the Excel file
    data = pd.read_excel(file_path)

    # Features: Number of VMs
    X = data[['NumVMs']].values  # 2D array required by scikit-learn

    # Target: Number of Servers
    y = data['ServersRequired'].values  # 1D array

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Mean Absolute Error on Test Set: {mae:.2f}")

    return model


def predict_servers(model, num_vms):
    """
    Predict the number of servers required for a given number of VMs using the trained model.
    
    Args:
        model: Trained ML model.
        num_vms (int): Number of VMs for prediction.
    
    Returns:
        int: Predicted number of servers.
    """
    num_servers = model.predict([[num_vms]])[0]
    return round(num_servers)


def simulate_ml_policy(file_path, num_vms):
    """
    Simulate the ML policy to predict the number of servers required.
    
    Args:
        file_path (str): Path to the Excel file containing policy results.
        num_vms (int): Number of VMs for simulation.
    
    Returns:
        int: Predicted number of servers.
    """
    # Train the ML model
    print("Training ML model using policy results...")
    ml_model = train_ml_model_from_excel(file_path)

    # Predict the number of servers for the specified number of VMs
    predicted_servers = predict_servers(ml_model, num_vms)
    return predicted_servers


def save_results_to_excel(policy, num_vms, num_servers):
    """
    Save the policy results to an Excel file.
    
    Args:
        policy (str): The bin-packing policy used.
        num_vms (int): Number of VMs simulated.
        num_servers (int): Number of servers required.
    """
    file_name = "policy_results.xlsx"
    new_row = {'Policy': policy, 'NumVMs': num_vms, 'ServersRequired': num_servers}

    if os.path.exists(file_name):
        df = pd.read_excel(file_name)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df.to_excel(file_name, index=False)
    print(f"Results saved to {file_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--numvms', type=int, default=1000, help='Number of VMs to simulate')
    parser.add_argument('--sqlite', type=str, help='Path to SQLite file with dataset (optional)')
    parser.add_argument('--file', type=str, default="policy_results.xlsx", help='Path to the policy results Excel file')
    args = parser.parse_args()

    if args.sqlite:
        print(f"Loading data from SQLite: {args.sqlite}")
        vm_df, vm_type_df = load_data_from_sqlite(args.sqlite)
        # If needed, you can append data to policy_results.xlsx here.
        print(f"Loaded {len(vm_df)} VMs and {len(vm_type_df)} VM types from SQLite.")

    # Simulate ML policy
    predicted_servers = simulate_ml_policy(args.file, args.numvms)
    print(f"Predicted number of servers for {args.numvms} VMs using ML policy: {predicted_servers}")

    # Save the prediction to Excel
    save_results_to_excel("ml", args.numvms, predicted_servers)


if __name__ == "__main__":
    main()
