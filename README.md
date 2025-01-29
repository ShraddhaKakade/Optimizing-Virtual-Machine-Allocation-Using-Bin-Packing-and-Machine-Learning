# Optimizing Virtual Machine Allocation Using Bin Packing and Machine Learning

### 1.	File Structure

a.	bin_packing.py

-- The script in this file simulates virtual machine (VM) allocation on servers using different bin-packing strategies (policies). 

-- It reads VM and VM Type data from an SQLite database, processes the events, and determines the number of servers required for a given number of VMs based on the selected policy.

b.	bin_packing_ml.py

-- This script introduces a machine learning based approach to predict the number of servers required to handle a given number of virtual machines (VMs). 

-- The ML model is trained on historical simulation data stored in an Excel file (policy_results.xlsx)

c.	packing_trace_zone_a_v1.sqlite

-- This file has two tables – vm and vmType.

-- The data from these tables is used during simulation of different policies for different counts of VMs.

-- Link to download the file: https://drive.google.com/file/d/1zqawUhrIiETr547MN_z-cuDCfIBX5wMw/view?usp=sharing

d.	policy_results.xlsx

-- This file is updated with the simulation results.

-- It has three columns viz. Policy, NumVms and ServersRequired.
 
### 2.	bin_packing.py Implementation

 The main() function orchestrates the simulation process:

a. Argument Parsing:

--policy: Bin-packing policy (e.g., ffd, bfd, cosine).

--numvms: Number of VMs to simulate (default: 1000).

--sqlite: Path to the SQLite file containing the dataset.

b. Data Loading: Reads VM and VM type data from the SQLite file.

c. Event Preprocessing: Converts VM data into a sorted list of events.

d. Simulation: Calls simulate() to calculate the number of servers required.

e. Results: Prints the number of servers needed and saves the results to an Excel file.

#### •	Server Capacities:

Server capacities for general purpose are defined as follows:

1.	Core = 64
2.	Memory = 256 GB
3.	SSD = 4000 GB
4.	NIC = 40 Gbps

#### •	Input:

1.	vm Table:
a.	vmId: Unique ID of the VM.
b.	vmTypeId: Type of the VM (links to vmType table).
c.	starttime: Start time of the VM.
d.	endtime: End time of the VM (optional).
2.	vmType Table:
a.	vmTypeId: Unique ID of the VM type.
b.	core, memory, ssd, nic: Resource requirements for CPU cores, memory (GB), SSD (GB), and NIC bandwidth (Gbps).

#### •	Output:

1.	Console: Prints the number of servers required for the simulation.
2.	Excel File: Updates or creates “policy_results.xlsx” with the simulation results.

#### •	Bin-Packing Policies:
1.	First Fit Decreasing (FFD): Allocates the VM to the first server with enough capacity.
2.	Best Fit Decreasing (BFD): Allocates the VM to the server with the least remaining capacity after allocation.
3.	Cosine Similarity: Allocates the VM to the server with the highest cosine similarity in resource utilization, promoting better alignment with server profiles.
 
### 3.	bin_packing_ml.py Implementation

-- Predicts the number of servers required for a given number of virtual machines (VMs) using a machine learning (ML) approach.

-- The code uses a Random Forest Regressor from the scikit-learn library, a robust ensemble learning method for regression tasks.

-- Training Data:
•	Features: Number of VMs (NumVMs).
•	Target: Number of servers required (ServersRequired).
•	Data is loaded from an Excel file (policy_results.xlsx) containing results from previous simulations using different policies.

-- Model performance is assessed using the Mean Absolute Error (MAE) metric on the test set.

-- In current scenario, limited data is available for training

### 4. Servers Required V/S Count of Virtual Machines 

<img width="453" alt="image" src="https://github.com/user-attachments/assets/8672ebf6-f71d-4447-b851-c56dc1489a2e" />

