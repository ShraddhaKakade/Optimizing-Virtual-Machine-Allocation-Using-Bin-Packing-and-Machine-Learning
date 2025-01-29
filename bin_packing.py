import pandas as pd
import sqlite3
import argparse
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class Server:
    def __init__(self, id, resources):
        self.id = id
        self.resources = resources  # Remaining resources
        self.vms = {}  # Dictionary of VM ID to resource allocation

    def can_allocate(self, vm_resources):
        return all(self.resources[i] >= vm_resources[i] for i in range(len(self.resources)))

    def allocate(self, vm_id, vm_resources):
        self.vms[vm_id] = vm_resources
        for i in range(len(self.resources)):
            self.resources[i] -= vm_resources[i]

    def deallocate(self, vm_id):
        if vm_id in self.vms:
            vm_resources = self.vms.pop(vm_id)
            for i in range(len(self.resources)):
                self.resources[i] += vm_resources[i]


def load_data(sqlite_file):
    conn = sqlite3.connect(sqlite_file)
    vm_df = pd.read_sql_query('SELECT * FROM vm', conn)
    vm_type_df = pd.read_sql_query('SELECT * FROM vmType', conn)
    conn.close()
    return vm_df, vm_type_df


def preprocess_events(vm_df):
    events = []
    for _, row in vm_df.iterrows():
        vm_id, vm_type_id, start_time, end_time = row['vmId'], row['vmTypeId'], row['starttime'], row['endtime']
        events.append((start_time, 'start', vm_id, vm_type_id))
        if not pd.isna(end_time):
            events.append((end_time, 'end', vm_id, vm_type_id))
    events.sort(key=lambda x: x[0])  # Sort by time
    return events


def determine_server_capacity():
    # General-purpose cloud server specification
    return [64, 256, 4000, 40]  # Core, Memory (GB), SSD (GB), NIC (Gbps)


def simulate(events, vm_type_df, policy, num_vms):
    vm_types = {
        row['vmTypeId']: [row['core'], row['memory'], row['ssd'], row['nic']] for _, row in vm_type_df.iterrows()
    }

    server_capacity = determine_server_capacity()
    servers = []

    for t, event, vm_id, vm_type_id in events[:num_vms]:
        vm_resources = vm_types[vm_type_id]

        if event == 'start':
            allocated = False

            if policy == 'ffd':
                for server in servers:
                    if server.can_allocate(vm_resources):
                        server.allocate(vm_id, vm_resources)
                        allocated = True
                        break

                if not allocated:
                    new_server = Server(len(servers), server_capacity[:])
                    new_server.allocate(vm_id, vm_resources)
                    servers.append(new_server)

            elif policy == 'bfd':
                best_server = None
                min_remaining = float('inf')

                for server in servers:
                    if server.can_allocate(vm_resources):
                        remaining = sum(server.resources[i] - vm_resources[i] for i in range(len(vm_resources)))
                        if remaining < min_remaining:
                            min_remaining = remaining
                            best_server = server

                if best_server:
                    best_server.allocate(vm_id, vm_resources)
                    allocated = True
                else:
                    new_server = Server(len(servers), server_capacity[:])
                    new_server.allocate(vm_id, vm_resources)
                    servers.append(new_server)

            elif policy == 'cosine':
                from scipy.spatial.distance import cosine
                best_server = None
                max_similarity = -1

                for server in servers:
                    if server.can_allocate(vm_resources):
                        similarity = 1 - cosine(server.resources, vm_resources)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_server = server

                if best_server:
                    best_server.allocate(vm_id, vm_resources)
                    allocated = True
                else:
                    new_server = Server(len(servers), server_capacity[:])
                    new_server.allocate(vm_id, vm_resources)
                    servers.append(new_server)

        elif event == 'end':
            for server in servers:
                if vm_id in server.vms:
                    server.deallocate(vm_id)
                    break

    return len(servers)


def save_results_to_excel(policy, num_vms, servers_required):
    # Load existing data if the file exists, else create a new DataFrame
    try:
        results_df = pd.read_excel('policy_results.xlsx')
    except FileNotFoundError:
        results_df = pd.DataFrame(columns=['Policy', 'NumVMs', 'ServersRequired'])

    # Append the new result
    new_result = {'Policy': policy, 'NumVMs': num_vms, 'ServersRequired': servers_required}
    results_df = pd.concat([results_df, pd.DataFrame([new_result])], ignore_index=True)

    # Save back to Excel
    results_df.to_excel('policy_results.xlsx', index=False)
    print("Results saved to policy_results.xlsx")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, required=True, help='Bin-packing policy (e.g., ffd, bfd, cosine, ml)')
    parser.add_argument('--numvms', type=int, default=1000, help='Number of VMs to simulate')
    parser.add_argument('--sqlite', type=str, required=True, help='Path to SQLite file with dataset')
    args = parser.parse_args()

    vm_df, vm_type_df = load_data(args.sqlite)
    events = preprocess_events(vm_df)
    num_servers = simulate(events, vm_type_df, args.policy, args.numvms)

    print(f"Number of servers required for {args.numvms} VMs using {args.policy} policy: {num_servers}")

    # Save the result to Excel
    save_results_to_excel(args.policy, args.numvms, num_servers)


if __name__ == "__main__":
    main()
