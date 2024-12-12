import pandas as pd
import sqlite3
import numpy as np
import argparse
from typing import List, Dict
import pickle
from enum import Enum
import os
from pathlib import Path
from sklearn.cluster import KMeans
from collections import defaultdict
import random

class EventType(Enum):
    START = 1
    END = 2

class Event:
    def __init__(self, time: float, event_type: EventType, vm_id: int, vm_specs: Dict):
        self.time = time
        self.event_type = event_type
        self.vm_id = vm_id
        self.vm_specs = vm_specs

class Server:
    def __init__(self, id: int):
        self.id = id
        self.vms: Dict[int, Dict] = {
                                        # vm_id -> vm_specs
                                    }
        self.resources = ['core', 'memory', 'ssd', 'nic']
        self.available_resources = {
            'core': 1.0,
            'memory': 1.0,
            'ssd': 1.0,
            'nic': 1.0
        }

    def fits(self, vm_specs: Dict) -> bool:
        for resource in self.resources:
            if vm_specs[resource] is not None:
                if self.available_resources[resource] < vm_specs[resource]:
                    return False
        return True

    def add_vm(self, vm_id: int, vm_specs: Dict):
        self.vms[vm_id] = vm_specs
        for resource in self.resources:
            if vm_specs[resource] is not None:
                self.available_resources[resource] -= vm_specs[resource]

    def remove_vm(self, vm_id: int):
        vm_specs = self.vms.pop(vm_id)
        for resource in self.resources:
            if vm_specs[resource] is not None:
                self.available_resources[resource] += vm_specs[resource]

    def get_utilization(self) -> Dict:
        return {
            resource: 1 - self.available_resources[resource]
            for resource in self.resources
        }

##########
     
class EnhancedBFD:
    def __init__(self):
        self.servers: List[Server] = []
        self.vm_to_server: Dict[int, int] = {}
        
    def calculate_server_efficiency(self, server: Server, vm_specs: Dict) -> float:
        """Calculate efficiency score for placing VM on server"""
        if not server.fits(vm_specs):
            return float('-inf')
            
        potential_utilization = sum(
            (1 - server.available_resources[r] + vm_specs[r]) 
            for r in ['core', 'memory', 'ssd', 'nic']
        ) / 4.0
        
        resource_waste = sum(
            server.available_resources[r] - vm_specs[r] 
            for r in ['core', 'memory', 'ssd', 'nic']
        )
        
        return potential_utilization - (0.3 * resource_waste)

    def find_best_server(self, vm_specs: Dict) -> int:
        best_server_id = len(self.servers)
        best_efficiency = float('-inf')
        
        for server in self.servers:
            efficiency = self.calculate_server_efficiency(server, vm_specs)
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_server_id = server.id
                
        return best_server_id

def simulate_with_ebfd(events: List[Event]) -> int:
    ebfd = EnhancedBFD()
    max_servers = 0
    
    current_time = None
    current_vms = []
    
    for event in events:
        if event.event_type == EventType.START:
            if current_time is None:
                current_time = event.time
                
            if event.time == current_time:
                current_vms.append(event)
                continue
            else:
                current_vms.sort(
                    key=lambda e: sum(e.vm_specs.values()), 
                    reverse=True
                )
                
                for vm_event in current_vms:
                    server_id = ebfd.find_best_server(vm_event.vm_specs)
                    
                    if server_id == len(ebfd.servers):
                        ebfd.servers.append(Server(server_id))
                    
                    ebfd.servers[server_id].add_vm(vm_event.vm_id, vm_event.vm_specs)
                    ebfd.vm_to_server[vm_event.vm_id] = server_id
                    max_servers = max(max_servers, len(ebfd.servers))
                
                current_time = event.time
                current_vms = [event]
                
        else:
            if event.vm_id in ebfd.vm_to_server:
                server_id = ebfd.vm_to_server[event.vm_id]
                try:
                    ebfd.servers[server_id].remove_vm(int(event.vm_id))
                except KeyError:
                    ebfd.servers[server_id].remove_vm(event.vm_id)
                del ebfd.vm_to_server[event.vm_id]
    
    if current_vms:
        current_vms.sort(key=lambda e: sum(e.vm_specs.values()), reverse=True)
        for vm_event in current_vms:
            server_id = ebfd.find_best_server(vm_event.vm_specs)
            
            if server_id == len(ebfd.servers):
                ebfd.servers.append(Server(server_id))
            
            ebfd.servers[server_id].add_vm(vm_event.vm_id, vm_event.vm_specs)
            ebfd.vm_to_server[vm_event.vm_id] = server_id
            max_servers = max(max_servers, len(ebfd.servers))
    
    return max_servers

class HybridVMPlacement:
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.cluster_servers = defaultdict(list)
        
    def fit(self, vm_df):
        features = vm_df[['core', 'memory', 'ssd', 'nic']].values
        self.kmeans.fit(features)
        
    def get_cluster(self, vm_specs):
        features = np.array([[
            vm_specs['core'],
            vm_specs['memory'],
            vm_specs['ssd'],
            vm_specs['nic']
        ]])
        return self.kmeans.predict(features)[0]

class ClusterAwareServer(Server):
    def __init__(self, id: int):
        super().__init__(id)
        self.cluster_id = None
        
def hybrid_placement_policy(servers: List[ClusterAwareServer], vm_specs: Dict, 
                          cluster_id: int, power_samples: int = 2) -> int:
    """Hybrid policy combining clustering with power-of-two random selection"""
    
    cluster_servers = [s for s in servers if s.cluster_id == cluster_id 
                      and s.fits(vm_specs)]
    
    if cluster_servers:
        samples = min(power_samples, len(cluster_servers))
        candidates = random.sample(cluster_servers, samples)
        return min(candidates, 
                  key=lambda s: sum(1 - s.available_resources[r] 
                                  for r in ['core', 'memory', 'ssd', 'nic'])).id
    
    available_servers = [s for s in servers if s.fits(vm_specs)]
    if available_servers:
        samples = min(power_samples, len(available_servers))
        candidates = random.sample(available_servers, samples)
        return min(candidates, 
                  key=lambda s: sum(1 - s.available_resources[r] 
                                  for r in ['core', 'memory', 'ssd', 'nic'])).id
    
    return len(servers)

def simulate_hybrid(events: List[Event], vm_df: pd.DataFrame) -> int:
    placer = HybridVMPlacement()
    placer.fit(vm_df)
    
    servers: List[ClusterAwareServer] = []
    vm_to_server: Dict[int, int] = {}
    max_servers = 0
    
    for event in events:
        if event.event_type == EventType.START:
            cluster_id = placer.get_cluster(event.vm_specs)
            server_id = hybrid_placement_policy(servers, event.vm_specs, cluster_id)
            
            if server_id == len(servers):
                servers.append(ClusterAwareServer(server_id))
                servers[-1].cluster_id = cluster_id
            
            servers[server_id].add_vm(event.vm_id, event.vm_specs)
            vm_to_server[event.vm_id] = server_id
            max_servers = max(max_servers, len(servers))
            
        else:
            if event.vm_id in vm_to_server:
                server_id = vm_to_server[event.vm_id]
                servers[server_id].remove_vm(event.vm_id)
                del vm_to_server[event.vm_id]
    
    return max_servers

class LiveMigrationStrategy:
    def __init__(self):
        self.servers: List[Server] = []
        self.vm_to_server: Dict[int, int] = {}
        
    def can_fit(self, server: Server, vm_specs: Dict) -> bool:
        """Check if VM can fit in server"""
        for resource in ['core', 'memory', 'ssd', 'nic']:
            if (server.available_resources[resource] < 
                vm_specs[resource] if vm_specs[resource] is not None else 0):
                return False
        return True
        
    def find_best_server(self, vm_specs: Dict) -> int:
        """Find server with best fit for VM"""
        best_server_id = len(self.servers)
        min_waste = float('inf')
        
        for server in self.servers:
            if not self.can_fit(server, vm_specs):
                continue
                
            waste = sum(
                server.available_resources[r] - vm_specs[r] 
                for r in ['core', 'memory', 'ssd', 'nic']
                if vm_specs[r] is not None
            )
            
            if waste < min_waste:
                min_waste = waste
                best_server_id = server.id
                
        return best_server_id
        
    def migrate_vms(self):
        """Optimize placement through live migration"""
        for source_id, source_server in enumerate(self.servers):
            for target_id, target_server in enumerate(self.servers):
                if source_id == target_id:
                    continue
                    
                # Try to migrate VMs from source to target
                for vm_id, vm_specs in list(source_server.vms.items()):
                    if self.can_fit(target_server, vm_specs):
                        # Migrate VM
                        source_server.remove_vm(vm_id)
                        target_server.add_vm(vm_id, vm_specs)
                        self.vm_to_server[vm_id] = target_id

def simulate_with_migration(events: List[Event]) -> int:
    strategy = LiveMigrationStrategy()
    max_servers = 0
    
    for event in events:
        if event.event_type == EventType.START:
            server_id = strategy.find_best_server(event.vm_specs)
            
            if server_id == len(strategy.servers):
                strategy.servers.append(Server(server_id))
            
            strategy.servers[server_id].add_vm(event.vm_id, event.vm_specs)
            strategy.vm_to_server[event.vm_id] = server_id
            
            if len(strategy.vm_to_server) % 100 == 0:
                strategy.migrate_vms()
            
            max_servers = max(max_servers, len(strategy.servers))
            
        else:
            if event.vm_id in strategy.vm_to_server:
                server_id = strategy.vm_to_server[event.vm_id]
                strategy.servers[server_id].remove_vm(event.vm_id)
                del strategy.vm_to_server[event.vm_id]
                
                if len(strategy.vm_to_server) % 100 == 0:
                    strategy.migrate_vms()
    
    return max_servers


def calculate_vm_size(vm_specs: Dict) -> float:
    """Calculate VM size for sorting using different methods"""
    return max(vm_specs[r] if vm_specs[r] is not None else 0 
              for r in ['core', 'memory', 'ssd', 'nic'])


def ffd_policy(servers: List[Server], vm_specs: Dict) -> int:
    """First-Fit Decreasing policy"""
    for server in servers:
        if server.fits(vm_specs):
            return server.id
    return len(servers)

def cosine_policy(servers: List[Server], vm_specs: Dict) -> int:
    """Cosine similarity based policy"""
    best_similarity = -1
    best_server_id = len(servers)
    
    vm_vector = np.array([vm_specs[r] if vm_specs[r] is not None else 0 
                         for r in ['core', 'memory', 'ssd', 'nic']])
    
    for server in servers:
        if not server.fits(vm_specs):
            continue
            
        server_used = np.array([1 - server.available_resources[r] 
                               for r in ['core', 'memory', 'ssd', 'nic']])
        
        if np.all(server_used == 0):
            return server.id
            
        similarity = np.dot(vm_vector, server_used) / (np.linalg.norm(vm_vector) * np.linalg.norm(server_used))
        if similarity > best_similarity:
            best_similarity = similarity
            best_server_id = server.id
            
    return best_server_id

def simulate(events: List[Event], policy_name: str) -> int:
    # N_SERVERS = 10000
    servers: List[Server] = []
    vm_to_server: Dict[int, int] = {}
    max_servers = 0

    if policy_name == 'ffd':
        events = sorted(
            events,
            key=lambda x: sum(x.vm_specs.values(), -x.time),   # sort by size desc, then time asc
            reverse=True
        )

    policy = ffd_policy if policy_name == 'ffd' else cosine_policy
    
    for event in events:
        if event.event_type == EventType.START:
            server_id = policy(servers, event.vm_specs)
            
            if server_id == len(servers):
                servers.append(Server(server_id))
                
            servers[server_id].add_vm(event.vm_id, event.vm_specs)
            # print(f"VM {event.vm_id} added to server {server_id}, utilization = {servers[server_id].get_utilization()}")
            vm_to_server[event.vm_id] = server_id
            max_servers = max(max_servers, len(servers))
            
        else:
            if event.vm_id in vm_to_server:
                server_id = vm_to_server[event.vm_id]
                servers[server_id].remove_vm(event.vm_id)
                # print(f"VM {event.vm_id} removed from server {server_id}, utilization = {servers[server_id].get_utilization()}")
                del vm_to_server[event.vm_id]
    
    return max_servers

def load(vm_df: pd.DataFrame) -> List[Event]:

    events = []
    for _, row in vm_df.iterrows():
        vm_specs = {
            'core': row['core'],
            'memory': row['memory'],
            'ssd': row['ssd'],
            'nic': row['nic']
        }
        
        events.append(Event(row['starttime'], EventType.START, row['vmId'], vm_specs))
        if pd.notna(row['endtime']):
            events.append(Event(row['endtime'], EventType.END, row['vmId'], vm_specs))
    
    events.sort(key=lambda x: x.time)
    return events

def load_events(num_vms: int, vm_df: pd.DataFrame, pickle_dir: str = 'pkl') -> List[Event]:

    Path(pickle_dir).mkdir(exist_ok=True)    
    pickle_path = os.path.join(pickle_dir, f'events_{num_vms}.pkl')

    # load pickled data
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    
    # create events
    events = load(vm_df)
    
    # save events
    with open(pickle_path, 'wb') as f:
        pickle.dump(events, f)
    
    return events

def main():

    db_path = 'data/packing_trace_zone_a_v1.sqlite'
    pickle_dir = 'pkl'

    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', choices=['ffd', 'cosine', 'hybrid', 'migration', 'ebfd'], required=True)
    parser.add_argument('--numvms', type=int, required=True)
    args = parser.parse_args()

    c = sqlite3.connect(db_path)
    q = 'SELECT vmId, vmTypeId, starttime, endtime FROM vm'
    vm_df = pd.read_sql_query(q, c)

    q = 'SELECT vmTypeId, core, memory, ssd, nic FROM vmType'
    vmtype_df = pd.read_sql_query(q, c)    

    # limit to number of VMs
    vm_df = vm_df.head(args.numvms)
    
    # merge into one df of VM specs
    vm_df = vm_df.merge(vmtype_df, on='vmTypeId')
    
    events = load_events(args.numvms, vm_df, pickle_dir)


    if args.policy == 'hybrid':
        num_servers = simulate_hybrid(events, vm_df)
    elif args.policy == 'migration':
        num_servers = simulate_with_migration(events)
    elif args.policy == 'ebfd':
        num_servers = simulate_with_ebfd(events)
    else:
        num_servers = simulate(events, args.policy)

    with open('num_servers.txt', 'a') as f:
        f.write(f"{args.policy}, {args.numvms}: Maximum number of servers used is {str(num_servers)}" + "\n")

    print(f"Maximum number of servers used: {num_servers}")

if __name__ == "__main__":
    main()