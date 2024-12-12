# VM Placement
The bin-packing simulation for virtual machine placement is designed and implemented to efficiently manage the large **Azure Trace for Packing 2020** dataset and optimize server usage.

The simulation is built to provide a flexible and efficient framework for evaluating VM placement
strategies. The Event class represents each VM’s lifecycle, encapsulating information like start time,
end time, and resource requirements. The Server class manages the physical server’s resources,
tracks hosted VMs, and calculates resource usage.  

To boost efficiency, the simulation uses the pickle module for serializing and storing processed event
data. This approach enables rapid loading of preprocessed events from disk, eliminating the need
for repeated database queries and data processing. By caching these events in a binary format, the
system significantly reduces initialization time, which is particularly advantageous when running
simulations with varying parameters.  

The simulation operates through a series of functions that coordinate the placement of VMs on
servers based on different strategies. The simulate function acts as the core driver, selecting a placement
policy based on the input and executing the corresponding logic. Each policy—such as First-Fit
Decreasing (FFD), cosine similarity, hybrid clustering, live migration, or Enhanced Best-Fit Decreasing
(EBFD)—is implemented in a modular manner, either as individual functions or dedicated
classes, allowing for easy customization and scalability.  

VMs are processed chronologically by start time within each strategy. For approaches like FFD,
VMs are sorted by size to prioritize larger VMs, maximizing space utilization. Advanced strategies,
such as hybrid clustering, employ machine learning to group similar VMs for placement. Live
migration periodically redistributes VMs across servers to optimize load balancing.  
