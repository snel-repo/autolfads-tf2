import time
from pprint import pprint

import ray

print(ray.__file__)
ray.init(address="localhost:10000")
pprint(ray.cluster_resources())


@ray.remote
def f():
    time.sleep(0.01)
    return ray.services.get_node_ip_address()


# Get a list of the IP addresses of the nodes that have joined the cluster.
pprint(set(ray.get([f.remote() for _ in range(1000)])))
