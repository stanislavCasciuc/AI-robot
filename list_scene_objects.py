"""
Run this while the simulation is open in CoppeliaSim (simulation can be stopped).
It prints all object names in the scene so you can copy the correct paths
into maze_robot.py.
"""

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim    = client.require("sim")

handles = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)
print(f"Found {len(handles)} objects:\n")
for h in handles:
    name = sim.getObjectAlias(h, -1)
    kind = sim.getObjectType(h)
    print(f"  [{kind:2}]  {name}")
