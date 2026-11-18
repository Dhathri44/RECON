import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.packet import packetize_tiles
from src.network_simulator import NetworkSimulator, NetworkConfig
from src.retransmission import SelectiveRepeatARQ
import numpy as np

tiles={i:{"position":(0,i*16),"data":np.full((16,16),i,dtype=np.uint8)} for i in range(8)}
packets=packetize_tiles(tiles, payload_bytes=100)
network=NetworkSimulator(NetworkConfig(loss_probability=0.2,queue_capacity=50),seed=11)
stats=SelectiveRepeatARQ(network,max_retries=5).send(packets)
print(stats)