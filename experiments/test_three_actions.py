"""Smoke test for all three RECON actions and QoS priority scheduling."""
import numpy as np
from src.image_tiling import divide_into_tiles
from src.rl_environment import ReconstructionTransmissionEnv
from src.dqn_agent import DQNAgent
from src.qos_scheduler import QoSScheduler
from src.packet import Packet


def main():
    rng = np.random.default_rng(17)
    image = rng.random((128, 128), dtype=np.float32)
    tiles = divide_into_tiles(image, tile_size=32)
    lost = {0: tiles[0]}
    roi = np.zeros((128, 128), dtype=np.uint8)
    roi[:32, :32] = 1

    for action, expected in enumerate(("Local reconstruction", "Normal retransmission", "QoS-prioritized retransmission")):
        env = ReconstructionTransmissionEnv(
            tiles, lost, roi_mask=roi, roi_priority_map=roi,
            evaluation_roi_mask=roi, seed=31
        )
        state = env.reset()
        assert state.shape == (13,)
        _, _, _, info = env.step(action)
        assert info["action_name"] == expected
        assert info["packet_priority"] == (0 if action == 0 else action)
        print(f"PASS action {action}: {expected}")

    agent = DQNAgent(state_dim=13, action_dim=3, device="cpu")
    assert agent.q.net[-1].out_features == 3
    print("PASS DQN output layer has 3 actions")

    packets = [
        Packet(10, 1, 0, 1, b"normal", priority=1),
        Packet(11, 1, 1, 2, b"roi", priority=2),
        Packet(12, 1, 2, 2, b"best-effort", priority=0),
    ]
    scheduled = QoSScheduler().schedule(packets)
    assert [p.priority for p in scheduled] == [2, 1, 0]
    print("PASS QoS scheduler orders priority 2 before 1 before 0")


if __name__ == "__main__":
    main()
