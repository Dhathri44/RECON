import numpy as np
import cv2

from src.network_simulator import NetworkConfig, NetworkSimulator
from src.packet import packetize_tiles, reassemble_tiles
from src.retransmission import SelectiveRepeatARQ
from src.qos_scheduler import QoSScheduler
from src.image_reconstruction import reconstruct_with_inpainting


class ReconstructionTransmissionEnv:
    """
    RL environment for adaptive medical image transmission.

    Actions:
        0 = Reconstruct missing tile locally
        1 = Normal packet retransmission using ARQ
        2 = QoS-prioritized packet retransmission using ARQ

    The agent never receives the original pixels of missing tiles.

    Ground truth is used internally only to calculate reward.
    """

    def __init__(
        self,
        tiles,
        lost_tiles,
        roi_mask=None,
        seed=42,
        *,
        roi_priority_map=None,
        evaluation_roi_mask=None,
        payload_bytes=1200,
        max_retries=3
    ):

        self.tiles = tiles
        self.lost_tiles = lost_tiles

        # Only use this map if it represents information available
        # to the receiver, not a hidden ground-truth segmentation.
        self.roi_priority_map = (
            roi_priority_map if roi_priority_map is not None else roi_mask
        )

        # This mask is evaluator-only. It is never included in the state.
        self.evaluation_roi_mask = evaluation_roi_mask

        # The sender communicates one compact priority byte per tile when
        # ROI-aware QoS metadata is enabled. These bytes are modelled as
        # piggybacked tile-header metadata: counted as overhead, with no
        # separate round-trip delay.
        self.priority_metadata_bytes = (
            len(self.tiles) if self.roi_priority_map is not None else 0
        )

        self.payload_bytes = payload_bytes
        self.max_retries = max_retries

        self.rng = np.random.default_rng(seed)

        self.ids = list(lost_tiles.keys())

        if not self.ids:
            raise ValueError("lost_tiles cannot be empty")

        # Infer image dimensions from tile positions.
        max_y = 0
        max_x = 0

        for tile in tiles.values():
            y, x = tile["position"]
            h, w = tile["data"].shape[:2]

            max_y = max(max_y, y + h)
            max_x = max(max_x, x + w)

        first_tile = next(iter(tiles.values()))["data"]

        if first_tile.ndim == 2:
            self.image_shape = (max_y, max_x)
        else:
            self.image_shape = (
                max_y,
                max_x,
                first_tile.shape[2]
            )

        # Reconstruct the original image internally.
        # This is used for evaluation, never for agent observations.
        self.original_image = self._assemble_original()

        self.original_dtype = self.original_image.dtype

        if (
            np.issubdtype(self.original_dtype, np.floating)
            and self.original_image.min() >= 0
            and self.original_image.max() <= 1
        ):
            self.data_range = 1.0

        elif self.original_dtype == np.uint8:
            self.data_range = 255.0

        else:
            value_range = (
                self.original_image.max() - self.original_image.min()
            )
            self.data_range = float(value_range) or 1.0

        # Five image/context features + five network features.
        self.state_dim = 13

        self.reset()

    def _assemble_original(self):

        image = np.zeros(self.image_shape, dtype=next(
            iter(self.tiles.values())
        )["data"].dtype)

        for tile in self.tiles.values():

            y, x = tile["position"]
            data = tile["data"]

            h, w = data.shape[:2]

            image[y:y+h, x:x+w, ...] = data

        return image

    def _make_network(self):

        bandwidth = float(
            self.rng.choice([0.25, 0.5, 1.0, 5.0, 10.0])
        )

        latency = float(
            self.rng.choice([20, 50, 100, 200])
        )

        loss = float(
            self.rng.choice([0.02, 0.05, 0.10, 0.20, 0.35])
        )

        deadline = float(
            self.rng.choice([0.15, 0.30, 0.60])
        )

        config = NetworkConfig(
            bandwidth_mbps=bandwidth,
            latency_ms=latency,
            jitter_ms=5.0,
            loss_probability=loss,
            queue_capacity=100
        )

        self.network = NetworkSimulator(
            config=config,
            seed=int(self.rng.integers(0, 1_000_000))
        )

        self.deadline = deadline

    def reset(self):

        self.i = 0

        # Random network conditions for this episode.
        self._make_network()

        self.total_bytes = int(self.priority_metadata_bytes)
        if self.priority_metadata_bytes:
            self.network.bytes_sent += self.priority_metadata_bytes
            self.network.bytes_delivered += self.priority_metadata_bytes

        # Receiver-side image canvas.
        self.canvas = np.zeros(
            self.image_shape,
            dtype=self.original_dtype
        )

        # Tracks which pixels are actually available.
        self.available_mask = np.zeros(
            self.image_shape[:2],
            dtype=bool
        )
        # Distinguish genuine received pixels from locally reconstructed
        # estimates. Only received pixels may guide later inpainting.
        self.received_mask = np.zeros(
            self.image_shape[:2],
            dtype=bool
        )

        lost_ids = set(self.lost_tiles.keys())

        # Initially received tiles are available to the receiver.
        self.received_tiles = {
            tid: tile
            for tid, tile in self.tiles.items()
            if tid not in lost_ids
        }

        for tile in self.received_tiles.values():

            y, x = tile["position"]
            data = tile["data"]

            h, w = data.shape[:2]

            self.canvas[y:y+h, x:x+w, ...] = data
            self.available_mask[y:y+h, x:x+w] = True
            self.received_mask[y:y+h, x:x+w] = True

        # Ground-truth missing map is retained only for offline evaluation.
        self.initial_missing_mask = ~self.received_mask.copy()
        return self._state()

    def _normalise_image(self, image):

        image = np.asarray(image, dtype=np.float32)

        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.data_range > 0:
            image = image / self.data_range

        return np.clip(image, 0.0, 1.0)

    def _context_features(self, tile):

        y, x = tile["position"]
        h, w = tile["data"].shape[:2]

        margin_y = h
        margin_x = w

        y1 = max(0, y - margin_y)
        x1 = max(0, x - margin_x)

        y2 = min(self.image_shape[0], y + h + margin_y)
        x2 = min(self.image_shape[1], x + w + margin_x)

        context = self.canvas[y1:y2, x1:x2]
        known = self.received_mask[y1:y2, x1:x2]

        gray = self._normalise_image(context)

        known_values = gray[known]

        if known_values.size == 0:

            mean_value = 0.0
            std_value = 0.0
            edge_value = 0.0

        else:

            mean_value = float(np.mean(known_values))
            std_value = float(np.std(known_values))

            gx = cv2.Sobel(
                gray, cv2.CV_32F, 1, 0, ksize=3
            )

            gy = cv2.Sobel(
                gray, cv2.CV_32F, 0, 1, ksize=3
            )

            edge = np.sqrt(gx * gx + gy * gy)

            edge_value = float(np.mean(edge[known]))

        # Known ROI priority metadata.
        roi_fraction = 0.0

        if self.roi_priority_map is not None:

            roi = self.roi_priority_map[y:y+h, x:x+w]

            roi_fraction = float(
                np.mean(np.asarray(roi) > 0)
            )

        # Fraction of known pixels in the surrounding context.
        context_confidence = float(np.mean(known))

        # Examine a one-pixel ring around the missing tile.
        ring = np.zeros(
            self.image_shape[:2],
            dtype=bool
            )

        y_start = max(0, y - 1)
        y_end = min(self.image_shape[0], y + h + 1)

        x_start = max(0, x - 1)
        x_end = min(self.image_shape[1], x + w + 1)

        ring[y_start:y_end, x_start:x_end] = True
        ring[y:y+h, x:x+w] = False

        boundary_pixels = self.received_mask[ring]

        boundary_confidence = (
            float(np.mean(boundary_pixels))
            if boundary_pixels.size > 0
            else 0.0
        )

        return [
            mean_value,
            std_value,
            edge_value,
            roi_fraction,
            context_confidence,
            boundary_confidence
        ]

    def _state(self):

        if self.i >= len(self.ids):
            return np.zeros(self.state_dim, dtype=np.float32)

        tid = self.ids[self.i]
        tile = self.lost_tiles[tid]

        image_features = self._context_features(tile)

        remaining = (
            len(self.ids) - self.i
        ) / max(1, len(self.ids))

        config = self.network.config

        bandwidth = config.bandwidth_mbps
        latency = config.latency_ms
        loss = config.loss_probability

        # Estimate the number of packets required for this tile.
        tile_bytes = tile["data"].nbytes

        packet_count = max(
            1,
            int(np.ceil(tile_bytes / self.payload_bytes))
        )

        # Approximate expected number of attempts.
        # Includes data packet and ACK loss.
        success_probability = max(
            (1.0 - loss) ** 2,
            1e-6
        )

        expected_attempts = 1.0 / success_probability

        # Approximate expected transmission bytes.
        expected_bytes = (
            packet_count
            * (self.payload_bytes + 3)
            * expected_attempts
        )

        # Account for the smaller final packet.
        actual_payload_bytes = tile_bytes

        expected_bytes = (
            packet_count * 3
            + actual_payload_bytes
        ) * expected_attempts

        # Approximate expected data + ACK delay.
        serialization_ms = (
            (actual_payload_bytes + packet_count * 3) * 8
            / (max(bandwidth, 1e-6) * 1e6)
            * 1000
        )

        expected_delay_ms = (
            packet_count * 2 * latency
            + serialization_ms
        ) * expected_attempts

        # Normalize estimates.
        expected_delay_ratio = min(
            expected_delay_ms / max(self.deadline * 1000, 1.0),
            5.0
        )

        expected_byte_ratio = min(
            expected_bytes / max(tile_bytes, 1),
            5.0
        )

        network_features = [

            bandwidth / 10.0,

            latency / 200.0,

            loss,

            self.deadline / 0.60,

            remaining,

            expected_delay_ratio / 5.0,

            expected_byte_ratio / 5.0

        ]

        return np.asarray(
            image_features + network_features,
            dtype=np.float32
        )

    def _inpaint_current_tile(self, tile):
        """Recover using only genuine receiver-side pixels.

        Previously reconstructed tiles are deliberately excluded from the
        observation mask so their errors cannot cascade into later fills.
        """
        if self.original_dtype == np.uint8:
            receiver_image = self.canvas.copy()
        else:
            receiver_image = self._normalise_image(self.canvas)

        missing_mask = (~self.received_mask).astype(np.uint8) * 255
        reconstructed = reconstruct_with_inpainting(
            receiver_image,
            missing_mask,
            method="hybrid",
            pyramid_levels=3,
            refine_iterations=80,
            lam=0.10,
            prior_weight=0.35
        )

        y, x = tile["position"]
        h, w = tile["data"].shape[:2]
        patch = reconstructed[y:y+h, x:x+w, ...]

        if self.original_dtype == np.uint8:
            return patch.astype(np.uint8)
        if self.data_range == 1.0:
            return np.clip(patch, 0.0, 1.0).astype(self.original_dtype)
        return (patch * self.data_range).astype(self.original_dtype)

    def _quality_score(self, reconstructed, original):

        reconstructed = np.asarray(
            reconstructed, dtype=np.float32
        )

        original = np.asarray(
            original, dtype=np.float32
        )

        mse = float(
            np.mean((reconstructed - original) ** 2)
        )

        normalised_mse = mse / (
            self.data_range ** 2 + 1e-12
        )

        return float(
            1.0 / (1.0 + normalised_mse)
        )

    def _roi_quality_score(self, reconstructed, original, tile):

        if self.evaluation_roi_mask is None:
            return None

        y, x = tile["position"]
        h, w = tile["data"].shape[:2]

        roi = np.asarray(
            self.evaluation_roi_mask[y:y+h, x:x+w]
        ) > 0

        if not np.any(roi):
            return None

        a = np.asarray(original, dtype=np.float32)
        b = np.asarray(reconstructed, dtype=np.float32)

        if a.ndim == 3:
            roi = np.repeat(roi[:, :, None], a.shape[2], axis=2)

        mse = float(np.mean((a[roi] - b[roi]) ** 2))

        normalised_mse = mse / (
            self.data_range ** 2 + 1e-12
        )

        return float(
            1.0 / (1.0 + normalised_mse)
        )

    def step(self, action):

        if self.i >= len(self.ids):
            raise RuntimeError(
                "Episode finished. Call reset()."
            )

        if int(action) not in (0, 1, 2):
            raise ValueError("Action must be 0, 1, or 2.")

        tid = self.ids[self.i]
        tile = self.lost_tiles[tid]

        original_patch = tile["data"]

        y, x = tile["position"]
        h, w = original_patch.shape[:2]

        bytes_before = self.network.bytes_sent
        time_before = self.network.time_s

        fallback_used = False

        # ACTION 0: LOCAL RECONSTRUCTION
        if int(action) == 0:

            reconstructed_patch = (
                self._inpaint_current_tile(tile)
            )

        # ACTION 1: PACKET RETRANSMISSION
        else:

            # The original tile is available at the sender.
            # It is NOT exposed to the DQN observation.
            # Action 1 uses normal priority; Action 2 marks packets as
            # high-priority and passes them through the QoS scheduler.
            packet_priority = 2 if int(action) == 2 else 1
            packets = packetize_tiles(
                {tid: tile},
                payload_bytes=self.payload_bytes,
                priority=packet_priority
            )
            if int(action) == 2:
                packets = QoSScheduler().schedule(packets)

            arq = SelectiveRepeatARQ(
                network=self.network,
                max_retries=self.max_retries,
                seed=int(self.rng.integers(0, 1_000_000)),
                simulate_ack=True
            )

            transmission_result = arq.send(packets)

            recovered_tiles = reassemble_tiles(
                packets,
                transmission_result["acknowledged_ids"]
            )

            if tid in recovered_tiles:

                reconstructed_patch = (
                    recovered_tiles[tid]["data"]
                )

            else:

                # If retransmission fails, fall back to inpainting.
                reconstructed_patch = (
                    self._inpaint_current_tile(tile)
                )

                fallback_used = True

        # Update receiver-side canvas.
        self.canvas[y:y+h, x:x+w, ...] = reconstructed_patch
        self.available_mask[y:y+h, x:x+w] = True
        # Local estimates are displayable output, not received evidence.
        # Only successful ARQ delivery becomes trusted receiver data.
        if int(action) in (1, 2) and not fallback_used:
            self.received_mask[y:y+h, x:x+w] = True

        # Ground truth is used only for reward calculation.
        quality = self._quality_score(
            reconstructed_patch,
            original_patch
        )

        roi_quality = self._roi_quality_score(
            reconstructed_patch,
            original_patch,
            tile
        )

        bytes_sent = (
            self.network.bytes_sent - bytes_before
        )
        metadata_charge = (
            self.priority_metadata_bytes if self.i == 0 else 0
        )
        bytes_sent += metadata_charge

        delay_ms = (
            self.network.time_s - time_before
        ) * 1000.0

        self.total_bytes += bytes_sent

        # -----------------------------------------
        # REWARD FUNCTION
        # -----------------------------------------

        # 1. Image reconstruction quality.
        quality_reward = 10.0 * quality

        # 2. Bandwidth consumption.
        # Express transmitted bytes relative to original tile size.
        byte_ratio = bytes_sent / max(original_patch.nbytes, 1)

        byte_penalty = 0.05 * min(
            byte_ratio,
            4.0
        )

        # 3. Network delay relative to the deadline.
        delay_ratio = delay_ms / max(
            self.deadline * 1000.0,
            1.0
        )

        delay_penalty = 0.15 * min(
            delay_ratio,
            3.0
        )

        # 4. Failed retransmission penalty.
        failure_penalty = (
            2.0 if fallback_used else 0.0
        )

        # 5. ROI reconstruction quality.
        roi_reward = 0.0

        if roi_quality is not None:
            roi_reward = 2.0 * roi_quality

        # Final reward.
        reward = (
            quality_reward
            - byte_penalty
            - delay_penalty
            - failure_penalty
            + roi_reward
        )

        self.i += 1

        done = self.i >= len(self.ids)

        next_state = self._state()

        info = {
            "tile_id": tid,
            "action": int(action),
            "action_name": {0: "Local reconstruction", 1: "Normal retransmission", 2: "QoS-prioritized retransmission"}[int(action)],
            "packet_priority": (0 if int(action) == 0 else (2 if int(action) == 2 else 1)),
            "quality": quality,
            "roi_quality": roi_quality,
            "bytes_sent": bytes_sent,
            "priority_metadata_bytes": metadata_charge,
            "delay_ms": delay_ms,
            "fallback_used": fallback_used
        }

        return next_state, float(reward), done, info