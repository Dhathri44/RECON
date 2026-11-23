"""Train the 3-action DQN on real BraTS FLAIR + segmentation pairs.

Expected layout (recursive search is used):
  DATASET_ROOT/...
    BraTS..._flair.nii.gz
    BraTS..._seg.nii.gz

Segmentation is used as sender-available ROI metadata for QoS priority and
as evaluator ground truth. It is never supplied as image pixels to the agent.
Patient-level train/validation split prevents slices from one patient crossing
between splits. This script does not claim clinical validity.
"""
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)
from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path
import numpy as np
import cv2
from src.medical_data import find_brats_pairs, load_nifti_pair
from src.image_tiling import divide_into_tiles
from src.network_simulator import simulate_packet_loss
from src.rl_environment import ReconstructionTransmissionEnv
from src.dqn_agent import DQNAgent
from src.unet_inpainting import load_inpainting_model


def prepare_pair(pair, size=256):
    flair_path, seg_path = pair
    image, seg, slice_idx = load_nifti_pair(flair_path, seg_path, slice_index=None, normalize=True)
    image = cv2.resize(image.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    roi = cv2.resize((seg > 0).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    return image, roi, slice_idx


def make_env(pair, seed, tile_size=32, inpainting_model=None, inpainting_device="cpu"):
    image, roi, slice_idx = prepare_pair(pair)
    tiles = divide_into_tiles(image, tile_size=tile_size)
    _, lost = simulate_packet_loss(tiles, loss_probability=0.22, seed=seed)
    if not lost:
        # A deterministic forced-loss case keeps the episode non-empty.
        tid = int(seed % len(tiles))
        lost = {tid: tiles[tid]}
    env = ReconstructionTransmissionEnv(
        tiles=tiles, lost_tiles=lost, roi_mask=roi,
        roi_priority_map=roi, evaluation_roi_mask=roi,
        seed=seed, payload_bytes=1200, max_retries=4,
        inpainting_model=inpainting_model,
        inpainting_device=inpainting_device
    )
    return env, slice_idx

def evaluate_final_image(env):

    original = env.original_image.astype(np.float32)
    reconstructed = env.canvas.astype(np.float32)

    data_range = env.data_range

    psnr = peak_signal_noise_ratio(
        original,
        reconstructed,
        data_range=data_range
    )

    ssim = structural_similarity(
        original,
        reconstructed,
        data_range=data_range
    )

    missing = env.initial_missing_mask

    missing_mse = np.mean(
        (original[missing] - reconstructed[missing]) ** 2
    ) if np.any(missing) else np.nan

    missing_psnr = (
        float("inf") if missing_mse == 0
        else 10 * np.log10(data_range ** 2 / missing_mse)
    ) if np.any(missing) else np.nan

    roi_psnr = np.nan

    if env.evaluation_roi_mask is not None:

        roi = np.asarray(env.evaluation_roi_mask) > 0

        if np.any(roi):

            roi_mse = np.mean(
                (original[roi] - reconstructed[roi]) ** 2
            )

            roi_psnr = (
                float("inf") if roi_mse == 0
                else 10 * np.log10(data_range ** 2 / roi_mse)
            )

    return {
        "psnr": float(psnr),
        "ssim": float(ssim),
        "missing_psnr": float(missing_psnr),
        "roi_psnr": float(roi_psnr)
    }

def run_episode(agent, pair, seed, training, tile_size=32, batch_size=64,
                inpainting_model=None, inpainting_device="cpu"):
    env, slice_idx = make_env(
        pair, seed, tile_size, inpainting_model, inpainting_device
    )
    state = env.reset(); total = 0.0; losses=[]; actions={0:0,1:0,2:0}
    done=False
    while not done:
        action = agent.act(state, training=training)
        actions[action] += 1
        ns, reward, done, info = env.step(action)
        if training:
            agent.buffer.add(state, action, reward, ns, done)
            loss = agent.learn(batch_size)
            if loss is not None: losses.append(loss)
        state=ns; total += reward
    return {"reward":total,"mean_loss":float(np.mean(losses)) if losses else 0.0,
            "actions":actions,"slice_index":slice_idx,"bytes_sent":env.total_bytes}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='Root folder containing BraTS *_flair.nii[.gz] and *_seg.nii[.gz] files')
    ap.add_argument('--episodes', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tile-size', type=int, default=32)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--eval-every', type=int, default=100)
    ap.add_argument(
        '--patient-split',
        required=True,
        help='Path to patient_split.json generated during U-Net training'
    )
    ap.add_argument('--unet-checkpoint', default='models/best_inpainting_model.pth',
                    help='Trained U-Net checkpoint used for local reconstruction during DQN training')
    args=ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    pairs=find_brats_pairs(args.data)
    if len(pairs)<2:
        raise SystemExit(f'Found {len(pairs)} matched FLAIR/SEG pair(s). Need at least 2 different patients. Check --data and BraTS filenames.')
        # Load the exact patient split used during U-Net training.
    split_path = Path(args.patient_split)

    if not split_path.exists():
        raise SystemExit(
            f"Patient split file not found: {split_path}. "
            "Pass the patient_split.json generated during U-Net training."
        )

    split_data = json.loads(split_path.read_text())

    # Convert matched BraTS pairs into a patient-ID dictionary.
    unique = {}

    for flair_path, seg_path in pairs:
        stem = flair_path.name.replace(
            '_flair.nii.gz', ''
        ).replace('_flair.nii', '')

        unique[stem] = (flair_path, seg_path)

    # Read patient IDs from the U-Net split.
    train_ids = set(split_data['train_patients'])
    val_ids = set(split_data['validation_patients'])
    test_ids = set(split_data['test_patients'])

    # Check that the splits do not overlap.
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    # Check that every patient exists in the current dataset.
    missing_ids = (train_ids | val_ids | test_ids) - set(unique.keys())

    if missing_ids:
        raise SystemExit(
            f"{len(missing_ids)} patients from patient_split.json "
            f"were not found in the BraTS dataset. "
            f"Examples: {list(missing_ids)[:5]}"
        )

    train_pairs = [unique[pid] for pid in split_data['train_patients']]
    val_pairs = [unique[pid] for pid in split_data['validation_patients']]
    test_pairs = [unique[pid] for pid in split_data['test_patients']]

    if not train_pairs or not val_pairs or not test_pairs:
        raise SystemExit("Train, validation, and test sets must all be non-empty.")

    print(
        f"BraTS patients: {len(unique)} | "
        f"train: {len(train_pairs)} | "
        f"validation: {len(val_pairs)} | "
        f"test: {len(test_pairs)}"
    )

    print("Using the same patient split as U-Net training.")
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    if not Path(args.unet_checkpoint).exists():
        raise SystemExit(
            f"U-Net checkpoint not found: {args.unet_checkpoint}. "
            "Train/copy the U-Net first, or explicitly pass --unet-checkpoint."
        )
    unet_model = load_inpainting_model(args.unet_checkpoint, device=device)
    agent=DQNAgent(state_dim=13, action_dim=3, lr=5e-4, gamma=0.97,
                   epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.999,
                   target_update=200, seed=args.seed, device=device)
    Path('models').mkdir(exist_ok=True); Path('results').mkdir(exist_ok=True)
    history=[]
    for ep in range(1,args.episodes+1):
        pair=train_pairs[(ep-1)%len(train_pairs)]
        result=run_episode(agent,pair,args.seed+ep,True,args.tile_size,args.batch_size,
                           unet_model,device)
        history.append({'episode':ep,'reward':result['reward'],'mean_loss':result['mean_loss'],
                        'action0':result['actions'][0],'action1':result['actions'][1],
                        'action2':result['actions'][2],'bytes_sent':result['bytes_sent']})
        if ep%25==0:
            print(f"Episode {ep}/{args.episodes} | reward {np.mean([x['reward'] for x in history[-25:]]):.3f} | epsilon {agent.epsilon:.3f}")
        if ep%args.eval_every==0 or ep==args.episodes:
            vals=[]
            for j,p in enumerate(val_pairs):
                vals.append(run_episode(agent,p,args.seed+100000+ep*100+j,False,
                                        args.tile_size,args.batch_size,unet_model,device))
            print(f"  Held-out patient validation mean reward: {np.mean([v['reward'] for v in vals]):.3f}")
    # Final held-out test is evaluated once after training/model selection.
    test_rows = [
        run_episode(agent, patient, args.seed + 200000 + i, False,
                    args.tile_size, args.batch_size, unet_model, device)
        for i, patient in enumerate(test_pairs)
    ]
    test_summary = {
        "patients": len(test_rows),

        "mean_reward": float(np.mean([
        r["reward"] for r in test_rows
        ])),

        "mean_psnr": float(np.mean([
            r["psnr"] for r in test_rows
        ])),

        "mean_ssim": float(np.mean([
            r["ssim"] for r in test_rows
        ])),

        "mean_missing_psnr": float(np.nanmean([
            r["missing_psnr"] for r in test_rows
        ])),

        "mean_roi_psnr": float(np.nanmean([
            r["roi_psnr"] for r in test_rows
        ])),

        "mean_bytes_sent": float(np.mean([
            r["bytes_sent"] for r in test_rows
        ]))
    }
    agent.save('models/recon_dqn.pt')
    with open('results/brats_training_metrics.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
    metadata={'dataset':'BraTS FLAIR + segmentation','patients_total':len(pairs),
              'train_patients':[p[0].name for p in train_pairs],
              'validation_patients':[p[0].name for p in val_pairs],
              'test_patients':[p[0].name for p in test_pairs],
              'test_summary':test_summary,
              'episodes':args.episodes,'state_dim':13,'action_dim':3,
              'roi_segmentation_role':'sender-available priority metadata and evaluator-only ground truth; metadata overhead charged',
              'local_reconstruction':'trained U-Net checkpoint supplied during DQN training',
              'unet_checkpoint':str(args.unet_checkpoint),
              'clinical_validation':False}
    Path('results/brats_training_metadata.json').write_text(json.dumps(metadata,indent=2))
    print(f"Held-out test summary: {test_summary}")
    print('Saved 3-action BraTS-trained checkpoint: models/recon_dqn.pt')
    print('Saved results/brats_training_metrics.csv and results/brats_training_metadata.json')

if __name__=='__main__': main()
