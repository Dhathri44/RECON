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


def prepare_pair(pair, size=256):
    flair_path, seg_path = pair
    image, seg, slice_idx = load_nifti_pair(flair_path, seg_path, slice_index=None, normalize=True)
    image = cv2.resize(image.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    roi = cv2.resize((seg > 0).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    return image, roi, slice_idx


def make_env(pair, seed, tile_size=32):
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
        seed=seed, payload_bytes=1200, max_retries=4
    )
    return env, slice_idx


def run_episode(agent, pair, seed, training, tile_size=32, batch_size=64):
    env, slice_idx = make_env(pair, seed, tile_size)
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
    args=ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    pairs=find_brats_pairs(args.data)
    if len(pairs)<2:
        raise SystemExit(f'Found {len(pairs)} matched FLAIR/SEG pair(s). Need at least 2 different patients. Check --data and BraTS filenames.')
    # Group by patient file stem to split at patient level.
    unique={}
    for f,s in pairs:
        stem=f.name.replace('_flair.nii.gz','').replace('_flair.nii','')
        unique[stem]=(f,s)
    pairs=list(unique.values()); random.shuffle(pairs)
    split=max(1,int(0.8*len(pairs)))
    if split>=len(pairs): split=len(pairs)-1
    train_pairs=pairs[:split]; val_pairs=pairs[split:]
    print(f'BraTS matched patients: {len(pairs)} | train: {len(train_pairs)} | validation: {len(val_pairs)}')
    agent=DQNAgent(state_dim=13, action_dim=3, lr=5e-4, gamma=0.97,
                   epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.999,
                   target_update=200, seed=args.seed)
    Path('models').mkdir(exist_ok=True); Path('results').mkdir(exist_ok=True)
    history=[]
    for ep in range(1,args.episodes+1):
        pair=train_pairs[(ep-1)%len(train_pairs)]
        result=run_episode(agent,pair,args.seed+ep,True,args.tile_size,args.batch_size)
        history.append({'episode':ep,'reward':result['reward'],'mean_loss':result['mean_loss'],
                        'action0':result['actions'][0],'action1':result['actions'][1],
                        'action2':result['actions'][2],'bytes_sent':result['bytes_sent']})
        if ep%25==0:
            print(f"Episode {ep}/{args.episodes} | reward {np.mean([x['reward'] for x in history[-25:]]):.3f} | epsilon {agent.epsilon:.3f}")
        if ep%args.eval_every==0 or ep==args.episodes:
            vals=[]
            for j,p in enumerate(val_pairs):
                vals.append(run_episode(agent,p,args.seed+100000+ep*100+j,False,args.tile_size,args.batch_size))
            print(f"  Held-out patient validation mean reward: {np.mean([v['reward'] for v in vals]):.3f}")
    agent.save('models/recon_dqn.pt')
    with open('results/brats_training_metrics.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
    metadata={'dataset':'BraTS FLAIR + segmentation','patients_total':len(pairs),
              'train_patients':[p[0].name for p in train_pairs],
              'validation_patients':[p[0].name for p in val_pairs],
              'episodes':args.episodes,'state_dim':13,'action_dim':3,
              'roi_segmentation_role':'sender-available ROI metadata and evaluator ground truth; not agent image input',
              'clinical_validation':False}
    Path('results/brats_training_metadata.json').write_text(json.dumps(metadata,indent=2))
    print('Saved 3-action BraTS-trained checkpoint: models/recon_dqn.pt')
    print('Saved results/brats_training_metrics.csv and results/brats_training_metadata.json')

if __name__=='__main__': main()
