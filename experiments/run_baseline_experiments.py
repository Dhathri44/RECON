import sys, csv, os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.image_tiling import load_image, divide_into_tiles, reconstruct_from_tiles
from src.network_simulator import simulate_packet_loss
from src.image_reconstruction import create_missing_mask, reconstruct_with_inpainting
from src.evaluation import evaluate_image

IMAGE_PATH="data/input_images/sample.jpg"
OUT="data/results/baseline_results.csv"
def run():
    image=load_image(IMAGE_PATH,size=(512,512))
    rows=[]
    for tile_size in (16,32,64):
        tiles=divide_into_tiles(image,tile_size)
        for loss in (0.05,0.1,0.2,0.3):
            for seed in (42,123,456,789,2026):
                received,lost=simulate_packet_loss(tiles,loss,seed)
                incomplete=reconstruct_from_tiles(received,image.shape,dtype=image.dtype)
                mask=create_missing_mask(image.shape,received)
                for method in ("telea","ns"):
                    out=reconstruct_with_inpainting(incomplete,mask,method)
                    metrics=evaluate_image(image,out,mask,data_range=255)
                    rows.append({"tile_size":tile_size,"requested_loss":loss,
                                 "realized_tile_loss":len(lost)/len(tiles),
                                 "seed":seed,"method":method,**metrics})
    os.makedirs("data/results",exist_ok=True)
    with open(OUT,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("Saved",OUT)
if __name__=="__main__": run()
