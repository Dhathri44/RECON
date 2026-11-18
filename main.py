import argparse
import cv2
from src.image_tiling import load_image, divide_into_tiles, reconstruct_from_tiles
from src.network_simulator import simulate_packet_loss
from src.image_reconstruction import create_missing_mask, reconstruct_with_inpainting
from src.evaluation import evaluate_image

def main():
    parser=argparse.ArgumentParser(description="RECON baseline image-loss/reconstruction demo")
    parser.add_argument("--image",default="data/input_images/sample.jpg")
    parser.add_argument("--tile-size",type=int,default=32)
    parser.add_argument("--loss",type=float,default=0.15)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--save-dir",default="data/results/demo")
    args=parser.parse_args()
    image=load_image(args.image,size=(512,512))
    tiles=divide_into_tiles(image,args.tile_size)
    received,lost=simulate_packet_loss(tiles,args.loss,args.seed)
    incomplete=reconstruct_from_tiles(received,image.shape,dtype=image.dtype)
    mask=create_missing_mask(image.shape,received)
    import os
    os.makedirs(args.save_dir,exist_ok=True)
    cv2.imwrite(os.path.join(args.save_dir,"incomplete.png"),incomplete)
    for method in ("telea","ns"):
        result=reconstruct_with_inpainting(incomplete,mask,method)
        cv2.imwrite(os.path.join(args.save_dir,f"{method}.png"),result)
        print(method,evaluate_image(image,result,mask,data_range=255))
    cv2.imwrite(os.path.join(args.save_dir,"missing_mask.png"),mask)
    print(f"Tiles={len(tiles)} received={len(received)} lost={len(lost)}")
    print("Results:",args.save_dir)
if __name__=="__main__": main()
