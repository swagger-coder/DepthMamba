# -m debugpy --listen localhost:6666 --wait-for-client 
CUDA_VISIBLE_DEVICES=7 python infer.py --encoder vits --img-path /home/ypf/workspace2/code/Depth-Anything/assets/examples --outdir ./results