OMP_NUM_THREADS=2 python train.py --arch lstm \
  --env simulator/Build/simulator.app \
  --num-envs 2 --arenas 16 \
  --n-steps 128 --batch-size 2048 \
  --hidden 8 --total-steps 5000000