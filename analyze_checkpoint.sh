python src/analyze_checkpoint_pid.py \
  --checkpoint-original saved/MASKED_GLORIA-book-seed999-Aug-12-2026-13-46-25.pth \
  --checkpoint-new saved/MASKED_GLORIA_MIPD-book-seed999-Aug-25-2026-04-52-41.pth \
  --target both \
  --estimator-seeds 999 1000 1001 \
  --device cuda:0 \
  --batch-size 256 \
  --discrim-epochs 40 \
  --ce-epochs 10 \
  --knn-chunk-size 1024