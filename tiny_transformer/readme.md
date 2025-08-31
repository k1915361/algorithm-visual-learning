## TODO

Training log will be saved to ./logs\training_log_2025-08-31_22-10-55.csv
Epoch 1: train_loss=2.8672, val_loss=2.9748, ppl=19.59
Epoch 2: train_loss=2.5223, val_loss=2.6428, ppl=14.05
Epoch 3: train_loss=2.3955, val_loss=2.5736, ppl=13.11
...
Epoch 49: train_loss=2.0722, val_loss=2.5700, ppl=13.07
Epoch 50: train_loss=2.0585, val_loss=2.5702, ppl=13.07
Total training time: 333.07 seconds

Generated text:
 To  aple arimimato itis lorbeabeliting log dor lunpe. What itsepen.Q: What  pop toke ite sto inof no ione Prentropition 

Training logs show that validation loss and perplexity plateau around ~13. This means the model has learned as much as it can from the current setup. Next steps you could try:

- [x] **Increase model capacity**: raise hidden size, embedding dim, 
    - [] or add more layers. This gives the network more ability to model the dataset.
- [] **Increase dataset size or variety**: with a very small dataset, the model saturates quickly. Adding more Q&A pairs (especially math/code) may lower perplexity further.
- [] **Tune regularization**: dropout rate, weight decay, or gradient clipping may be too strong/weak.
- [] **Adjust learning schedule**: try longer warmup, slower decay, or smaller final LR.
- [x] **Batch size vs. epochs**: larger batches simulate more stable updates. If you keep the model small, more epochs won’t help much once loss is flat.
- [] **Utilize hardware better**: JAX by default may not saturate CPU. Consider running on GPU (CUDA/cuDNN) to get higher utilization and tokens/sec.

In summary: plateau means you need **more capacity or more data**. If your priority is keeping the model tiny, focus on expanding/cleaning the dataset rather than just training longer.