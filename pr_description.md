## Summary
Adds the multi-stage training Docker image (docker/Dockerfile.train) for the
CIFAR-10 classifier and verifies it end-to-end with a real training run.

## Changes
- docker/Dockerfile.train: multi-stage build (base deps -> training runtime),
  pinned dependencies in requirements/train.txt, PYTHONUNBUFFERED=1,
  ENTRYPOINT [python, src/train.py]
- src/train.py: reads hyperparameters from configs/training_config.yaml,
  logs structured JSON-lines metrics per epoch, saves checkpoints, supports
  early stopping
- configs/training_config.yaml: training hyperparameters (10 epochs,
  ResNet-18, CIFAR-10)

## Verification
Built and ran the training image locally with mounted volumes.
Training completed successfully over 10 epochs on CIFAR-10:
- Final train_accuracy: 0.8935
- Final val_accuracy: 0.8767
- best_val_loss: 0.377
- Checkpoint saved to /app/checkpoints/classifier_v1.pt after every improving epoch

Terminal log screenshot attached in this PR as evidence.
