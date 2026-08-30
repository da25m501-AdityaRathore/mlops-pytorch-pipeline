# Reflection Write-up (300–500 words) — TEMPLATE

> This is a starting skeleton, not a finished submission. Fill it in with
> your own experience once you've actually run the pipeline end-to-end —
> the assignment asks you to be able to explain every line, and this
> reflection is where that shows.

## What was the most challenging part?

(Candidates to discuss, pick whichever were true for you:)

- Adapting ResNet-18's stem for 32x32 CIFAR-10 inputs instead of
  ImageNet-sized images, and understanding *why* the default stride-2
  stem + maxpool loses too much spatial information for small images.
- Getting the multi-stage Docker build right: separating build-time
  dependencies (compilers for wheels) from the runtime image so the
  serving image stays slim and doesn't ship training-only libraries.
- Coordinating PVC lifecycles between the training Job and the serving
  Deployment — the Job must complete and write a checkpoint before the
  Deployment's pods can pass their readiness probe.
- Debugging Kubernetes readiness/liveness probe timing (initialDelaySeconds
  vs. actual model load time) — pods restarting in a crash loop because
  the probe fired before the model finished loading.
- Getting `kubectl apply` ordering right (namespace → PVCs/ConfigMap →
  Job → wait for completion → Deployment/Service/HPA).

## What else to cover

- One concrete bug you hit and how you diagnosed it (e.g. via
  `kubectl describe pod`, `kubectl logs`, or a failed CI run).
- A design decision you'd revisit given more time (e.g. ReadWriteOnce vs.
  ReadWriteMany for the checkpoints PVC if scaling across nodes; using a
  model registry instead of a raw PVC for the checkpoint).
- What you learned about the difference between "works in Docker locally"
  and "works in Kubernetes" (resource limits, probes, config injection).

---

*Word count target: 300–500. Delete this template text before submitting.*
