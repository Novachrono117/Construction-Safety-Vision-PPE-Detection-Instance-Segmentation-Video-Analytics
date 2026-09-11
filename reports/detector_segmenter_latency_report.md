# Detector versus segmenter - computational cost (phase 10C)

Status: **`DETECTOR_SEGMENTER_COST_BENCHMARK_COMPLETE`** - `CONTROLLED_LOCAL_HARDWARE_BENCHMARK`

This report measures only the cost half of the frozen phase 10A comparison. `FROZEN_PROTOCOL`: no model was trained, no average precision was recomputed, no spatial or association analysis was rerun, no threshold was tuned, and the holdout was not touched.

Repository commit at write time: `d62b3664e131884033ec6c0a852834a17a3e45b0`.

## 1. Benchmark objective

Phase 10A asked what the mask adds and what it costs. Phase 10B answered the first half on validation. This phase answers the second: how much latency and how much inference memory the frozen segmenter requires relative to the frozen detector, under one symmetric benchmark on one machine.

It is still not a contest. D2 emits a class, a confidence and a box; S1 emits those plus an instance mask. `winner_declared: false`, `combined_latency_score: false`, and both are refused by the validator rather than merely discouraged.

## 2. Frozen phase 10A protocol

| Field | Value |
| --- | --- |
| Protocol | `DETECTOR_VERSUS_SEGMENTER_CONTROLLED_COMPARISON` |
| Configuration | `configs/detector_segmenter_comparison.yaml` |
| Protocol fingerprint | `d92a157637baf52f9a7248bf24257d5a177e8496b7727ffb5afc874ab96d1c4d` |
| Protocol adapted after results | `false` |

Every frozen quantity the benchmark obeys - batch, resolution, precision, warmup count, timed repetitions, benchmark membership, execution order, synchronisation and the two timing boundaries - is read from that configuration rather than restated as a constant in the runner, and the run aborts if any of them fails to take effect.

**One gap in the frozen protocol is recorded rather than papered over.** `latency_protocol` declares no confidence threshold. `LIMITATION`: The frozen latency_protocol declares batch, resolution, precision, warmup, repetitions, membership and execution order, but no confidence threshold. The operational block is the only frozen operating point - the protocol itself states that the AP block's 0.001 is deliberately not one, because average precision needs the low-scoring tail - so the benchmark runs at the operational 0.25. Recorded as a gap in the frozen protocol and settled before any timing existed; latency at 0.001 was not measured and is not claimed, and it would differ, because a lower threshold pushes more candidates through NMS and, for the segmenter, more masks through reconstruction.

## 3. Frozen model identities

| | Detector | Segmenter |
| --- | --- | --- |
| Experiment | `D2` | `S1` |
| Architecture | `YOLO11n` | `YOLO11n-seg` |
| Input resolution | 768 | 768 |
| Checkpoint SHA-256 | `0466f872a9de22898c70d8834cf1fcfb3d77f6c9fb27e81ee6248d3d19cbc206` | `29337d671459d0f742fb713613cc41d0eafcb8a21f5846ac0da65b2ffc024f20` |
| Trained in this phase | `false` | `false` |

Both were resolved by digest through their freeze accessors, not by path, and the binary on disk was re-hashed before anything ran. The segmenter also carries `overlap_mask: false` and `mask_ratio: 4`, which is part of its identity rather than a detail: S0 and S1 are the same architecture at the same size and their checkpoints are the same number of bytes, so only the digest separates them.

## 4. Hardware and runtime

| Field | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU |
| Compute capability | 12.0 |
| Device memory | 7.959534 GiB |
| Driver | 610.88 |
| CUDA runtime | 12.8 |
| torch | 2.11.0+cu128 |
| ultralytics | 8.4.138 |
| Python | 3.12.13 |
| Platform | Windows 11 |
| cuDNN benchmark mode | `false` |
| Power source | `AC_POWER` |
| Benchmark start | 2026-09-10T23:51:36+00:00 |
| Benchmark end | 2026-09-10T23:52:36+00:00 |

## 5. Precision parity

Status: **`EFFECTIVE_PRECISION_PARITY_VERIFIED`**, frozen intent `FP32`.

| Evidence | D2 | S1 |
| --- | --- | --- |
| `backend_fp16_flag` | `false` | `false` |
| `parameter_dtypes` | `["torch.float32"]` | `["torch.float32"]` |
| `input_dtype` | `"torch.float32"` | `"torch.float32"` |
| `autocast_enabled_during_forward` | `false` | `false` |
| `quantization_config_present` | `false` | `false` |
| `resolved_precision_value` | `32` | `32` |
| `device_type` | `"cuda"` | `"cuda"` |

`FROZEN_PROTOCOL`: this is probed, not read off the configuration. A configuration value is an intention; the evidence above is what the runtime did with the tensor that reached the network, captured by a forward pre-hook before any timing existed. A difference between the two models would have stopped the phase as `PRECISION_PROTOCOL_MISMATCH`, because a latency comparison across two precisions measures the precision.

## 6. Benchmark population

| Field | Value |
| --- | --- |
| Split | `validation` |
| Images | 20 |
| Selection rule | `STABLE_SHA256_RANK_OF_IMAGE_ID` |
| Ordered membership fingerprint | `45059c2cdda1285b4fa6dbfdcfbeac6fedcad551e551e6810b1369af90f7f162` |
| Membership artifact | `reports/detector_segmenter_latency_membership.csv` |
| Drawn from validation membership | `54e4ae8afd711314b472649b1564286d0a6d3d8f13e32c035d31402cbfa5f152` |
| Reselected | `false` |
| Images inspected to select | 0 |

The subset was verified three independent ways before anything ran: the committed order reproduces the frozen ordered fingerprint, every committed rank reproduces from its own image id, and re-deriving the frozen selection rule over the frozen validation membership returns the same twenty ids in the same order. **No image was opened to choose it** - that is the point of ranking by the digest of the identifier, and it is why the subset cannot have been picked for being easy, crowded or visually interesting.

Both models received identical decoded input: each image is decoded once with the framework's own reader (`ultralytics.utils.patches.imread`) outside every timed region, and the same array is handed to both. `identical_input_tensor_shapes: true` - the run aborts if the two models ever receive differently shaped tensors for one image, because that would make the benchmark a resolution comparison.

## 7. Deterministic execution order

| Field | Value |
| --- | --- |
| Passes | `PASS_A_DETECTOR_FIRST`, `PASS_B_SEGMENTER_FIRST` |
| Timed blocks | 80 |
| Interleaved | `true` |
| Symmetric | `true` |
| Randomized | `false` |
| Execution-plan fingerprint | `9734f8f057860a0970cce50fe6395f30d90413c3d8f7fedac9072351db54c3cb` |
| Order changed after observing timings | `false` |

Pass A walks the twenty benchmark images in the frozen order and times the detector then the segmenter on each. Pass B walks the same images in the same order and times the segmenter then the detector. Both passes feed the reported distribution, so residual thermal or ordering drift falls on both models equally instead of on whichever ran second. Running one model to completion and then the other would have measured the laptop's thermal state as much as the models.

The plan is **derived** from the frozen membership by `detector_segmenter_cost.build_execution_plan` rather than written out, so it cannot drift away from the images. Its fingerprint changes if the membership, the order or the pass structure changes, and the validator recomputes it from the executed blocks.

## 8. Warmup

| Field | Value |
| --- | --- |
| Iterations | 20 |
| Schedule | `PER_MODEL_PER_IMAGE_PER_TIMED_BLOCK` |
| Path exercised | `FULL_END_TO_END_PATH_INCLUDING_MASK_RECONSTRUCTION` |
| Outputs discarded | `true` |
| Count changed after observing timings | `false` |

The protocol freezes *how many* warmup iterations there are, not when they run. The full twenty are applied immediately before every timed block, for both models equally - declared before the run, not chosen after seeing a distribution. A new input shape can carry a one-off kernel-selection cost, and a schedule that warmed once at the start would have pushed that cost into the first timed repetition of every image.

Warmup exercises the **full end-to-end path**, mask reconstruction included, which is the wider of the two boundaries; the narrower model-inference boundary is a strict subset of it, so one warmup set covers both.

## 9. Timing methodology

| Field | Value |
| --- | --- |
| Batch | 1 |
| Input resolution | 768 |
| Precision | `FP32` (`quantize: 32`) |
| Confidence | 0.25 (`operational_inference`) |
| NMS IoU | 0.7 |
| max_det | 300 |
| Timed repetitions per block | 30 |
| Timing primitive | `time.perf_counter` |
| Synchronisation primitive | `torch.cuda.synchronize` |
| Synchronise before timed region | `true` |
| Synchronise after timed region | `true` |
| Same primitive for both models | `true` |
| Percentile convention | `NUMPY_PERCENTILE_LINEAR_INTERPOLATION` |
| Standard-deviation convention | `SAMPLE_STANDARD_DEVIATION_DDOF_1` |
| Model load inside timing | `false` |
| Disk decode inside timing | `false` |
| Saving predictions during timing | `false` |
| torch.compile / TensorRT / ONNX | `false` / `false` / `false` |

**CUDA work is asynchronous**, so every timed region is bracketed by an explicit `torch.cuda.synchronize()` on both edges, with the same primitive for both models. Without it a wall-clock reading measures how long the work took to *queue*, and whichever model dispatches faster would look faster regardless of how long it actually runs.

Both models are timed through **structurally the same framework calls**. The framework's own prediction pipeline is `preprocess`, `inference`, `postprocess`; the detector and the segmenter share the first two verbatim and differ only in which `postprocess` override runs - and `SegmentationPredictor.postprocess` is a subclass of the detector's. Nothing here benchmarks one model through a high-level API and the other through a low-level path; an inability to isolate a boundary equivalently would have stopped the phase as `LATENCY_BOUNDARY_IMPLEMENTATION_BLOCKED`.

Lazy evaluation is defeated deliberately: before each end timestamp the outputs are materialised, so the boundary times the work rather than the construction of a wrapper object.

## 10. `MODEL_INFERENCE_LATENCY` boundary

The forward pass on an already-prepared input tensor. The tensor is built outside the timer and reused across the block's repetitions, so the measurement is the model core.

| Field | Value |
| --- | --- |
| Framework call | `BasePredictor.inference` |
| Preprocessing included | `false` |
| Postprocessing included | `false` |
| Mask reconstruction included | `false` |
| Disk I/O included | `false` |
| Model load included | `false` |
| Same call for both models | `true` |

## 11. `END_TO_END_MODEL_OUTPUT_LATENCY` boundary

What a caller waits for: input preprocessing, the forward pass, NMS and postprocessing, and for the segmenter the mask reconstruction that exposes final instance masks on the original canvas.

| Field | Value |
| --- | --- |
| Framework calls | `BasePredictor.preprocess`, `BasePredictor.inference`, `DetectionPredictor.postprocess / SegmentationPredictor.postprocess` |
| Preprocessing included | `true` |
| NMS and postprocessing included | `true` |
| Segmenter mask reconstruction included | `true` |
| Mask reconstruction call | `ultralytics.utils.ops.process_mask_native` |
| Mask reconstruction site | `SegmentationPredictor.construct_result` |
| Outputs materialised before end timestamp | `true` |
| Detector boundary ends at | `CLASS_CONFIDENCE_BOX_AVAILABLE` |
| Segmenter boundary ends at | `CLASS_CONFIDENCE_BOX_AND_INSTANCE_MASK_AVAILABLE` |
| Host transfer | `DEVICE_TO_HOST_TRANSFER_OUTSIDE_BOTH_BOUNDARIES` |

**Mask reconstruction is inside the segmenter's timer, and that is a fact read from the installed source rather than an assumption.** With `retina_masks` enabled, `SegmentationPredictor.construct_result` calls `ultralytics.utils.ops.process_mask_native`, which combines the prototypes with the per-instance coefficients and upsamples the result onto the original image canvas - all inside `postprocess`, and therefore inside this boundary. The run verifies, for every benchmark image that produced an instance, that the returned masks are on the original canvas, and aborts otherwise. Excluding this work would hide precisely the cost the comparison exists to quantify.

`LIMITATION`: `DEVICE_TO_HOST_TRANSFER_OUTSIDE_BOTH_BOUNDARIES`. The frozen boundary lists preprocessing, the forward pass, NMS/postprocessing and mask reconstruction; copying the resulting tensors into host memory appears in neither its includes nor its excludes, so it is left outside the timer for **both** models. That is symmetric, but the segmenter's outputs are far larger than the detector's, so a pipeline that needs them in host memory would pay more than the figures below show. No third boundary was invented to cover it.

## 12. D2 latency

`COMPUTED_RESULT`. `MODEL_INFERENCE_LATENCY`:

| Statistic | Value (ms) |
| --- | --- |
| count | 1200 |
| mean | 6.05574 |
| median | 4.5507 |
| std | 2.412568 |
| p50 | 4.5507 |
| p90 | 9.95982 |
| p95 | 10.12162 |
| p99 | 10.514269 |
| min | 4.1543 |
| max | 11.1425 |
| images/sec from mean | 165.132585 |

`END_TO_END_MODEL_OUTPUT_LATENCY`:

| Statistic | Value (ms) |
| --- | --- |
| count | 1200 |
| mean | 9.157766 |
| median | 7.30475 |
| std | 3.002305 |
| p50 | 7.30475 |
| p90 | 13.87224 |
| p95 | 14.145295 |
| p99 | 14.500324 |
| min | 6.0559 |
| max | 16.3404 |
| images/sec from mean | 109.196937 |

## 13. S1 latency

`COMPUTED_RESULT`. `MODEL_INFERENCE_LATENCY`:

| Statistic | Value (ms) |
| --- | --- |
| count | 1200 |
| mean | 7.777487 |
| median | 5.4443 |
| std | 2.91804 |
| p50 | 5.4443 |
| p90 | 11.43956 |
| p95 | 11.59655 |
| p99 | 11.911593 |
| min | 4.9609 |
| max | 12.2581 |
| images/sec from mean | 128.576235 |

`END_TO_END_MODEL_OUTPUT_LATENCY`:

| Statistic | Value (ms) |
| --- | --- |
| count | 1200 |
| mean | 11.914757 |
| median | 9.96245 |
| std | 4.003441 |
| p50 | 9.96245 |
| p90 | 16.60673 |
| p95 | 17.240505 |
| p99 | 24.721743 |
| min | 7.7405 |
| max | 29.4292 |
| images/sec from mean | 83.929534 |

## 14. Absolute latency cost

| Boundary | D2 mean (ms) | S1 mean (ms) | Absolute delta (ms) |
| --- | --- | --- | --- |
| `MODEL_INFERENCE_LATENCY_MS` | 6.05574 | 7.777487 | **1.721747** |
| `END_TO_END_MODEL_OUTPUT_LATENCY_MS` | 9.157766 | 11.914757 | **2.756991** |

`absolute_latency_delta_ms` is the segmenter's mean minus the detector's at an identical measurement boundary. The two boundaries are never combined into a single figure.

## 15. Relative latency cost

| Boundary | Relative S1 cost | Reading |
| --- | --- | --- |
| `MODEL_INFERENCE_LATENCY_MS` | **0.284317** | S1 requires 28.43% more mean latency than D2 |
| `END_TO_END_MODEL_OUTPUT_LATENCY_MS` | **0.301055** | S1 requires 30.11% more mean latency than D2 |

`relative_latency_cost` is `(S1 mean / D2 mean) - 1`, recomputed by the validator.

## 16. Throughput

| Boundary | D2 images/sec | S1 images/sec | Throughput ratio |
| --- | --- | --- | --- |
| `MODEL_INFERENCE_LATENCY_MS` | 165.132585 | 128.576235 | **0.778624** |
| `END_TO_END_MODEL_OUTPUT_LATENCY_MS` | 109.196937 | 83.929534 | **0.768607** |

`images_per_second_from_mean` is `1000 / mean_latency_ms` at **batch 1**, derived from the mean and never from the fastest repetition: one lucky iteration describes a scheduling accident, not a rate either model sustains. `LIMITATION`: this is a batch-1 latency reciprocal, not batched throughput under load, and it must not be quoted as one.

## 17. Mask-reconstruction cost context

The end-to-end delta is **2.756991 ms** while the model-inference delta is **1.721747 ms**. The difference between those two numbers is where the extra segmentation work outside the forward pass lands: NMS over mask coefficients, prototype combination, and the upsample onto the original canvas.

`ADDITIONAL_SEGMENTATION_PIPELINE_COST` is the only label this benchmark supports for the difference. It is **not** `PURE_MASK_RECONSTRUCTION_CAUSAL_COST`: `pure_mask_reconstruction_cost_isolated: false`. YOLO11n and YOLO11n-seg differ in the mask branch of the network as well as in postprocessing, and this benchmark isolates neither from the other. The measured difference is the cost of the whole segmentation pipeline relative to the whole detection pipeline.

## 18. Inference memory

`INFERENCE_MEMORY`, batch 1 at imgsz 768 in `FP32`.

| Field | D2 | S1 |
| --- | --- | --- |
| `pre_load_allocated_bytes` | 0 | 0 |
| `baseline_allocated_bytes` | 44005888 | 46727680 |
| `baseline_reserved_bytes` | 92274688 | 106954752 |
| `peak_memory_allocated_bytes` | 78815744 | 248700928 |
| `peak_memory_reserved_bytes` | 134217728 | 318767104 |
| `peak_memory_allocated_gib` | 0.073403 | 0.231621 |
| `peak_memory_reserved_gib` | 0.125 | 0.296875 |

| Derived | Value |
| --- | --- |
| `peak_memory_allocated_delta_bytes` | 169885184 |
| `peak_memory_allocated_delta_gib` | 0.158218 |
| `peak_memory_allocated_ratio` | 3.155473 |
| `peak_memory_reserved_delta_bytes` | 184549376 |
| `peak_memory_reserved_delta_gib` | 0.171875 |
| `peak_memory_reserved_ratio` | 2.375 |

`FROZEN_PROTOCOL`: peak statistics are reset with `torch.cuda.reset_peak_memory_stats` **after** the frozen 20 warmup iterations, so the figure describes inference rather than the allocator's warmup high-water mark.

Isolation is `SINGLE_MODEL_RESIDENCY_IN_A_DEDICATED_PROCESS`, with `models_resident_during_measurement: 1`. An in-process measurement was rejected before any memory figure was published: after releasing one model with del, gc.collect() and torch.cuda.empty_cache(), torch.cuda.memory_allocated() still reported 33554432 bytes of cuBLAS workspace, which the next model measured would have been charged for. The decision follows from that structural fact, not from any model's memory being preferable. `pre_load_allocated_bytes` is recorded as the evidence that the isolation held, rather than asserted.

`LIMITATION`: `NOT_MEASURED_TRAINING_MEMORY_IS_A_DIFFERENT_QUANTITY`. S0's and S1's training peaks and the phase 8B and S1 feasibility smoke tests measured a different quantity under a different protocol; they are never compared with these figures, and `training_memory_reused: false`.

## 19. Static model complexity

`STATIC_MODEL_COMPLEXITY`, `recomputed_in_this_phase: false`.

| Field | D2 | S1 |
| --- | --- | --- |
| Parameters | 2624080 | 2843583 |
| GFLOPs | 6.673 | 9.8 |
| Layers | 319 | 204 |
| Fused parameters | - | 2835543 |
| Fused GFLOPs | - | 9.6 |
| Source method | `ULTRALYTICS_TORCH_UTILS` | `FRAMEWORK_MODEL_SUMMARY_LINE` |
| Source artifact | `reports/detection_D2_manifest.json` | `reports/segmentation_S1_result_manifest.json` |

Parameter delta: **219503**. GFLOPs delta at the reference input: **3.127**.

`LIMITATION`: `measured_at_benchmark_input_size: false`, `reference_input_size: FRAMEWORK_DEFAULT_640_NOT_THE_BENCHMARK_768`. ultralytics.utils.torch_utils.get_flops and model_info both default to imgsz=640, and both committed figures came from those paths. The benchmark runs at 768, so these are not the FLOPs of the timed configuration. The two figures also came from different framework paths in different phases. The detector's committed count is unfused, so it is paired with the segmenter's unfused count. The segmenter's fused figures are reported alongside rather than substituted. These are static architecture counts, not runtime measurements, and they must not be read as an explanation of the latency figures above.

## 20. Thermal and power limitations

`CONTROLLED_LOCAL_HARDWARE_BENCHMARK`. `LIMITATION`, and not a small one:

- Power source at benchmark time: `AC_POWER`. `power_settings_changed_by_this_phase: false` - no Windows power plan, GPU clock, fan curve, undervolt or performance mode was altered for this phase.
- `thermal_correction_applied: false`. No thermal-correction mathematics was introduced. A laptop GPU throttles, and the symmetric interleaved order is the only mitigation applied: it spreads drift across both models rather than removing it.
- These numbers are valid for this machine, this driver, this runtime and this protocol. **No claim of hardware-independent latency is made or supported.**

**The observed latency distribution is wide, and the mean alone would mislead.** `POST_HOC_HARDWARE_BEHAVIOR_DIAGNOSTIC` / `POST_HOC_DIAGNOSTIC_ONLY` - written after the benchmark ran, so it is a diagnostic and not a finding. It is computed from `ALL_VALID_OBSERVATIONS` (4800 observations), `observations_discarded: 0`, and it `replaces_frozen_statistics: false`.

| Evidence | Boundary | D2 | S1 |
| --- | --- | --- | --- |
| `mean` | inference | 6.05574 | 7.777487 |
| `mean` | end-to-end | 9.157766 | 11.914757 |
| `median` | inference | 4.5507 | 5.4443 |
| `median` | end-to-end | 7.30475 | 9.96245 |
| `mean_to_median_ratio` | inference | 1.330727 | 1.428556 |
| `mean_to_median_ratio` | end-to-end | 1.253673 | 1.195967 |
| `p90` | inference | 9.95982 | 11.43956 |
| `p90` | end-to-end | 13.87224 | 16.60673 |
| `min` | inference | 4.1543 | 4.9609 |
| `min` | end-to-end | 6.0559 | 7.7405 |
| `max` | inference | 11.1425 | 12.2581 |
| `max` | end-to-end | 16.3404 | 29.4292 |
| `timed_blocks` | inference | 40 | 40 |
| `timed_blocks` | end-to-end | 40 | 40 |
| `block_mean_min` | inference | 4.318367 | 5.118167 |
| `block_mean_min` | end-to-end | 6.33297 | 8.092287 |
| `block_mean_max` | inference | 9.98153 | 11.413863 |
| `block_mean_max` | end-to-end | 13.88759 | 22.381947 |

| Per-pass mean of block means | Boundary | D2 | S1 |
| --- | --- | --- | --- |
| `PASS_A_DETECTOR_FIRST` | inference | 5.699809 | 8.012124 |
| `PASS_B_SEGMENTER_FIRST` | inference | 6.411671 | 7.542849 |
| `PASS_A_DETECTOR_FIRST` | end-to-end | 9.252903 | 11.910657 |
| `PASS_B_SEGMENTER_FIRST` | end-to-end | 9.062628 | 11.918857 |

Quote the mean together with the median, P90 and the range. The per-block figures show that the spread separates between blocks rather than within them: a block's thirty repetitions cluster, while block means span the range given. Which level a block sits at does not follow the image, the model or the pass.

**Both models show the same kind of skew, and that is all that is claimed.** `proportionality_across_models_demonstrated: false`. Each model shows a mean above its own median and a P90 near its own maximum, and each model's figures are reported separately rather than as a ratio between the two. That two distributions share a shape is a description, not a demonstration that any mechanism scales them proportionally, and no such claim is made.

**`LIMITATION` `NO_SYNCHRONIZED_PER_OBSERVATION_POWER_STATE_TELEMETRY`.** `telemetry_synchronous_with_timed_blocks: false`. No GPU clock, P-state, utilisation, temperature or power reading was sampled during or between the timed regions. Sampling one inside a timed region would have added its own cost to the measurement; sampling one outside could not be mapped to an individual observation. The consequence is recorded rather than worked around: no latency observation can be attributed to a device state.

So `causal_attribution: UNKNOWN` and the mechanism stays `UNTESTED_HYPOTHESIS`. The shape is consistent with mobile-GPU DVFS and power-state behaviour on this laptop: a block's thirty repetitions cluster tightly, while different blocks sit at different levels independently of the image, the model and the pass. That is a hypothesis about a class of mechanism, not a measurement of one. No clock, P-state, utilisation, temperature or power reading was taken during the timed regions, so nothing here maps an observation to a device state, and no alternative explanation was excluded. Mobile-GPU DVFS and power-state behaviour contributes to the observed latency distribution on this machine; which observation sat at which device state is not something this benchmark can say.

**Nothing in the protocol was adapted after the timings were seen**, which is exactly when a frozen protocol earns its keep:

| After observing the distribution | Value |
| --- | --- |
| `benchmark_subset_changed` | `false` |
| `warmup_iterations_changed` | `false` |
| `timed_repetitions_changed` | `false` |
| `execution_order_changed` | `false` |
| `precision_changed` | `false` |
| `batch_changed` | `false` |
| `imgsz_changed` | `false` |
| `timing_boundaries_changed` | `false` |
| `observations_removed` | `false` |
| `outlier_rejection_introduced` | `false` |
| `observations_normalized_or_rescaled` | `false` |
| `power_clock_or_fan_setting_changed` | `false` |
| `benchmark_rerun` | `false` |
| `statistical_definitions_changed` | `false` |

The distribution turned out to be untidy, which is exactly when a frozen protocol earns its keep. The benchmark subset, the 20 warmup iterations, the 30 timed repetitions, the symmetric interleaved order, FP32, batch 1, imgsz 768 and both timing boundaries are all as frozen in phase 10A. No observation was removed, no outlier rule was introduced, no timing was normalised or rescaled, no power, clock or fan setting was touched, and the benchmark was not re-run to obtain a tidier result.

The symmetric interleaved order was the one mitigation the frozen protocol provided, and the per-pass table above is what there is to say about how it behaved. It spreads the machine's state across both models rather than removing it: the means in sections 14 to 16 carry real machine variance, and a difference of this size measured on a quieter machine could look different.

## 21. Relationship to phase 10B

Phase 10B's numbers are referred to here only as context, and **none was recomputed**: `ap_metrics_recomputed: false`, `spatial_analysis_rerun: false`, `association_analysis_rerun: false`, `mask_iou_diagnostic_rerun: false`.

For the benefit half of the trade-off, read `reports/detector_segmenter_validation_comparison.md`. Its committed reading stands unchanged: on canonical box localisation S1 retains broadly similar capability to D2, the positive all-class delta is carried by the highly uncertain `vest_loose` class, and excluding that class the descriptive support sensitivity puts S1 **below** D2. What masks demonstrably added there was representation - a third of the median predicted box is not the object - rather than new person-PPE association discovery at the frozen containment floor.

This phase adds the cost side of that same trade-off and nothing else. It changes no frozen model, no frozen protocol and no phase 10B number.

## 22. Holdout compliance

`HOLDOUT_POLICY`: **`PROTECTED_NOT_ACCESSED`**.

Phase 10C timed both frozen models on the frozen 20-image validation benchmark subset only. The holdout was not read, materialised, adapted, counted, predicted on, timed or inspected; no holdout identifier, image, prediction, timing or statistic exists in any artifact this phase wrote.

| Field | Value |
| --- | --- |
| Holdout images read | 0 |
| Holdout predictions | 0 |
| Holdout timings | 0 |
| Holdout statistics | 0 |
| `holdout_accessed` | `false` |
| Raw timings store holdout identifiers | `false` |

The runner refuses to start at all when `CSVISION_ALLOW_TEST_SPLIT` is set, and the benchmark subset is drawn from the frozen validation membership by fingerprint, so there is no path by which a holdout image could enter it.

## 23. Interpretation limits

What this phase measured, and only this:

- `LIMITATION` A laptop GPU throttles. These figures describe this machine under whatever power and thermal state it was in, and are labelled CONTROLLED_LOCAL_HARDWARE_BENCHMARK rather than presented as a property of either architecture.
- `LIMITATION` Mobile-GPU DVFS and power-state behaviour contributes to the observed latency distribution: block means span a wide range and the mean sits well above the median for both models at both boundaries. No GPU clock, P-state, utilisation or power telemetry accompanied the timed regions (NO_SYNCHRONIZED_PER_OBSERVATION_POWER_STATE_TELEMETRY), so no observation can be mapped to a device state and the cause of any feature of the distribution is UNKNOWN. Nothing was normalised, filtered or re-run in response.
- `LIMITATION` Batch 1 measures latency, not throughput under load. No batched or concurrent scenario is covered, and images_per_second is derived from the mean of a batch-1 latency rather than measured as a sustained rate.
- `LIMITATION` Neither model is exported, quantised or otherwise optimised for deployment. These are the PyTorch checkpoints as trained.
- `LIMITATION` One run of each model was ever trained, and each was timed once under this protocol. The distribution describes repetition-to-repetition variation on this machine, not run-to-run variance of training or of the benchmark itself.
- `LIMITATION` Copying the outputs to host memory is outside both boundaries, for both models alike, because the frozen boundary lists neither includes nor excludes it. The segmenter's outputs are far larger, so a pipeline that needs them in host memory would pay more than these figures show.
- `LIMITATION` The frozen latency protocol declares no confidence threshold. The benchmark uses the operational 0.25 and says so; a lower threshold would push more candidates through NMS and more masks through reconstruction, and that was not measured.
- `LIMITATION` The committed parameter and GFLOPs figures are static complexity at the framework's default 640 reference input, not at the benchmark's 768.

What it did **not** establish: that the latency difference is caused by mask reconstruction alone; that either model is faster in general or on other hardware; that either model should be preferred; or anything at all about test performance. `winner_declared: false` and `combined_latency_score: false` remain refused by both parsers, and neither frozen model changed.

## 24. Next phase

Phase 10D - the operational and scientific synthesis - reads phase 10B's benefit figures alongside this phase's cost figures and states the trade-off. It has **not** started, and no synthesis, recommendation or deployment reading is offered here.

The holdout stays locked until phase 11.

## Fingerprints

| Artifact | Fingerprint |
| --- | --- |
| Phase 10A protocol | `d92a157637baf52f9a7248bf24257d5a177e8496b7727ffb5afc874ab96d1c4d` |
| Benchmark membership (ordered) | `45059c2cdda1285b4fa6dbfdcfbeac6fedcad551e551e6810b1369af90f7f162` |
| Execution plan | `9734f8f057860a0970cce50fe6395f30d90413c3d8f7fedac9072351db54c3cb` |
| Raw timing dataset | `fce637ee2550ef814de8840d5e0a0eb6f09a46839ed0a8f0a5dbaa821c2f968c` |
| Latency result | `27c1705f15887690f575b91e17e892df8e0226f93b16a35e2b799b505fdc8813` |
| Memory result | `7f452c8a7be93b8dbdec9f89d316629522c092918dbc53420bb18e9d95132ed6` |

Raw observations: **4800** timed readings (1200 per model per boundary), row-level in `artifacts/benchmark/latency_observations.csv` (git-ignored) and summarised per block in `reports/detector_segmenter_latency_blocks.csv`.

