# Native Bench

This folder is for low-level native microbenchmarks.

Current scope:

- CPU-only host-runtime microbenches
- ownership and residency bookkeeping cost
- bounded decode-state update cost

The first benchmark here is:

- `bounded_runtime_bench.cpp`
- `aliased_restore_bench.cpp`
- `copy_on_grow_restore_bench.cpp`
- `c_api_restore_bench.cpp`
- `metal_bounded_sequence_bench.mm`

It measures the host-side cost of repeated bounded decode-state updates for the
current `Qwen3.5`-shaped profile.

It also now measures the host-side cost of repeated:

- shared-prefix restore with borrowed constant-state
- decode release
- copy-on-grow restore with optional sequence promotion
- raw `c_api` restore modes on the same prefix-backed workload
- Metal bounded append and trim over real `MTLBuffer` slices
- Metal ring-buffer append over real `MTLBuffer` slices
- direct segmented-window consumers against `linearize + consume`

Interpretation note:

- the Metal window microbench is intentionally light
- if another model is active on the same machine, read the ns as directional
- keep the relative ordering, not the exact number, as the useful signal

Example:

```bash
clang++ -std=c++20 -I native/include native/src/c_api.cpp native/bench/bounded_runtime_bench.cpp -o /tmp/bounded_runtime_bench
/tmp/bounded_runtime_bench
clang++ -std=c++20 -I native/include native/bench/aliased_restore_bench.cpp -o /tmp/aliased_restore_bench
/tmp/aliased_restore_bench
clang++ -std=c++20 -I native/include native/bench/copy_on_grow_restore_bench.cpp -o /tmp/copy_on_grow_restore_bench
/tmp/copy_on_grow_restore_bench
clang++ -std=c++20 -I native/include native/src/c_api.cpp native/bench/c_api_restore_bench.cpp -o /tmp/c_api_restore_bench
/tmp/c_api_restore_bench
clang++ -std=c++20 -fobjc-arc -ObjC++ -I native/include native/src/c_api.cpp native/src/metal_backing_store.mm native/src/metal_buffer_ops.mm native/src/metal_bounded_sequence.mm native/src/metal_circular_sequence.mm native/src/metal_compute_runtime.mm native/src/metal_window_linearizer.mm native/src/metal_window_gather.mm native/src/metal_window_checksum.mm native/src/metal_window_score.mm native/src/metal_window_stats.mm native/bench/metal_bounded_sequence_bench.mm -framework Foundation -framework Metal -o /tmp/metal_bounded_sequence_bench
/tmp/metal_bounded_sequence_bench
/tmp/metal_bounded_sequence_bench --json
/tmp/metal_bounded_sequence_bench --json --append-bytes 16384 --iterations 1000
python3 scripts/metal_window_path_sweep.py --output-dir bench/results/2026-03-26-window-path-sweep
python3 scripts/metal_window_path_sweep.py --output-dir bench/results/2026-03-26-window-path-capacity-sweep --capacity-bytes 131072 524288 1048576 --append-bytes 4096 --iterations 500
```

The first four are host-runtime microbenches.

`metal_bounded_sequence_bench.mm` is the first light GPU-backed microbench in
this folder. It exists to keep the new byte-oriented Metal append/trim path
honest before deeper KV-aware operators exist.

It now also compares:

- the ring-buffer append path against the older compaction-based bounded update path
- the segmented-window `linearize` bridge
- direct segmented consumers (`score`, `stats`) against `linearize + consume`
