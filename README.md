# ed25519-vanity-rs
Ed25519 vanity generator with CUDA in Rust

## Benchmark results

* RTX5090: 100m operations/sec

```
[0] GLOBAL STATS (vanity_length: 1): Found 151612 matches in 11.34s (0.000075s/match, 13367.518279 matches/sec, 13.31M ops/sec, 13.31M ops/sec/device, 0.0010M ops/match)
[0] GLOBAL STATS (vanity_length: 2): Found 2635 matches in 11.45s (0.004346s/match, 230.094896 matches/sec, 13.55M ops/sec, 13.55M ops/sec/device, 0.0589M ops/match
[0] GLOBAL STATS (vanity_length: 3): Found 37 matches in 11.45s (0.309542s/match, 3.230577 matches/sec, 13.55M ops/sec, 13.55M ops/sec/device, 4.1943M ops/match)
[0] GLOBAL STATS (vanity_length: 4): Found 2 matches in 10.56s (5.282434s/match, 0.189307 matches/sec, 13.50M ops/sec, 13.50M ops/sec/device, 71.3032M ops/match)
[0] GLOBAL STATS (vanity_length: 5): Found 1 matches in 303.20s (303.196929s/match, 0.003298 matches/sec, 13.21M ops/sec, 13.21M ops/sec/device, 4005.5603M ops/match)
```