# ed25519-vanity-rs
Ed25519 vanity generator with CUDA in Rust

## Benchmark results

### GeForce RTX 4060 Max-Q

```
[0] GLOBAL STATS (1 prefix length, 0 suffix length): Found 851103 matches in 62.27s (0.000073s/match, 13668.399847 matches/sec, 13.65M ops/sec, 13.65M ops/sec/device, 0.0010M ops/match)
[0] GLOBAL STATS (2 prefix length, 0 suffix length): Found 6734 matches in 28.60s (0.004248s/match, 235.413664 matches/sec, 13.75M ops/sec, 13.75M ops/sec/device, 0.0584M ops/match)
[0] GLOBAL STATS (3 prefix length, 0 suffix length): Found 63 matches in 16.75s (0.265877s/match, 3.761134 matches/sec, 13.82M ops/sec, 13.82M ops/sec/device, 3.6742M ops/match)
[0] GLOBAL STATS (4 prefix length, 0 suffix length): Found 2 matches in 20.60s (10.299815s/match, 0.097089 matches/sec, 13.82M ops/sec, 13.82M ops/sec/device, 142.3442M ops/match)
[0] GLOBAL STATS (5 prefix length, 0 suffix length): Found 1 matches in 60.86s (60.863319s/match, 0.016430 matches/sec, 13.84M ops/sec, 13.84M ops/sec/device, 842.5308M ops/match)
```
