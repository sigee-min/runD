# Host Raw Byte Hash Evidence

## Contract Under Test

Public host bytes, network completed prefixes, replay payloads, and replay
chunks use the private seeded XXH3-64 owner in
`/node/src/host/hash/bytes.cpp`. Structured field identities remain separate.
The pinned xxHash 0.8.3 header SHA-256 is
`17973c0dc49d9854ca26caa191f0e12f7a424b68858d9a78de3860d959d85e4b`.

The one-shot seed is `0x72756e642e627974`. The invalid null/nonzero-length
domain seed is `0x72756e642e6e756c`. Streaming and one-shot hashing must
produce identical output for the same ordered bytes, including updates split
at 64, 128, 240, and 241 byte boundaries. The host contract fixes independent
goldens and verifies zero-length updates.

## Replay Capture

The live scheduler computes one private `Capture` at the canonical input or
host-I/O boundary. The payload store accepts only that capture; there is no
parallel `(bytes, supplied hash)` admission API and therefore no second owner
that can re-scan or disagree with the event hash.

For chunks `C[0..m)`, payload `P`, and stable field mixer `F`, replay identity
is framed as

```text
record = F(role, identity fields, source fields, |P|, H(P), m,
           (H(C[0]), |C[0]|), ..., (H(C[m-1]), |C[m-1]|))
archive = F(record count, record[0], ..., record[n-1])
```

Input-only source fields are absent from host-record framing. Ordered chunk
hashes and lengths bind split, order, duplication, and truncation. A payload no
larger than the fixed 64 KiB chunk uses `H(P)` as its only chunk hash, so live
admission performs exactly one content-hash pass before encoding and exact
deduplication. Larger payloads retain the full payload hash and independently
hash each chunk; both byte walks are required by the two recorded identities.
Record and archive construction then cost `O(m)` field mixes and no payload
byte walk.

XXH3-64 remains evidence identity, not adversarial authentication. Its final
codomain has 64 bits; under the uniform-hash model the best possible pairwise
collision probability is `2^-64`. Deduplication never trusts that probability:
candidate chunks still require exact byte equality. Resident archives validate
encoded bytes during load. Spill archives validate sealed structure during
load and segment bytes on access.

The common `m = 1` store path also bypasses multi-chunk staging containers.
Its piece reference lives inline in the record; only `m > 1` creates dynamic
piece storage and the within-record deduplication index. Thus a repeated
single-chunk payload needs no transient staging allocation, while new payload
bytes allocate only their retained encoding and persistent indices.

## Throughput Measurement

Measurement date: 2026-07-18. Platform: Apple M4 Pro, macOS 26.3 build 25D125,
arm64, Apple Clang 21.0.0. The command compiled an ignored `.cache` harness and
the production `bytes.cpp` together with C++20, `-O3`, and `-DNDEBUG`, then ran
the process three times. The harness mutated the first byte before every call,
folded every result through a volatile sink, and used 2,000,000 iterations for
64 bytes, 200,000 for 1,500 bytes, and 5,000 for 65,536 bytes. Its SHA-256 was
`62f620ba5a17978831e8adb718234b3db1178a8151437f91a3827504a248c00d`.

`one-shot` calls the production contiguous owner. `stream` constructs one
production state and updates spans of 1, 63, and the remaining bytes. Values
are medians of the three process runs.

| Bytes | XXH3 one-shot ns | XXH3 stream ns |
| ---: | ---: | ---: |
| 64 | 3.261 | 20.591 |
| 1,500 | 35.542 | 106.683 |
| 65,536 | 1,460.517 | 3,643.733 |

These numbers establish the bottleneck direction on this host, not a portable
latency guarantee. The single-slice network fast path selects one-shot hashing;
multi-slice inputs pay state initialization and descriptor traversal without an
allocation or gather copy. Cache
state, power state, compiler, architecture, and slice count can change absolute
latency, so release evidence must remeasure rather than copying this table.

### Network Retention Order

Measurement date: 2026-07-19. Platform: Apple M4 Pro, macOS 26.3 build 25D125,
arm64. The ignored candidate harness was compiled with Apple Clang 21.0.0
(`clang-2100.1.1.101`), C++20, `-O3`, and `-DNDEBUG`; its SHA-256 was
`3c692a2b1bd7e487463a5e5d04c8cb8f9d0333f271e79f754e73e906196f05f7`.
The measured binary SHA-256 was
`497b85e58f67814b04bb72e39ff21320c8ad997e69baaf2c540a0646d27eadd6`.
Each process run used 2,000,000 iterations for 64 bytes, 300,000 for 1,500
bytes, and 5,000 for 65,536 bytes. Values are medians of three process runs.

| Bytes | Identity only ns | Identity then copy ns | Copy then identity ns |
| ---: | ---: | ---: | ---: |
| 64 | 3.419 | 4.541 | 4.636 |
| 1,500 | 35.007 | 42.373 | 50.655 |
| 65,536 | 1,451.058 | 1,932.733 | 2,076.975 |

The default ingress path invokes payload identity once and performs no
retention copy. The table measured two flat contiguous retention candidates.
Copy-first was 2.1%, 19.5%, and 7.5% slower at 64, 1,500, and 65,536 bytes,
respectively. It did not measure vectored descriptor projection or distinguish
source reads from hot retained-ring reads.

The ingress owner now selects hash-only or hash-and-retain once, before its
payload loop. Hash-only preserves the measured one-shot path outside the
scheduler evidence mutex. The retained path projects each completed-prefix
slice once, copies it into the ring, and feeds the exact destination spans to
StableHash in canonical wrap order. For `J` slices and `P` bytes, descriptor
work falls from `2J` to `J`, source traffic falls from `2P` reads to `P` reads,
and the required hash moves its `P` reads to the just-written ring storage.
Ring writes remain `P`. Enabled capture performs that work inside serialized
host-event commit so its wall-time and contention require fresh measurement;
the table does not prove a latency improvement for the new structure. It is
host evidence, not a portable latency guarantee.

XXH3 is non-cryptographic. Deduplication and cache admission use its digest only
to select candidates and still require exact byte comparison.

## Verification

```text
tools/test/run runtime.task.host
tools/test/run runtime.task.net-vectored
tools/test/run runtime.task.replay-payload-store
tools/test/run runtime.task.replay-spill-storage
```
