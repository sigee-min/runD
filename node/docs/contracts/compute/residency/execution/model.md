# Execution Model

`device/residency/execution` seals one invocation formula for arbitrary
`Q>=1` without storing Q mutable snapshots. `execution::Plan` owns the page
and epoch counts, input/output materializations, optional transactional
publication capability, two-bank Host and Device regions, dirty tail,
active-mask projection, target-write intent, and Host-service boundary.

A node for epoch `e` is derived only from this formula and `e mod 2`. A caller
cannot replace its owner, `CacheKey`, access, dirty extent, future-use fact,
phase, bank, or publication generation. Plan identity hashes all these facts.
Sealing checks arithmetic, same-tier region disjointness, and representative
first/interior/tail recurrence classes before Authority changes state.

Each projected node names its mutation regions. Input names Host fill and
Device Input, Dispatch names Device Output, and Output names Host Output. A
coherent alias can elide a copy but cannot erase the mutation or its failure
quarantine.

`GraphMaterialization::boundary_extent` is the invocation-wide semantic
active-prefix identity carried only by the right-boundary cache key. It is not
a page byte count and may exceed `page_bytes` or `payload_bytes`. Interior
keys retain extent zero; the last key receives the exact sealed extent.

Plan storage and admission work are O(1) in Q. Admission validates checked
cardinality, regions, recurrence, and first/interior/tail representatives;
enumerating all `Q*3` nodes would be Host epoch orchestration even if its
retained memory were constant.
