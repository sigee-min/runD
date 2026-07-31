# Accel Partition Contract

Node owns resident backend execution for kernel-planned stable `Partition`
graph steps. Kernel owns the descriptor, hash, pure plan, reference law, and
stable reason vocabulary.

## Authority

Implementation authority:

- `/node/src/accel/partition/model.hpp` as the sole host parameter ABI owner
- `/node/src/accel/partition/shape.{hpp,cpp}` as the sole resident-shape owner
- `/node/src/accel/cpu/partition.cpp`
- `/node/src/accel/metal/partition/`
- `/node/src/accel/vulkan/partition/`
- `/node/src/accel/kernel/bindings/partition.cpp`

Verification authority:

- `/node/tests/contract/accel/kernel/partition.cpp`
- `/node/tests/contract/accel/kernel/partition/`
- `/node/tests/contract/accel/kernel/model.cpp`

## Contract

Partition preserves the input order within the false and true groups. The
backend classify pass produces one false flag per element, the canonical Scan
plan produces offsets, and scatter writes the false group before the true
group. CPU, Metal, and Vulkan consume the same planned element count and
resident shapes; physical lane arrival is not ordering authority.

The one-field `PartitionParams` host ABI is owned outside both native backends.
Metal and Vulkan retain only their backend-language declarations and must match
the common byte layout through the parameter-model contract. Prepared
execution uploads that same eight-byte value once; there is no backend adapter,
conversion, or second parameter representation.
