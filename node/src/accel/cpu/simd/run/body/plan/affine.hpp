#pragma once

namespace rund::node::accel::cpu_simd_detail {
namespace {

// Dense lanes retain the original vector-by-vector read-before-write order,
// including partial tails and in-place bindings. Strided views use the generic
// runner. Fixed-point arithmetic never reaches this modular integer path.
[[nodiscard]] LoopCount ExecuteAffine(const PreparedAffineRun &affine,
                                      const CpuSimdBindingView &bindings,
                                      const u64 begin,
                                      const u64 length) noexcept {
  const Vec multiplier = RUND_CPU_SIMD_SPLAT(
      std::bit_cast<Scalar>(static_cast<BitsScalar>(affine.multiplier)));
  const Vec addend = RUND_CPU_SIMD_SPLAT(
      std::bit_cast<Scalar>(static_cast<BitsScalar>(affine.addend)));
  const auto apply = [&](Vec input) noexcept {
    return RUND_CPU_SIMD_ADD_WRAP(RUND_CPU_SIMD_MUL_LOW(input, multiplier),
                                  addend);
  };
  const auto *read =
      bindings.reads[0u].data + ByteOffset(begin, sizeof(Scalar));
  auto *write = bindings.writes[0u].data + ByteOffset(begin, sizeof(Scalar));
  const u64 full = length - length % kLaneCount;
  for (u64 offset = 0u; offset < full; offset += kLaneCount) {
    Vec input{};
    std::memcpy(&input, read, sizeof(input));
    const Vec output = apply(input);
    std::memcpy(write, &output, sizeof(output));
    read += sizeof(Vec);
    write += sizeof(Vec);
  }
  const auto tail = static_cast<std::size_t>(length - full);
  if (tail != 0u) {
    std::array<Scalar, kLaneCount> lanes{};
    std::memcpy(lanes.data(), read, tail * sizeof(Scalar));
    const Vec output = apply(RUND_CPU_SIMD_LOAD(lanes.data()));
    RUND_CPU_SIMD_STORE(lanes.data(), output);
    std::memcpy(write, lanes.data(), tail * sizeof(Scalar));
  }
  return {.tiles = length,
          .vectors = full / kLaneCount,
          .tails = tail != 0u ? 1u : 0u};
}

} // namespace
} // namespace rund::node::accel::cpu_simd_detail
