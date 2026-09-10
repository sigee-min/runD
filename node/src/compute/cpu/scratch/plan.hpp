#pragma once

#include "../graph.hpp"
#include "../state/arena.hpp"
#include "../state/primitive.hpp"

#include "../../status.hpp"

#include <kernel/core/checked.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>

namespace rund::compute::detail {

enum class CpuPrimitiveScratchShape : std::uint8_t {
  None,
  Sort,
  Scatter,
  ScatterReduce,
  Transform,
  FactorQr,
  SolveQrFactor,
  SolveLu,
  SolveCholesky,
  SolveQrMatrix,
  SpectrumEigen,
  SpectrumSvdValues,
  SpectrumSvdVectors,
  RangeI32,
  RangeU32,
  RangeI64,
  RangeU64,
};

// Frozen request descriptor shared by Pipeline preflight and the allocator.
// `counts` is shape-specific. CpuExecutionStoragePlan is the sole aggregate
// byte authority; the materializer may only consume these exact element
// counts.
struct CpuPrimitiveScratchPlan final {
  CpuPrimitiveScratchShape shape{CpuPrimitiveScratchShape::None};
  std::array<std::size_t, 5u> counts{};
  std::uint64_t host_bytes{};
  std::uint8_t element_bytes{};

  [[nodiscard]] constexpr bool
  operator==(const CpuPrimitiveScratchPlan &) const noexcept = default;
};

template <class T>
[[nodiscard]] inline bool bind_cpu_scratch_buffer(
    std::span<T> &target, const std::span<T> storage, std::size_t &cursor,
    const std::size_t count) noexcept {
  if (cursor > storage.size() || count > storage.size() - cursor) {
    return false;
  }
  target = storage.subspan(cursor, count);
  cursor += count;
  return true;
}

template <class Scratch>
[[nodiscard]] inline Result<CpuPrimitiveScratch>
prepared_cpu_scratch(Scratch *const prepared) {
  if (prepared == nullptr) {
    return Result<CpuPrimitiveScratch>::fail(Reason::BufferCapacity);
  }
  CpuPrimitiveScratch scratch{std::in_place_type<Scratch *>, prepared};
  return Result<CpuPrimitiveScratch>::success(std::move(scratch));
}

template <class Plan>
[[nodiscard]] inline const Plan *
active_cpu_scratch_plan(const CpuRuntimePrimitive &primitive) noexcept {
  const Plan *const plan = std::get_if<Plan>(&primitive.plan);
  return plan != nullptr && plan->ok ? plan : nullptr;
}

[[nodiscard]] inline bool
cpu_scratch_to_size(const kernel::u64 count, std::size_t &result) noexcept {
  if (count >
      static_cast<kernel::u64>(std::numeric_limits<std::size_t>::max())) {
    return false;
  }
  result = static_cast<std::size_t>(count);
  return true;
}

[[nodiscard]] inline bool cpu_scratch_product_size(
    const kernel::u64 left, const kernel::u64 right,
    std::size_t &result) noexcept {
  kernel::u64 product = 0u;
  if (!kernel::checked::mul(left, right, product)) {
    return false;
  }
  return cpu_scratch_to_size(product, result);
}

[[nodiscard]] inline Result<CpuPrimitiveScratchPlan>
make_cpu_scratch_plan(const CpuPrimitiveScratchShape shape,
                      const std::uint8_t element_bytes,
                      const std::array<std::size_t, 5u> counts,
                      const std::uint64_t host_bytes) noexcept {
  return Result<CpuPrimitiveScratchPlan>::success(CpuPrimitiveScratchPlan{
      .shape = shape,
      .counts = counts,
      .host_bytes = host_bytes,
      .element_bytes = element_bytes,
  });
}

[[nodiscard]] inline Result<CpuPrimitiveScratchPlan>
empty_cpu_scratch_plan() noexcept {
  return Result<CpuPrimitiveScratchPlan>::success({});
}

[[nodiscard]] inline bool scratch_shape_matches(
    const Primitive primitive,
    const CpuPrimitiveScratchShape shape) noexcept {
  switch (primitive) {
  case Primitive::Sort:
  case Primitive::Argsort:
    return shape == CpuPrimitiveScratchShape::Sort;
  case Primitive::Scatter:
    return shape == CpuPrimitiveScratchShape::Scatter;
  case Primitive::ScatterReduce:
    return shape == CpuPrimitiveScratchShape::ScatterReduce;
  case Primitive::Transform:
    return shape == CpuPrimitiveScratchShape::Transform;
  case Primitive::Factor:
    return shape == CpuPrimitiveScratchShape::None ||
           shape == CpuPrimitiveScratchShape::FactorQr;
  case Primitive::Solve:
    return shape == CpuPrimitiveScratchShape::None ||
           shape == CpuPrimitiveScratchShape::SolveQrFactor ||
           shape == CpuPrimitiveScratchShape::SolveLu ||
           shape == CpuPrimitiveScratchShape::SolveCholesky ||
           shape == CpuPrimitiveScratchShape::SolveQrMatrix;
  case Primitive::Spectrum:
    return shape == CpuPrimitiveScratchShape::SpectrumEigen ||
           shape == CpuPrimitiveScratchShape::SpectrumSvdValues ||
           shape == CpuPrimitiveScratchShape::SpectrumSvdVectors;
  case Primitive::Window:
    return shape == CpuPrimitiveScratchShape::None ||
           shape == CpuPrimitiveScratchShape::RangeI32 ||
           shape == CpuPrimitiveScratchShape::RangeU32 ||
           shape == CpuPrimitiveScratchShape::RangeI64 ||
           shape == CpuPrimitiveScratchShape::RangeU64;
  case Primitive::SegmentedScan:
  case Primitive::SegmentedReduce:
  case Primitive::Compact:
  case Primitive::Gather:
  case Primitive::Histogram:
  case Primitive::Partition:
  case Primitive::Reduce:
  case Primitive::Stencil:
  case Primitive::Matrix:
    return shape == CpuPrimitiveScratchShape::None;
  }
  return false;
}

[[nodiscard]] inline Status append_cpu_scratch_object(
    CpuExecutionStoragePlan &arena, const std::uint64_t host_bytes) noexcept {
  constexpr std::uint64_t alignment = alignof(std::max_align_t);
  std::uint64_t storage_bytes = 0u;
  if (host_bytes == 0u ||
      !kernel::checked::align_up(host_bytes, alignment, storage_bytes) ||
      storage_bytes > std::numeric_limits<std::size_t>::max() ||
      host_bytes > std::numeric_limits<std::size_t>::max() ||
      arena.primitive_object_storage_bytes >
          std::numeric_limits<std::size_t>::max() -
              static_cast<std::size_t>(storage_bytes) ||
      arena.primitive_object_payload_bytes >
          std::numeric_limits<std::size_t>::max() -
              static_cast<std::size_t>(host_bytes)) {
    return Status::fail(Reason::ProgramCapacity);
  }
  arena.primitive_object_storage_bytes += static_cast<std::size_t>(storage_bytes);
  arena.primitive_object_payload_bytes += static_cast<std::size_t>(host_bytes);
  return Status::success();
}

[[nodiscard]] inline bool add_cpu_scratch_count(std::size_t &target,
                                                 const std::size_t value) noexcept {
  if (value > std::numeric_limits<std::size_t>::max() - target) {
    return false;
  }
  target += value;
  return true;
}

} // namespace rund::compute::detail
