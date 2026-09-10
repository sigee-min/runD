#include "reduce.hpp"

#include "../../device/residency/pool.hpp"
#include "../../pipeline/run/clock.hpp"
#include "../backing.hpp"

#include <kernel/program/compute/reduce/model.hpp>
#include <rund/counter.hpp>

#include <algorithm>
#include <cstring>
#include <limits>
#include <span>

namespace rund::compute::detail {
namespace {

using ::rund::detail::counter::Accumulate;

template <class T>
[[nodiscard]] T load(const std::byte *const source) noexcept {
  T value{};
  std::memcpy(&value, source, sizeof(value));
  return value;
}

template <class T>
void merge_extreme(VirtualReduction &reduction,
                   const std::byte *const source) noexcept {
  const T part = load<T>(source);
  if (!reduction.has_value) {
    std::memcpy(reduction.value.data(), &part, sizeof(part));
    reduction.has_value = true;
    return;
  }
  const T current = load<T>(reduction.value.data());
  const T merged =
      reduction.operation == static_cast<std::uint32_t>(kernel::ReduceOp::Min)
          ? std::min(current, part)
          : std::max(current, part);
  std::memcpy(reduction.value.data(), &merged, sizeof(merged));
}

template <class T>
void merge_unsigned(VirtualReduction &reduction,
                    const std::byte *const source) noexcept {
  reduction.total += static_cast<unsigned __int128>(load<T>(source));
}

[[nodiscard]] Status merge(VirtualReduction &reduction,
                           const std::byte *const source) noexcept {
  const bool additive =
      reduction.operation ==
          static_cast<std::uint32_t>(kernel::ReduceOp::Sum) ||
      reduction.operation ==
          static_cast<std::uint32_t>(kernel::ReduceOp::CountNonzero);
  if (additive) {
    if (reduction.type == Type::U32) {
      merge_unsigned<std::uint32_t>(reduction, source);
      return Status::success();
    }
    if (reduction.type == Type::U64) {
      merge_unsigned<std::uint64_t>(reduction, source);
      return Status::success();
    }
    return Status::fail(Reason::PrimitiveUnsupported);
  }
  switch (reduction.type) {
  case Type::I32:
  case Type::FixedLane32:
    merge_extreme<std::int32_t>(reduction, source);
    return Status::success();
  case Type::U32:
    merge_extreme<std::uint32_t>(reduction, source);
    return Status::success();
  case Type::I64:
  case Type::FixedLane64:
    merge_extreme<std::int64_t>(reduction, source);
    return Status::success();
  case Type::U64:
    merge_extreme<std::uint64_t>(reduction, source);
    return Status::success();
  }
  return Status::fail(Reason::PrimitiveUnsupported);
}

template <class T>
[[nodiscard]] Status encode_total(VirtualReduction &reduction) noexcept {
  if (reduction.total >
      static_cast<unsigned __int128>(std::numeric_limits<T>::max())) {
    return Status::fail(
        reduction.operation ==
                static_cast<std::uint32_t>(kernel::ReduceOp::CountNonzero)
            ? Reason::ReduceCountOverflow
            : Reason::ReduceSumOverflow);
  }
  const T value = static_cast<T>(reduction.total);
  std::memcpy(reduction.value.data(), &value, sizeof(value));
  reduction.has_value = true;
  return Status::success();
}

} // namespace

Status begin_virtual_reduction(const VirtualRunProjection &run,
                               VirtualReduction &reduction) noexcept {
  reduction =
      VirtualReduction{.operation = run.operation, .type = run.output_type};
  if (!run.reduction || run.output_payload_bytes == 0u ||
      run.output_payload_bytes > reduction.value.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const auto operation = static_cast<kernel::ReduceOp>(run.operation);
  if (operation == kernel::ReduceOp::Sum ||
      operation == kernel::ReduceOp::CountNonzero) {
    return run.output_type == Type::U32 || run.output_type == Type::U64
               ? Status::success()
               : Status::fail(Reason::PrimitiveUnsupported);
  }
  return operation == kernel::ReduceOp::Min ||
                 operation == kernel::ReduceOp::Max
             ? Status::success()
             : Status::fail(Reason::PrimitiveUnsupported);
}

Status consume_virtual_reduction(const VirtualRunProjection &run,
                                 const residency::EpochLease lease,
                                 VirtualReduction &reduction) noexcept {
  if (!run.reduction || lease.bindings.empty()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  for (const residency::CacheBinding &binding : lease.bindings) {
    const std::byte *const frame =
        virtual_resident_output_frame(run, binding.frame);
    if (frame == nullptr) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const Status status = merge(reduction, frame);
    if (!status) {
      return status;
    }
  }
  return Status::success();
}

Status
finish_virtual_reduction(VirtualBacking &output,
                         const VirtualRunProjection &run,
                         VirtualReduction &reduction, ResidencyStats &stats,
                         ::rund::node::hash_detail::Fnv &output_hash) noexcept {
  const auto operation = static_cast<kernel::ReduceOp>(reduction.operation);
  if ((operation == kernel::ReduceOp::Min ||
       operation == kernel::ReduceOp::Max) &&
      !reduction.has_value) {
    return Status::fail(Reason::ReduceCountZero);
  }
  if (operation == kernel::ReduceOp::Sum ||
      operation == kernel::ReduceOp::CountNonzero) {
    const Status encoded = reduction.type == Type::U32
                               ? encode_total<std::uint32_t>(reduction)
                               : encode_total<std::uint64_t>(reduction);
    if (!encoded) {
      return encoded;
    }
  }
  const std::size_t bytes = static_cast<std::size_t>(run.output_payload_bytes);
  VirtualBackingAccess::require_recovery(output, bytes);
  const VirtualWrite write{
      .offset = 0u,
      .bytes = std::span<const std::byte>{reduction.value.data(), bytes},
  };
  const std::uint64_t started = pipeline_clock();
  const Status status =
      output.write_batch(std::span<const VirtualWrite>{&write, 1u});
  Accumulate(stats.backing_io_ns, pipeline_clock() - started);
  if (!status) {
    return status;
  }
  VirtualBackingAccess::publish_write(output);
  output_hash.Bytes(
      reinterpret_cast<const std::uint8_t *>(reduction.value.data()), bytes);
  Accumulate(stats.backing_write_bytes, bytes);
  Accumulate(stats.page_out_bytes, bytes);
  Accumulate(stats.page_out_count, 1u);
  return Status::success();
}

} // namespace rund::compute::detail
