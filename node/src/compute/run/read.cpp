#include "../../hash/fnv.hpp"
#include "../backend.hpp"
#include "../pipeline/claim.hpp"
#include "../program/state.hpp"
#include "../size.hpp"
#include "../status.hpp"
#include "../type.hpp"
#include "state.hpp"
#include <rund/counter.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>

namespace rund::compute::detail {
namespace {

void mix_hash(std::uint64_t &hash, const std::uint64_t value) noexcept {
  ::rund::node::hash_detail::MixU64(hash, value);
}

void record_output_hash(const RunState &run, const std::size_t output,
                        const std::uint64_t leaf,
                        const std::size_t logical_count) noexcept {
  const std::size_t output_count = run.program->output_types.size();
  run.output_hashes[output] = leaf;
  run.output_hash_counts[output] = static_cast<std::uint32_t>(logical_count);
  run.output_hash_mask |= std::uint32_t{1u} << output;
  if (output_count == 1u) {
    run.stats.output_hash = leaf;
    return;
  }
  const std::uint32_t complete =
      (std::uint32_t{1u} << output_count) - std::uint32_t{1u};
  if (run.output_hash_mask != complete) {
    run.stats.output_hash = 0u;
    return;
  }
  std::uint64_t hash = ::rund::node::hash_detail::kFnvOffset;
  mix_hash(hash, output_count);
  for (std::size_t index = 0u; index < output_count; ++index) {
    const auto &buffer = run.outputs[index];
    mix_hash(hash, index);
    mix_hash(hash, static_cast<std::uint64_t>(buffer->type));
    mix_hash(hash, run.output_hash_counts[index]);
    mix_hash(hash, run.output_hashes[index]);
  }
  run.stats.output_hash = hash;
}

[[nodiscard]] Status
read_bytes(const RunState &run, const std::shared_ptr<BufferState> &buffer,
           void *data, const std::size_t bytes, const std::size_t logical_count,
           std::uint64_t *const staging_bytes, bool *const staging_reused,
           std::uint64_t *const staging_budget, const bool destination_zeroed) {
  if (run.program == nullptr || buffer == nullptr) {
    return Status::fail(Reason::RunInvalid);
  }
  const BufferClaim claim{buffer.get(), false};
  const Status claimed =
      acquire_claims(*run.program->device, std::span{&claim, 1u});
  if (!claimed) {
    return claimed;
  }
  ClaimGuard claim_guard{*run.program->device, std::span{&claim, 1u}};
  const std::size_t output_count = run.program->output_types.size();
  std::size_t output_index = output_count;
  for (std::size_t index = 0u; index < output_count; ++index) {
    if (run.outputs[index] == buffer) {
      output_index = index;
      break;
    }
  }
  if (output_index == output_count) {
    return Status::fail(Reason::ReadBufferMismatch);
  }
  const std::size_t element_bytes =
      buffer->count == 0u ? 0u : buffer->bytes / buffer->count;
  std::size_t expected_bytes = 0u;
  if (logical_count > buffer->count ||
      logical_count > std::numeric_limits<std::uint32_t>::max() ||
      !size::multiply(logical_count, element_bytes, expected_bytes) ||
      expected_bytes != bytes || bytes > buffer->bytes ||
      (data == nullptr && bytes != 0u)) {
    return Status::fail(Reason::ShapeMismatch);
  }
  if (run.program->empty()) {
    if (bytes != 0u && !destination_zeroed) {
      std::memset(data, 0, bytes);
    }
    record_output_hash(run, output_index,
                       ::rund::node::hash_detail::ZeroHash(bytes),
                       logical_count);
    return Status::success();
  }
  if (bytes == 0u) {
    record_output_hash(run, output_index,
                       ::rund::node::hash_detail::ZeroHash(0u), 0u);
    return Status::success();
  }
  std::uint64_t output_hash = 0u;
  if (run.program->device->backend == Backend::Cpu) {
    const CpuBufferState *const cpu = cpu_buffer(*buffer);
    if (cpu == nullptr || cpu->bytes < bytes) {
      return Status::fail(Reason::TransferInvalid);
    }
    output_hash = ::rund::node::hash_detail::CopyHash(cpu->data.get(), data,
                                                      bytes);
  } else {
    if (run.program->device->ops == nullptr ||
        run.program->device->ops->download == nullptr) {
      return Status::fail(Reason::TransferInvalid);
    }
    const DownloadResult transfer = run.program->device->ops->download(
        *run.program->device, *buffer, data, bytes);
    if (staging_bytes != nullptr) {
      *staging_bytes = transfer.staging_bytes;
    }
    if (staging_reused != nullptr) {
      *staging_reused = transfer.staging_reused;
    }
    if (staging_budget != nullptr) {
      *staging_budget = transfer.staging_budget;
    }
    ::rund::detail::counter::Accumulate(run.stats.buffer_allocations,
                                        transfer.buffer_allocations);
    ::rund::detail::counter::Accumulate(run.stats.buffer_reuses,
                                        transfer.buffer_reuses);
    ::rund::detail::counter::Accumulate(run.stats.command_submits,
                                        transfer.command_submits);
    ::rund::detail::counter::Accumulate(run.stats.readback_ns,
                                        transfer.readback_ns);
    if (!transfer.status) {
      return transfer.status;
    }
    if (!transfer.payload_hash_valid) {
      return Status::fail(Reason::TransferInvalid);
    }
    output_hash = transfer.payload_hash;
  }
  ::rund::detail::counter::Accumulate(run.stats.download_events, 1u);
  const std::uint64_t byte_count = static_cast<std::uint64_t>(bytes);
  ::rund::detail::counter::Accumulate(run.stats.downloaded_bytes, byte_count);
  record_transfer(*run.program->device, bytes);
  record_output_hash(run, output_index, output_hash, logical_count);
  return Status::success();
}

} // namespace

Status read_typed_raw(const RunState &run,
                      const std::shared_ptr<BufferState> &buffer,
                      const Type type, void *const data,
                      const std::size_t bytes, const std::size_t count) {
  if (buffer == nullptr || buffer->type != type || buffer->count != count ||
      bytes != count * type_bytes(type)) {
    return Status::fail(Reason::ShapeMismatch);
  }
  return read_bytes(run, buffer, data, bytes, count, nullptr, nullptr, nullptr,
                    false);
}

Status read_job_buffer(const RunState &run,
                       const std::shared_ptr<BufferState> &buffer,
                       void *const data, const std::size_t bytes,
                       const std::size_t logical_count,
                       std::uint64_t &staging_bytes, bool &staging_reused,
                       std::uint64_t &staging_budget,
                       const bool destination_zeroed) {
  staging_bytes = 0u;
  staging_reused = false;
  staging_budget = 0u;
  return read_bytes(run, buffer, data, bytes, logical_count, &staging_bytes,
                    &staging_reused, &staging_budget, destination_zeroed);
}

} // namespace rund::compute::detail
