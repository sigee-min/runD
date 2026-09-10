#include "model.hpp"

#include "src/hash/fnv.hpp"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <utility>

namespace rund_node_test_pipeline_transfer {
namespace {

using namespace rund::compute;
using namespace rund::compute::detail;
using rund::node::accel::detail::DownloadRangeState;
using rund::node::accel::detail::MarkDownloadComplete;
using rund::node::accel::detail::MarkDownloadFailure;

[[nodiscard]] ResidentBatchSpy *
resident_spy(const BufferState *const buffer) noexcept {
  const AccelBufferState *const storage =
      buffer == nullptr ? nullptr : accel_buffer(*buffer);
  return storage == nullptr
             ? nullptr
             : static_cast<ResidentBatchSpy *>(storage->buffer.owner.get());
}

[[nodiscard]] bool copy_resident(const DownloadRequest &request,
                                 ResidentBatchSpy &spy) noexcept {
  if (request.bytes != 0u && request.data == nullptr) {
    return false;
  }
  if (request.offset > spy.source.size() ||
      request.bytes > spy.source.size() - request.offset) {
    return false;
  }
  if (request.bytes != 0u) {
    std::memcpy(request.data, spy.source.data() + request.offset,
                request.bytes);
  }
  const std::uint64_t hash = rund::node::hash_detail::HashBytes(
      spy.source.data() + request.offset, request.bytes);
  if (request.payload_hash != nullptr) {
    *request.payload_hash = hash;
  }
  MarkDownloadComplete(request.outcome, request.bytes, hash,
                       request.payload_hash != nullptr);
  return true;
}

DownloadResult ResidentDownload(DeviceState &, const BufferState &buffer,
                                void *const data, const std::size_t bytes,
                                const std::size_t offset) {
  ResidentBatchSpy *const spy = resident_spy(&buffer);
  if (spy == nullptr) {
    return DownloadResult{.status = Status::fail(Reason::TransferInvalid)};
  }
  ++spy->scalar_calls;
  ++spy->scalar_routes;
  DownloadRequest request{
      .buffer = &buffer,
      .data = data,
      .bytes = bytes,
      .offset = offset,
  };
  if (!copy_resident(request, *spy)) {
    return DownloadResult{.status = Status::fail(Reason::TransferInvalid)};
  }
  return DownloadResult{.status = Status::success(),
                        .confirmed_bytes = bytes,
                        .ordered_prefix = 1u,
                        .payload_hash_valid = true};
}

DownloadResult ResidentDownloadBatch(
    DeviceState &, const std::span<const DownloadRequest> requests,
    const rund::node::accel::detail::TransferAuthority authority) {
  if (requests.empty()) {
    return DownloadResult{.status = Status::fail(Reason::TransferInvalid)};
  }
  ResidentBatchSpy *const spy = resident_spy(requests.front().buffer);
  if (spy == nullptr) {
    return DownloadResult{.status = Status::fail(Reason::TransferInvalid)};
  }
  ++spy->batch_calls;
  spy->batch_routes = requests.size();
  spy->authority = authority;
  spy->outcomes.fill({});
  DownloadResult result{.status = Status::success(),
                        .command_submits = 1u,
                        .readback_ns = 13u,
                        .payload_hash_valid = true};
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    const DownloadRequest &request = requests[index];
    if (spy->late_failure && index == 1u) {
      if (request.bytes != 0u && request.data == nullptr) {
        MarkDownloadFailure(request.outcome, DownloadRangeState::FailedNoWrite);
        spy->outcomes[index] = *request.outcome;
        result.status = Status::fail(Reason::TransferInvalid);
        result.payload_hash_valid = false;
        result.first_failed = index;
        result.first_failed_valid = true;
        return result;
      }
      const std::size_t partial = std::min<std::size_t>(2u, request.bytes);
      if (request.offset > spy->source.size() ||
          request.bytes > spy->source.size() - request.offset) {
        MarkDownloadFailure(request.outcome, DownloadRangeState::FailedNoWrite);
        spy->outcomes[index] = *request.outcome;
        result.status = Status::fail(Reason::TransferInvalid);
        result.payload_hash_valid = false;
        result.first_failed = index;
        result.first_failed_valid = true;
        return result;
      }
      if (partial != 0u) {
        std::memcpy(request.data, spy->source.data() + request.offset, partial);
      }
      MarkDownloadFailure(request.outcome, DownloadRangeState::FailedMayWrite,
                          partial);
      spy->outcomes[index] = *request.outcome;
      result.status = Status::fail(Reason::TransferInvalid);
      result.payload_hash_valid = false;
      result.confirmed_bytes = requests[0u].bytes;
      result.ordered_prefix = 1u;
      result.first_failed = index;
      result.first_failed_valid = true;
      return result;
    }
    if (!copy_resident(request, *spy)) {
      MarkDownloadFailure(request.outcome, DownloadRangeState::FailedNoWrite);
      spy->outcomes[index] = *request.outcome;
      result.status = Status::fail(Reason::TransferInvalid);
      result.payload_hash_valid = false;
      result.first_failed = index;
      result.first_failed_valid = true;
      return result;
    }
    spy->outcomes[index] = *request.outcome;
    result.confirmed_bytes += request.bytes;
    ++result.ordered_prefix;
  }
  return result;
}

const DeviceOps resident_operations{
    .download = ResidentDownload,
    .download_batch = ResidentDownloadBatch,
    .download_prefix_capacity = 64u,
};

const DeviceOps resident_scalar_operations{.download = ResidentDownload};

UploadResult
UploadBatch(DeviceState &, const std::span<const UploadRequest> requests,
            const rund::node::accel::detail::TransferCompletion completion,
            const rund::node::accel::detail::TransferAuthority authority) {
  if (active_probe == nullptr) {
    return UploadResult{.status = Status::fail(Reason::TransferInvalid)};
  }
  ++active_probe->upload_calls;
  active_probe->upload_routes = requests.size();
  if (completion != rund::node::accel::detail::TransferCompletion::Complete ||
      authority != rund::node::accel::detail::TransferAuthority::Shared ||
      active_probe->fail_upload) {
    return UploadResult{.status = Status::fail(Reason::TransferInvalid),
                        .command_submits = 1u};
  }
  for (const UploadRequest &request : requests) {
    NativeBuffer *const target =
        request.buffer == nullptr ? nullptr : native(*request.buffer);
    if (target == nullptr || request.bytes > target->bytes.size()) {
      return UploadResult{.status = Status::fail(Reason::TransferInvalid),
                          .command_submits = 1u};
    }
    std::memcpy(target->bytes.data(), request.data, request.bytes);
  }
  return UploadResult{.status = Status::success(),
                      .staging_bytes = 28u,
                      .staging_peak_bytes = 16u,
                      .staging_reused_bytes = 28u,
                      .staging_budget = 64u,
                      .buffer_reuses = 2u,
                      .command_submits = 1u};
}

DownloadResult
DownloadBatch(DeviceState &, const std::span<const DownloadRequest> requests,
              const rund::node::accel::detail::TransferAuthority authority) {
  if (active_probe == nullptr ||
      authority != rund::node::accel::detail::TransferAuthority::Shared) {
    return DownloadResult{.status = Status::fail(Reason::TransferInvalid)};
  }
  ++active_probe->download_calls;
  active_probe->download_routes = requests.size();
  if (active_probe->fail_download_once) {
    active_probe->fail_download_once = false;
    return DownloadResult{.status = Status::fail(Reason::TransferInvalid),
                          .command_submits = 1u};
  }
  for (const DownloadRequest &request : requests) {
    const NativeBuffer *const source =
        request.buffer == nullptr ? nullptr : native(*request.buffer);
    if (source == nullptr || request.payload_hash == nullptr ||
        request.bytes > source->bytes.size()) {
      return DownloadResult{.status = Status::fail(Reason::TransferInvalid),
                            .command_submits = 1u};
    }
    std::memcpy(request.data, source->bytes.data(), request.bytes);
    *request.payload_hash =
        rund::node::hash_detail::HashBytes(source->bytes.data(), request.bytes);
  }
  return DownloadResult{.status = Status::success(),
                        .staging_bytes = 28u,
                        .staging_peak_bytes = 12u,
                        .staging_reused_bytes = 28u,
                        .staging_budget = 64u,
                        .buffer_reuses = 2u,
                        .command_submits = 1u,
                        .readback_ns = 17u,
                        .payload_hash_valid = true};
}

const DeviceOps operations{.upload_batch = UploadBatch,
                           .download_batch = DownloadBatch};

} // namespace

TransferProbe *active_probe{};

ResidentBatchFixture make_resident_batch_spy() {
  ResidentBatchFixture fixture{
      .spy = std::make_shared<ResidentBatchSpy>(),
      .device = std::make_shared<DeviceState>(),
      .buffer = std::make_shared<BufferState>(),
      .backing = std::make_shared<ResidentVirtualBacking>(),
  };
  fixture.device->backend = Backend::Vulkan;
  fixture.device->ops = &resident_operations;
  fixture.buffer->device = fixture.device;
  fixture.buffer->type = Type::U32;
  fixture.buffer->count = fixture.spy->source.size() / sizeof(std::uint32_t);
  fixture.buffer->bytes = fixture.spy->source.size();
  fixture.buffer->physical_bytes = fixture.buffer->bytes;
  rund::AccelBuffer accel{};
  accel.check = {true, "ok"};
  accel.byte_extent = fixture.buffer->bytes;
  accel.owner = fixture.spy;
  fixture.buffer->storage.emplace<AccelBufferState>(
      AccelBufferState{.buffer = std::move(accel)});
  for (std::size_t index = 0u; index < fixture.spy->source.size(); ++index) {
    fixture.spy->source[index] = static_cast<std::byte>(index + 1u);
  }
  VirtualBackingAccess::bind_resident(*fixture.backing, fixture.buffer);
  return fixture;
}

ResidentBatchFixture make_resident_scalar_spy() {
  ResidentBatchFixture fixture = make_resident_batch_spy();
  fixture.device->ops = &resident_scalar_operations;
  return fixture;
}

NativeBuffer *native(BufferState &buffer) noexcept {
  AccelBufferState *const storage = accel_buffer(buffer);
  return storage == nullptr
             ? nullptr
             : static_cast<NativeBuffer *>(storage->buffer.owner.get());
}

const NativeBuffer *native(const BufferState &buffer) noexcept {
  const AccelBufferState *const storage = accel_buffer(buffer);
  return storage == nullptr
             ? nullptr
             : static_cast<const NativeBuffer *>(storage->buffer.owner.get());
}

Fixture::Fixture()
    : device{std::make_shared<DeviceState>()},
      state{std::make_shared<PipelineState>()} {
  device->backend = Backend::Vulkan;
  device->ops = &operations;
  state->device = device;
  state->publication = std::make_shared<PipelinePublicationState>();
  state->publication->device = device;
  state->publication->generation = 3u;
  state->preparing = false;
  state->phase = PipelinePhase::Ready;
  state->steps.emplace_back();
  inputs[0u] = make_buffer(first_bytes, 0u);
  inputs[1u] = make_buffer(tail_bytes, 0u);
  outputs[0u] = make_buffer(first_bytes, 3u);
  outputs[1u] = make_buffer(tail_bytes, 3u);
  for (std::size_t index = 0u; index < inputs.size(); ++index) {
    state->resources.push_back(PipelineResource{
        .buffer = inputs[index],
        .type = Type::U32,
        .count = inputs[index]->count,
        .bytes = inputs[index]->bytes,
    });
  }
  for (std::size_t index = 0u; index < outputs.size(); ++index) {
    const std::uint32_t resource =
        static_cast<std::uint32_t>(state->resources.size());
    state->resources.push_back(PipelineResource{
        .buffer = outputs[index],
        .type = Type::U32,
        .count = outputs[index]->count,
        .bytes = outputs[index]->bytes,
        .output = static_cast<std::uint32_t>(index),
    });
    state->outputs.push_back(
        PipelineOutputState{.generation = 3u, .resource = resource});
  }
  state->unobserved_outputs = state->outputs.size();
  fill_output(0u, std::byte{0x31});
  fill_output(1u, std::byte{0xa7});
}

std::shared_ptr<BufferState>
Fixture::make_buffer(const std::size_t bytes,
                     const std::uint64_t generation) const {
  auto buffer = std::make_shared<BufferState>();
  auto owner = std::make_shared<NativeBuffer>();
  rund::AccelBuffer accel{};
  accel.owner = owner;
  buffer->device = device;
  buffer->storage.emplace<AccelBufferState>(
      AccelBufferState{.buffer = std::move(accel)});
  buffer->type = Type::U32;
  buffer->count = bytes / sizeof(std::uint32_t);
  buffer->bytes = bytes;
  buffer->generation = generation;
  return buffer;
}

void Fixture::fill_output(const std::size_t index, const std::byte seed) const {
  NativeBuffer *const storage = native(*outputs[index]);
  for (std::size_t byte = 0u; byte < outputs[index]->bytes; ++byte) {
    storage->bytes[byte] =
        static_cast<std::byte>(static_cast<unsigned>(seed) + byte);
  }
}

PipelineFrameUploadResult
transfer_locked(Fixture &fixture,
                const std::span<const PipelineFrameUpload> frames) {
  std::lock_guard pipeline_lock{fixture.state->gate};
  std::lock_guard publication_lock{fixture.state->publication->gate};
  return upload_pipeline_frames(*fixture.state, frames);
}

PipelineFrameDownloadResult
transfer_locked(Fixture &fixture,
                const std::span<const PipelineFrameDownload> frames) {
  std::lock_guard pipeline_lock{fixture.state->gate};
  std::lock_guard publication_lock{fixture.state->publication->gate};
  return download_pipeline_frames(*fixture.state, frames);
}

std::array<PipelineFrameUpload, 2u>
uploads(Fixture &fixture, const std::array<std::uint32_t, 4u> &first,
        const std::array<std::uint32_t, 3u> &tail) noexcept {
  return {{{fixture.inputs[0u].get(), first.data(), Fixture::first_bytes},
           {fixture.inputs[1u].get(), tail.data(), Fixture::tail_bytes}}};
}

} // namespace rund_node_test_pipeline_transfer
