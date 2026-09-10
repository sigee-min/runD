#include "internal.hpp"

#include <cstdio>

namespace rund_node_test_pipeline_residency::device_vsm_test::binary_detail {

[[nodiscard]] bool RejectBinary(const rund::compute::Backend backend,
                                const std::uint64_t pages,
                                const char *const phase) noexcept {
  std::fprintf(stderr, "DeviceVsm binary backend=%u q=%llu phase=%s\n",
               static_cast<unsigned>(backend),
               static_cast<unsigned long long>(pages), phase);
  return false;
}

[[nodiscard]] std::unique_ptr<PreparedBinary>
PrepareBinary(const rund::compute::Backend backend,
              const std::shared_ptr<rund::compute::detail::DeviceState> &device,
              const std::uint64_t pages, const ActualPrepare prepare,
              const ActualQueueCount queue_count) {
  constexpr std::uint64_t PayloadElements = 16u;
  auto result = std::make_unique<PreparedBinary>();
  result->backend = backend;
  result->device = device;
  result->pages = pages;
  result->elements = static_cast<std::size_t>(pages * PayloadElements - 3u);
  result->native = device == nullptr
                       ? nullptr
                       : rund::compute::detail::accel_device(*device);
  if (result->native == nullptr || prepare == nullptr ||
      queue_count == nullptr) {
    static_cast<void>(RejectBinary(backend, pages, "owner"));
    return {};
  }
  result->first.resize(result->elements);
  result->second.resize(result->elements);
  result->zero.assign(result->elements, 0u);
  for (std::size_t index = 0u; index < result->elements; ++index) {
    result->first[index] = static_cast<std::uint32_t>(index * 3u + pages);
    result->second[index] = static_cast<std::uint32_t>(index * 5u + 7u);
  }
  result->buffers = std::array<rund::AccelBuffer, 3u>{
      rund::node::accel::CreateAccelBuffer(
          result->native->context,
          rund::AccelBufferDesc{.scalar_width_bytes = sizeof(std::uint32_t),
                                .count = result->elements,
                                .usage = rund::BufferUsage::ReadOnly}),
      rund::node::accel::CreateAccelBuffer(
          result->native->context,
          rund::AccelBufferDesc{.scalar_width_bytes = sizeof(std::uint32_t),
                                .count = result->elements,
                                .usage = rund::BufferUsage::ReadOnly}),
      rund::node::accel::CreateAccelBuffer(
          result->native->context,
          rund::AccelBufferDesc{.scalar_width_bytes = sizeof(std::uint32_t),
                                .count = result->elements,
                                .usage = rund::BufferUsage::WriteOnly}),
  };
  std::array<accel::UploadEntry, 3u> uploads{
      accel::UploadEntry{.buffer = &result->buffers[0u],
                         .data = result->first.data(),
                         .bytes = result->first.size() * sizeof(std::uint32_t)},
      accel::UploadEntry{.buffer = &result->buffers[1u],
                         .data = result->second.data(),
                         .bytes =
                             result->second.size() * sizeof(std::uint32_t)},
      accel::UploadEntry{.buffer = &result->buffers[2u],
                         .data = result->zero.data(),
                         .bytes = result->zero.size() * sizeof(std::uint32_t)},
  };
  if (!result->buffers[0u] || !result->buffers[1u] || !result->buffers[2u] ||
      !accel::UploadAccelBuffers(result->native->context, uploads,
                                 result->routes,
                                 accel::TransferCompletion::Complete)
           .check.ok) {
    static_cast<void>(RejectBinary(backend, pages, "staging"));
    return {};
  }
  for (std::size_t index = 0u; index < result->routes.size(); ++index) {
    result->routes[index] = accel::ProjectAccelBufferRoute(
        result->native->context, result->buffers[index]);
    if (result->routes[index].handle == nullptr) {
      static_cast<void>(RejectBinary(backend, pages, "projection"));
      return {};
    }
  }
  result->proof = BuildProof(ApiFor(backend), pages, result->routes);
  if (result->proof == nullptr ||
      !accel::device_vsm_proof_valid(*result->proof)) {
    if (result->proof != nullptr) {
      const auto &proof = *result->proof;
      std::fprintf(
          stderr,
          "DeviceVsm proof backend=%u artifact=%u/%u plan=%u/%u "
          "dispatch=%llu/%llu params=%llu/%llu input=%llu/%llu/%u/%llu/%llu "
          "output=%llu/%llu/%u/%llu/%llu geometry=%llu/%llu/%llu/%llu/%u "
          "width=%u\n",
          static_cast<unsigned>(backend),
          static_cast<unsigned>(proof.artifact != nullptr),
          static_cast<unsigned>(proof.artifact != nullptr &&
                                proof.artifact->ok),
          static_cast<unsigned>(proof.plan.ok),
          static_cast<unsigned>(proof.artifact != nullptr &&
                                proof.artifact->key.api == proof.plan.api),
          static_cast<unsigned long long>(proof.plan.dispatch_count),
          static_cast<unsigned long long>(proof.window_count),
          static_cast<unsigned long long>(proof.parameter_bytes),
          static_cast<unsigned long long>(proof.plan.param_bytes),
          static_cast<unsigned long long>(proof.residents.rows[0u].backing.id),
          static_cast<unsigned long long>(
              proof.residents.rows[0u].backing.bytes),
          proof.residents.rows[0u].backing.usage,
          static_cast<unsigned long long>(
              proof.residents.rows[0u].backing.element_bytes),
          static_cast<unsigned long long>(
              proof.residents.rows[0u].backing.count),
          static_cast<unsigned long long>(proof.residents.rows[1u].backing.id),
          static_cast<unsigned long long>(
              proof.residents.rows[1u].backing.bytes),
          proof.residents.rows[1u].backing.usage,
          static_cast<unsigned long long>(
              proof.residents.rows[1u].backing.element_bytes),
          static_cast<unsigned long long>(
              proof.residents.rows[1u].backing.count),
          static_cast<unsigned long long>(proof.geometry.logical_bytes),
          static_cast<unsigned long long>(proof.geometry.payload_bytes),
          static_cast<unsigned long long>(proof.geometry.frame_bytes),
          static_cast<unsigned long long>(proof.geometry.page_count),
          proof.geometry.element_bytes, proof.width);
    }
    static_cast<void>(RejectBinary(backend, pages, "proof"));
    return {};
  }
  result->preparation =
      accel::PrepareDeviceVsm(result->native->pick, result->proof);
  if (!result->preparation ||
      result->preparation.capability.width != result->proof->width) {
    static_cast<void>(RejectBinary(
        backend, pages, result->preparation.capability.check.reason));
    return {};
  }
  const std::shared_ptr<residency::DeviceVsmRegistration> registration =
      residency::register_device_vsm_proof(device->residency, result->proof);
  residency::DirectRecurrenceLease lease =
      registration == nullptr
          ? residency::DirectRecurrenceLease{}
          : device->residency->authority().direct_recurrences().begin_direct_recurrence(
                registration->request());
  result->wait = std::make_shared<wait_detail::Owner>(std::move(lease));
  result->wait->device = device;
  result->wait->registry = device->residency;
  result->wait->registration = registration;
  result->wait->self = result->wait;
  result->wait->request = accel::DeviceVsmRequest{
      .proof = result->proof,
      .lowering = result->preparation.lowering,
      .admission = registration,
      .token = result->wait->lease.token(),
      .generation = result->wait->lease.generation(),
      .nonce = result->wait->lease.owner(),
      .submission_control = &result->wait->submission_control,
      .final = wait_detail::complete,
      .user = result->wait.get(),
  };
  return result;
}

} // namespace rund_node_test_pipeline_residency::device_vsm_test::binary_detail
