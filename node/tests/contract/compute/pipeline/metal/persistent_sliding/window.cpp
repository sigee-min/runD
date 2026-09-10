#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) &&                                    \
    !defined(RUND_NODE_TEST_BACKEND_VULKAN)

namespace rund_node_test_pipeline_metal_persistent {

[[nodiscard]] bool RunPublicSpatialWindowN48() {
  using namespace rund::compute;
  constexpr std::size_t elements = 48u;
  constexpr std::size_t frame = 16u;
  constexpr std::size_t radius = 2u;
  constexpr std::uint64_t native_batches = 2u;
  constexpr std::uint64_t queue_calls = 2u;
  auto opened = open(Target::metal());
  if (!opened) {
#if defined(__APPLE__)
    return false;
#else
    return opened.reason() == Reason::AdapterUnavailable;
#endif
  }
  Device device = std::move(opened).value();
  auto program =
      on(device)
          .map<std::int32_t>("public-spatial-window-pre", frame,
                             [](auto value) { return value + 3; })
          .window(WindowSpec{
              .op = Window::Sum, .radius = radius, .edge = WindowEdge::Clamp})
          .map("public-spatial-window-post",
               [](auto value) { return value * 2; })
          .compile();
  auto input_backing =
      std::make_shared<PersistentBacking>(elements * sizeof(std::int32_t));
  auto output_backing =
      std::make_shared<PersistentBacking>(elements * sizeof(std::int32_t));
  auto input = virtual_buffer<std::int32_t>(elements, input_backing);
  auto output = virtual_buffer<std::int32_t>(elements, output_backing);
  auto prepared =
      program && input && output
          ? virtual_pipeline(*program, *input, *output, ResidencyConfig{})
          : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                Reason::PipelineInvalid);
  if (!prepared) {
    return false;
  }
  const auto &virtual_state =
      rund::compute::detail::VirtualPipelineAccess::state(*prepared);
  const auto publication_snapshot = [&]() noexcept {
    constexpr std::uint64_t invalid = std::numeric_limits<std::uint64_t>::max();
    if (virtual_state == nullptr || virtual_state->pipeline == nullptr ||
        virtual_state->alternate_pipeline == nullptr ||
        virtual_state->pipeline->publication == nullptr ||
        virtual_state->alternate_pipeline->publication == nullptr) {
      return std::array<std::uint64_t, 4u>{invalid, invalid, invalid, invalid};
    }
    std::scoped_lock lock{virtual_state->pipeline->publication->gate,
                          virtual_state->alternate_pipeline->publication->gate};
    return std::array<std::uint64_t, 4u>{
        virtual_state->pipeline->publication->generation,
        virtual_state->pipeline->publication->payload_epoch,
        virtual_state->alternate_pipeline->publication->generation,
        virtual_state->alternate_pipeline->publication->payload_epoch};
  };
  const auto publication_before = publication_snapshot();
  const std::uint64_t output_version_before =
      rund::compute::detail::VirtualBackingAccess::version(*output_backing);
  if (publication_before[0u] == std::numeric_limits<std::uint64_t>::max()) {
    return false;
  }
  if (!prepared->run()) {
    return false;
  }
  const auto publication_after = publication_snapshot();
  auto *const common = virtual_state == nullptr ||
                               virtual_state->pipeline == nullptr ||
                               !virtual_state->pipeline->prepared.ok
                           ? nullptr
                           : static_cast<accel::prepared::PipelineState *>(
                                 virtual_state->pipeline->prepared.owner.get());
  if (common == nullptr || common->backend == nullptr) {
    return false;
  }
  // an authenticated local row through the existing Persistent request API;
  // the same proof/local guard must reject it before any native owner exists.
  accel::PersistentResidencySlidingRequest forged{};
  forged.plan_identity = 0x5350'574eu;
  forged.token = 0x5350'574eu;
  forged.generation = 1u;
  forged.coordinate_count = 1u;
  forged.tail_local_count = 1u;
  forged.admission = std::make_shared<std::uint8_t>(1u);
  forged.final = IgnorePersistentFinal;
  forged.user = &forged;
  forged.memory = accel::ResidencySlidingMemory::HostCoherent;
  forged.width = 1u;
  forged.roles[0u] = accel::PersistentResidencySlidingRole{
      .prepared = common->backend,
      .locals = {accel::ResidencyWindowLocalCapacity},
      .local_count = 1u,
      .first_control_generation = 1u,
      .control_generation_stride = 1u,
      .first_descriptor_generation = 1u,
      .descriptor_generation_stride = 1u,
      .slot = 0u,
  };
  const accel::MetalPersistentResidencySlidingPreparation forged_preparation =
      accel::PrepareMetalPersistentResidencySliding(forged);
  if (forged_preparation.capability.check.ok ||
      forged_preparation.lowering != nullptr) {
    return false;
  }
  std::array<std::byte, elements * sizeof(std::int32_t)> observed{};
  if (!output_backing->read(0u, observed)) {
    return false;
  }
  for (std::size_t index = 0u; index < elements; ++index) {
    std::int32_t value = 0;
    std::memcpy(&value, observed.data() + index * sizeof(value), sizeof(value));
    if (value != 30) {
      return false;
    }
  }
  const Stats stats = prepared->stats();
  const ResidencyStats &residency = stats.pipeline.residency;
  // Native/GPU/Final acceptance remains backend-owned by validation.hpp; the
  // public surface additionally proves both private publication generations,
  // payload epochs, and the one backing-version commit.
  const bool shape_io =
      residency.page_count == 4u && residency.frame_capacity == 2u &&
      residency.page_in_count == 4u &&
      residency.backing_read_bytes == elements * sizeof(std::int32_t) &&
      residency.page_in_bytes == 4u * frame * sizeof(std::int32_t) &&
      output_backing->write_bytes() == elements * sizeof(std::int32_t);
  const bool submission = residency.window_handoff_count == 1u &&
                          residency.window_batch_count == native_batches &&
                          residency.window_queue_call_count == queue_calls &&
                          stats.command_submits == queue_calls;
  const bool publication =
      publication_after[0u] == publication_before[0u] + 1u &&
      publication_after[1u] == publication_before[1u] + 1u &&
      publication_after[2u] == publication_before[2u] + 1u &&
      publication_after[3u] == publication_before[3u] + 1u &&
      rund::compute::detail::VirtualBackingAccess::version(*output_backing) ==
          output_version_before + 1u;
  const bool valid = shape_io && submission && publication;
  if (!valid) {
    std::fprintf(
        stderr,
        "Metal public spatial Window N48 shape_io=%u submission=%u "
        "publication=%u page=%llu/%u frame=%llu/%u pagein=%llu/%u "
        "pagebytes=%llu/%llu read=%llu/%llu write=%llu/%llu "
        "batch=%llu/%llu queue=%llu/%llu command=%llu/%llu\n",
        static_cast<unsigned>(shape_io), static_cast<unsigned>(submission),
        static_cast<unsigned>(publication),
        static_cast<unsigned long long>(residency.page_count), 4u,
        static_cast<unsigned long long>(residency.frame_capacity), 2u,
        static_cast<unsigned long long>(residency.page_in_count), 4u,
        static_cast<unsigned long long>(residency.page_in_bytes),
        static_cast<unsigned long long>(4u * frame * sizeof(std::int32_t)),
        static_cast<unsigned long long>(residency.backing_read_bytes),
        static_cast<unsigned long long>(elements * sizeof(std::int32_t)),
        static_cast<unsigned long long>(output_backing->write_bytes()),
        static_cast<unsigned long long>(elements * sizeof(std::int32_t)),
        static_cast<unsigned long long>(residency.window_batch_count),
        static_cast<unsigned long long>(native_batches),
        static_cast<unsigned long long>(residency.window_queue_call_count),
        static_cast<unsigned long long>(queue_calls),
        static_cast<unsigned long long>(stats.command_submits),
        static_cast<unsigned long long>(queue_calls));
  }
  return valid;
}

[[nodiscard]] bool RunNaturalFallbackWindowN53() {
  using namespace rund::compute;
  constexpr std::size_t elements = 53u;
  constexpr std::size_t frame = 16u;
  constexpr std::size_t radius = 2u;
  constexpr std::size_t pages = 5u;
  constexpr std::uint64_t logical_bytes = elements * sizeof(std::int32_t);
  constexpr std::uint64_t physical_bytes = pages * frame * sizeof(std::int32_t);
  auto opened = open(Target::metal());
  if (!opened) {
#if defined(__APPLE__)
    return false;
#else
    return opened.reason() == Reason::AdapterUnavailable;
#endif
  }
  Device device = std::move(opened).value();
  auto program =
      on(device)
          .map<std::int32_t>("public-fallback-window-pre", frame,
                             [](auto value) { return value + 3; })
          .window(WindowSpec{
              .op = Window::Sum, .radius = radius, .edge = WindowEdge::Clamp})
          .map("public-fallback-window-post",
               [](auto value) { return value * 2; })
          .compile();
  auto input_backing = std::make_shared<PersistentBacking>(logical_bytes, 2u);
  auto output_backing = std::make_shared<PersistentBacking>(logical_bytes);
  auto input = virtual_buffer<std::int32_t>(elements, input_backing);
  auto output = virtual_buffer<std::int32_t>(elements, output_backing);
  auto prepared =
      program && input && output
          ? virtual_pipeline(*program, *input, *output, ResidencyConfig{})
          : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                Reason::PipelineInvalid);
  if (!prepared) {
    return false;
  }
  const std::uint64_t output_version_before =
      rund::compute::detail::VirtualBackingAccess::version(*output_backing);
  if (!prepared->run()) {
    return false;
  }
  std::array<std::byte, logical_bytes> observed{};
  if (!output_backing->read(0u, observed)) {
    return false;
  }
  for (std::size_t index = 0u; index < elements; ++index) {
    std::int32_t actual = 0;
    std::memcpy(&actual, observed.data() + index * sizeof(actual),
                sizeof(actual));
    if (actual != 30) {
      return false;
    }
  }
  const Stats stats = prepared->stats();
  const ResidencyStats &residency = stats.pipeline.residency;
  const std::uint64_t epochs = prepared->plan().residency.epoch_count;
  const bool exact =
      prepared->plan().residency.page_count == pages && epochs != 0u &&
      residency.page_count == pages && residency.page_in_count == pages &&
      residency.page_in_bytes == physical_bytes &&
      residency.backing_read_bytes == logical_bytes &&
      output_backing->write_bytes() == logical_bytes &&
      residency.window_handoff_count == 0u &&
      residency.window_batch_count == 0u &&
      residency.window_queue_call_count == 0u &&
      // max_parallel_reads=2 deliberately selects the legacy overlap fallback,
      // so its three successful rolling epoch writebacks remain non-atomic.
      // The serialized N48 case above is the transaction publication oracle.
      rund::compute::detail::VirtualBackingAccess::version(*output_backing) ==
          output_version_before + epochs;
  if (!exact) {
    std::fprintf(
        stderr,
        "Metal natural fallback N53 pages=%llu/%zu pagein=%llu/%zu "
        "pagebytes=%llu/%llu read=%llu/%llu window=%llu/%llu/%llu "
        "epochs=%llu version=%llu/%llu\n",
        static_cast<unsigned long long>(residency.page_count), pages,
        static_cast<unsigned long long>(residency.page_in_count), pages,
        static_cast<unsigned long long>(residency.page_in_bytes),
        static_cast<unsigned long long>(physical_bytes),
        static_cast<unsigned long long>(residency.backing_read_bytes),
        static_cast<unsigned long long>(logical_bytes),
        static_cast<unsigned long long>(residency.window_handoff_count),
        static_cast<unsigned long long>(residency.window_batch_count),
        static_cast<unsigned long long>(residency.window_queue_call_count),
        static_cast<unsigned long long>(epochs),
        static_cast<unsigned long long>(
            rund::compute::detail::VirtualBackingAccess::version(
                *output_backing)),
        static_cast<unsigned long long>(output_version_before + epochs));
  }
  return exact;
}

} // namespace rund_node_test_pipeline_metal_persistent

#endif
