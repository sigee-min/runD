#include "internal.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) &&                                    \
    !defined(RUND_NODE_TEST_BACKEND_METAL) &&                                  \
    defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace rund_node_test_pipeline_vulkan_persistent {

std::unique_ptr<PreparedCase>
PreparePersistentCase(rund::compute::Device &device,
                      const std::uint64_t coordinate_count,
                      const std::uint64_t identity) {
  using namespace rund::compute;
  constexpr std::size_t elements = 8u;
  auto result = std::make_unique<PreparedCase>();
  auto program = on(device)
                     .map<std::uint32_t>("vulkan-persistent-sliding", 1u,
                                         [](auto value) { return value + 1u; })
                     .compile();
  auto input_backing =
      std::make_shared<PersistentBacking>(elements * sizeof(std::uint32_t));
  result->output_backing =
      std::make_shared<PersistentBacking>(elements * sizeof(std::uint32_t));
  auto input = detail::make_virtual_buffer(
      elements, sizeof(std::uint32_t), detail::Type::U32, {}, input_backing);
  auto output = detail::make_virtual_buffer(elements, sizeof(std::uint32_t),
                                            detail::Type::U32, {},
                                            result->output_backing);
  auto prepared =
      program && input && output
          ? detail::prepare_virtual_pipeline(
                detail::ProgramAccess::state(*program),
                std::move(input).value(), std::move(output).value(),
                ResidencyConfig{})
          : Result<std::shared_ptr<detail::VirtualPipelineState>>::fail(
                Reason::PipelineInvalid);
  if (!prepared || prepared.value() == nullptr ||
      prepared.value()->pipeline == nullptr ||
      prepared.value()->alternate_pipeline == nullptr) {
    std::fprintf(stderr, "Vulkan persistent: virtual prepare\n");
    return {};
  }
  result->primary = NativeBackend(prepared.value()->pipeline);
  result->alternate = NativeBackend(prepared.value()->alternate_pipeline);
  auto *const native =
      static_cast<accel::VulkanPipeline *>(result->primary.get());
  if (result->primary == nullptr || result->alternate == nullptr ||
      result->primary == result->alternate || native == nullptr ||
      native->adapter == nullptr) {
    std::fprintf(stderr, "Vulkan persistent: native owners\n");
    return {};
  }
  result->native = native;

  auto &wait = result->wait;
  wait.request.plan_identity = 0x56'50'53'4cu + identity;
  wait.request.token = 500'000u + identity;
  wait.request.generation = 600'000u + identity;
  wait.request.owner_nonce = 900'000u + identity;
  wait.request.coordinate_count = coordinate_count;
  wait.request.tail_local_count = 1u;
  wait.request.admission = std::make_shared<std::uint8_t>(1u);
  wait.request.final = CompletePersistent;
  wait.request.user = &wait;
  wait.request.memory = accel::ResidencySlidingMemory::HostCoherent;
  wait.request.width = 2u;
  const auto &primary_common = prepared.value()->pipeline->prepared;
  const auto &alternate_common = prepared.value()->alternate_pipeline->prepared;
  std::array<accel::PreparedResidencyPersistentSlidingRole, 2u> roles{};
  for (std::size_t slot = 0u; slot < wait.request.width; ++slot) {
    roles[slot] = accel::PreparedResidencyPersistentSlidingRole{
        .pipeline = slot == 0u ? primary_common : alternate_common,
        .locals = {0u},
        .local_count = 1u,
        .first_control_generation =
            static_cast<std::uint32_t>(700'000u + identity * 1024u + slot),
        .control_generation_stride = 2u,
        .first_descriptor_generation = 800'000u + identity * 1024u + slot,
        .descriptor_generation_stride = 2u,
        .slot = static_cast<std::uint8_t>(slot),
    };
  }
  result->lowering = accel::PrepareKernelPipelinePersistentSliding(
      wait.request,
      std::span<const accel::PreparedResidencyPersistentSlidingRole>{
          roles.data(), roles.size()});
  if (!result->lowering) {
    std::fprintf(stderr, "Vulkan persistent: common lowering\n");
    return {};
  }
  wait.request = result->lowering.active_request();
  wait.request.lowering = result->lowering.backend.lowering;
  const auto &capability = result->lowering.backend.capability;
  if (wait.request.ticket == nullptr || wait.request.ticket->cell == nullptr ||
      wait.request.cell_id == 0u || wait.request.cell_domain == 0u ||
      wait.request.ticket->cell->id != wait.request.cell_id ||
      wait.request.ticket->cell->domain != wait.request.cell_domain ||
      !accel::persistent_sliding_request_valid(capability, wait.request) ||
      result->lowering.backend.encoded_coordinate_count != coordinate_count ||
      result->lowering.backend.encoded_intermediate_bytes !=
          capability.transient_bytes ||
      result->lowering.backend.service.fail_service == nullptr) {
    std::fprintf(stderr, "Vulkan persistent: common request\n");
    return {};
  }
  if (identity == 1u &&
      !RejectOversizedCapability(*native->adapter, wait.request)) {
    std::fprintf(stderr, "Vulkan persistent: oversized capability accepted\n");
    return {};
  }
  if (!accel::persistent_sliding_product_capable(capability) ||
      accel::persistent_sliding_gpu_driven_capable(capability) ||
      capability.retained_bytes == 0u || capability.transient_bytes == 0u) {
    std::fprintf(
        stderr, "Vulkan persistent: capability ok=%u reason=%s bytes=%llu\n",
        static_cast<unsigned>(capability.check.ok), capability.check.reason,
        static_cast<unsigned long long>(capability.transient_bytes));
    return {};
  }
  const auto ticket = wait.request.ticket;
  if (ticket == nullptr || ticket->cell == nullptr || ticket->active ||
      ticket->pending.has_value()) {
    std::fprintf(stderr, "Vulkan persistent: ticket not cold\n");
    return {};
  }
  return result;
}

} // namespace rund_node_test_pipeline_vulkan_persistent

#endif
