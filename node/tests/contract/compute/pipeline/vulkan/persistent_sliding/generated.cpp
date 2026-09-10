#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) &&                                    \
    !defined(RUND_NODE_TEST_BACKEND_METAL) &&                                  \
    defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace rund_node_test_pipeline_vulkan_persistent {

void CompleteGenericProbe(void *, accel::KernelResult) noexcept {}

[[nodiscard]] bool
RejectGeneratedMapGeneric(const std::shared_ptr<void> &prepared,
                          accel::VulkanPipeline &native) noexcept {
  if (native.residency == nullptr ||
      native.residency->mode !=
          accel::VulkanResidencyMode::GeneratedIndirectMap) {
    return false;
  }
  const std::array<std::uint32_t, 1u> locals{0u};
  std::array<VkCommandBuffer, 3u> commands{};
  std::size_t command_count = 0u;
  std::uint64_t dispatch_count = 0u;
  std::uint64_t control_count = 0u;
  std::uint64_t reset_count = 0u;
  std::uint64_t reset_bytes = 0u;
  std::uint64_t before = 0u;
  {
    std::lock_guard lock{native.adapter->mutex};
    before = native.adapter->command_submit_count;
  }
  const rund::AccelCheck built = accel::BuildVulkanResidencySubmission(
      native, std::span<const std::uint32_t>{locals.data(), locals.size()},
      std::span<VkCommandBuffer>{commands.data(), commands.size()},
      command_count, dispatch_count, control_count, reset_count, reset_bytes);
  accel::vulkan_generated_indirect_detail::Plan plan{};
  const rund::AccelCheck selected =
      accel::vulkan_generated_indirect_detail::select(
          native, std::span<const std::uint32_t>{locals.data(), locals.size()},
          std::span<VkCommandBuffer>{commands.data(), commands.size()},
          command_count, dispatch_count, control_count, plan);
  const accel::BackendResidencySlidingDescriptor descriptor{
      .owner = &native,
      .plan_identity = 1u,
      .token = 1u,
      .generation = 1u,
      .coordinate = 0u,
      .turn = 0u,
      .read_mask = 0u,
      .write_mask = 0u,
      .descriptor_generation = 1u,
      .control_generation = 1u,
      .stride = 1u,
      .slot = 0u,
  };
  const rund::AccelCheck submitted = accel::SubmitVulkanResidencySliding(
      prepared, descriptor, CompleteGenericProbe, &native,
      accel::KernelTiming::None, accel::PipelineSubmitMode::Residency,
      std::span<const std::uint32_t>{locals.data(), locals.size()});
  std::uint64_t after = 0u;
  {
    std::lock_guard lock{native.adapter->mutex};
    after = native.adapter->command_submit_count;
  }
  return !built.ok && !selected.ok && !submitted.ok && command_count == 0u &&
         dispatch_count == 0u && control_count == 0u && reset_count == 0u &&
         reset_bytes == 0u && !plan.generated && before == after;
}

[[nodiscard]] bool CheckGeneratedReadAtMap() {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 8u> values{10u, 20u, 30u, 40u,
                                                 50u, 60u, 70u, 80u};
  constexpr std::array<std::uint32_t, 4u> indices{7u, 3u, 5u, 1u};
  auto opened = open(Target::vulkan());
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable;
  }
  Device device = std::move(opened).value();
  auto program = on(device)
                     .input<std::uint32_t>(values.size())
                     .zip_input<std::uint32_t>(indices.size())
                     .branch([](auto source, auto requested) {
                       return source.gather(requested).map(
                           "vulkan-persistent-read-at",
                           [](auto value) { return value + 1u; });
                     })
                     .compile();
  auto source = rund_node_test_pipeline::Upload(device, values);
  auto requested = rund_node_test_pipeline::Upload(device, indices);
  auto output = device.buffer<std::uint32_t>(indices.size());
  if (!program || !source || !requested || !output) {
    std::fprintf(stderr, "Vulkan ReadAt Map construction failed\n");
    return false;
  }
  auto prepared = pipeline(device)
                      .then(*program, read(*source, *requested), write(*output))
                      .prepare();
  if (!prepared) {
    std::fprintf(stderr, "Vulkan ReadAt Map prepare reason=%s\n",
                 prepared.error().data());
    return false;
  }
  const std::shared_ptr<detail::PipelineState> state =
      detail::PipelineStateAccess::state(*prepared);
  const std::shared_ptr<void> backend = NativeBackend(state);
  auto *const native = static_cast<accel::VulkanPipeline *>(backend.get());
  if (native == nullptr || native->residency == nullptr ||
      native->residency->mode !=
          accel::VulkanResidencyMode::GeneratedIndirectMap) {
    std::fprintf(stderr, "Vulkan ReadAt Map mode=%u\n",
                 native == nullptr || native->residency == nullptr
                     ? 255u
                     : static_cast<unsigned>(native->residency->mode));
    return false;
  }
  if (!RejectGeneratedMapGeneric(backend, *native)) {
    std::fprintf(stderr, "Vulkan ReadAt Map generic entry accepted\n");
    return false;
  }
  return true;
}

} // namespace rund_node_test_pipeline_vulkan_persistent

#endif
