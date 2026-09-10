#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) && \
    !defined(RUND_NODE_TEST_BACKEND_METAL) && \
    defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace rund_node_test_pipeline::vulkan_transfer {

int ExactVulkanResidencySelection() {
  using namespace rund::compute;
  using namespace rund::node::accel::detail;
  constexpr std::array<std::uint32_t, 8u> input{1u, 2u, 3u, 4u,
                                                 5u, 6u, 7u, 8u};
  constexpr std::array<std::uint32_t, 8u> zero{};
  constexpr std::array<std::uint32_t, 8u> sentinel{99u, 99u, 99u, 99u,
                                                   99u, 99u, 99u, 99u};
  auto opened = open(Target::vulkan());
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable ? 0 : 1;
  }
  Device device = std::move(opened).value();
  auto plus =
      on(device)
          .map<std::uint32_t>("vulkan-residency-selected-local", input.size(),
                              [](auto value) { return value + 3u; })
          .compile();
  auto source = Upload(device, input);
  auto middle = Upload(device, zero);
  auto output = Upload(device, zero);
  if (!plus || !source || !middle || !output) {
    return 2;
  }
  auto resident_pipeline = pipeline(device)
                               .then(*plus, read(*source), write(*middle))
                               .then(*plus, read(*middle), write(*output))
                               .prepare();
  if (!resident_pipeline) {
    return 3;
  }
  const std::shared_ptr<detail::PipelineState> state =
      detail::PipelineStateAccess::state(*resident_pipeline);
  bool supported = false;
  bool ready = false;
  if (state == nullptr ||
      !QueryPreparedKernelPipelineResidency(state->prepared, supported) ||
      !PreparedKernelPipelineResidencyReady(state->prepared, ready) ||
      !supported || !ready) {
    return 4;
  }
  auto *const common =
      static_cast<prepared::PipelineState *>(state->prepared.owner.get());
  auto *const native =
      common == nullptr ? nullptr
                        : static_cast<VulkanPipeline *>(common->backend.get());
  const PreparedPipelineMemory memory =
      ReadPreparedKernelPipelineMemory(state->prepared);
  if (native == nullptr || native->residency == nullptr ||
      !native->residency->ready || native->residency->steps.size() != 2u ||
      native->residency->host_bytes == 0u ||
      memory.host.current < native->residency->host_bytes ||
      native->residency->prefix.buffer == VK_NULL_HANDLE ||
      native->residency->suffix.buffer == VK_NULL_HANDLE ||
      native->residency->steps[0].declared_step != 0u ||
      native->residency->steps[1].declared_step != 1u ||
      native->residency->steps[0].command.buffer == VK_NULL_HANDLE ||
      native->residency->steps[1].command.buffer == VK_NULL_HANDLE ||
      native->residency->steps[0].command.buffer ==
          native->residency->steps[1].command.buffer) {
    return 5;
  }

  constexpr std::array<std::uint32_t, 2u> reversed{1u, 0u};
  std::array<std::uint32_t, input.size()> observed_middle{};
  std::array<std::uint32_t, input.size()> observed_output{};
  PreparedPipelineEvidence reversed_evidence{};
  if (!SeedVulkanSelection(state)) {
    return 8;
  }
  if (!RunVulkanSelection(state, reversed, reversed_evidence)) {
    return 8;
  }
  if (reversed_evidence.shared.run.work.dispatch_count != 2u) {
    return 16;
  }
  if (!DownloadVulkanSelection(state, *middle, observed_middle) ||
      !DownloadVulkanSelection(state, *output, observed_output)) {
    return 10;
  }
  for (std::size_t index = 0u; index < input.size(); ++index) {
    if (observed_middle[index] != input[index] + 3u ||
        observed_output[index] != 3u) {
      return 11;
    }
  }
  if (!CheckVulkanRangeFailure(state, detail::BufferAccess::state(*middle),
                               observed_middle)) {
    return 17;
  }

  constexpr std::array<std::uint32_t, 1u> out_of_range{2u};
  constexpr std::array<std::uint32_t, 2u> duplicate{0u, 0u};
  if (!RejectVulkanSelection(state, out_of_range) ||
      !RejectVulkanSelection(state, duplicate)) {
    return 6;
  }

  constexpr std::array<std::uint32_t, 1u> tail{0u};
  auto tail_middle = Upload(device, zero);
  auto tail_output = Upload(device, sentinel);
  auto tail_pipeline =
      tail_middle && tail_output
          ? pipeline(device)
                .then(*plus, read(*source), write(*tail_middle))
                .then(*plus, read(*tail_middle), write(*tail_output))
                .prepare()
          : Result<Pipeline>::fail(Reason::PipelineInvalid);
  const std::shared_ptr<detail::PipelineState> tail_state =
      tail_pipeline ? detail::PipelineStateAccess::state(*tail_pipeline)
                    : nullptr;
  PreparedPipelineEvidence tail_evidence{};
  if (tail_state == nullptr) {
    return 12;
  }
  supported = false;
  ready = false;
  if (!QueryPreparedKernelPipelineResidency(tail_state->prepared, supported) ||
      !PreparedKernelPipelineResidencyReady(tail_state->prepared, ready) ||
      !supported || !ready || !SeedVulkanSelection(tail_state)) {
    return 12;
  }
  if (!RunVulkanSelection(tail_state, tail, tail_evidence) ||
      tail_evidence.shared.run.work.dispatch_count != 1u ||
      !DownloadVulkanSelection(tail_state, *tail_middle, observed_middle) ||
      !DownloadVulkanSelection(tail_state, *tail_output, observed_output)) {
    return 12;
  }
  for (std::size_t index = 0u; index < input.size(); ++index) {
    if (observed_middle[index] != input[index] + 3u ||
        observed_output[index] != sentinel[index]) {
      return 13;
    }
  }

  constexpr std::array<std::uint32_t, 2u> ordered{0u, 1u};
  auto ordered_middle = Upload(device, zero);
  auto ordered_output = Upload(device, zero);
  auto ordered_pipeline =
      ordered_middle && ordered_output
          ? pipeline(device)
                .then(*plus, read(*source), write(*ordered_middle))
                .then(*plus, read(*ordered_middle), write(*ordered_output))
                .prepare()
          : Result<Pipeline>::fail(Reason::PipelineInvalid);
  const std::shared_ptr<detail::PipelineState> ordered_state =
      ordered_pipeline ? detail::PipelineStateAccess::state(*ordered_pipeline)
                       : nullptr;
  PreparedPipelineEvidence ordered_evidence{};
  if (ordered_state == nullptr) {
    return 14;
  }
  supported = false;
  ready = false;
  if (!QueryPreparedKernelPipelineResidency(ordered_state->prepared,
                                            supported) ||
      !PreparedKernelPipelineResidencyReady(ordered_state->prepared, ready) ||
      !supported || !ready || !SeedVulkanSelection(ordered_state)) {
    return 14;
  }
  if (!RunVulkanSelection(ordered_state, ordered, ordered_evidence) ||
      ordered_evidence.shared.run.work.dispatch_count != 2u ||
      !DownloadVulkanSelection(ordered_state, *ordered_output,
                               observed_output)) {
    return 14;
  }
  for (std::size_t index = 0u; index < input.size(); ++index) {
    if (observed_output[index] != input[index] + 6u) {
      return 15;
    }
  }
  return 0;
}

} // namespace rund_node_test_pipeline::vulkan_transfer

#endif
