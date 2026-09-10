#include "../../../target/selection.hpp"

#include "../../virtual/product/backing.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include "src/accel/kernel/prepared/model.hpp"
#include "src/accel/vulkan/kernel/pipeline/residency/local.hpp"
#include "src/compute/backend/accel/diagnostic.hpp"
#include "src/compute/virtual/state.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>

#include "graph_admission/local.hpp"

#if defined(RUND_NODE_TEST_BACKEND_CPU) ||                                     \
    defined(RUND_NODE_TEST_BACKEND_METAL) ||                                   \
    !defined(RUND_NODE_HAVE_VULKAN_SDK)

int RunComputePipelineVulkanGraphAdmissionContract() { return 0; }

#else

namespace {

constexpr std::size_t FrameElements = 16u;
constexpr std::size_t ElementCount = 73u;
constexpr std::size_t PageBytes = FrameElements * sizeof(std::uint64_t);
constexpr std::size_t StageCount = 4u;
constexpr std::size_t WideStageCount = 2u;
constexpr std::size_t BankCount = 2u;
constexpr std::size_t StageLeafCount = 340u;

const char *candidate_name(
    const rund::node::accel::detail::VulkanResidencyAdmissionCandidate value) {
  using Candidate =
      rund::node::accel::detail::VulkanResidencyAdmissionCandidate;
  switch (value) {
  case Candidate::GraphDirect:
    return "graph-direct";
  case Candidate::GraphStageGeneratedIndirect:
    return "graph-generated";
  case Candidate::GraphStageSequence:
    return "graph-sequence";
  case Candidate::GeneratedIndirect:
    return "generated";
  case Candidate::Direct:
    return "direct";
  }
  return "unknown";
}

template <std::uint64_t First, std::size_t Count, class Expression>
[[nodiscard]] constexpr auto add_stage(Expression value) {
  if constexpr (Count == 1u) {
    return value + First;
  } else {
    constexpr std::size_t Left = Count / 2u;
    return add_stage<First, Left>(value) +
           add_stage<First + Left, Count - Left>(value);
  }
}

} // namespace

[[nodiscard]] bool
check_wide_graph_generated(const rund::compute::Device &device) {
  using namespace rund::compute;
  constexpr std::size_t WideFrameElements = 16u;
  constexpr std::size_t WideElements = 73u;
  constexpr std::size_t WideInputs = 7u;
  constexpr std::size_t WidePageBytes =
      WideFrameElements * sizeof(std::uint64_t);
  constexpr std::size_t WideInputLeafCount = 47u;
  constexpr std::size_t WideStageLeafCount = 127u;
  auto program =
      on(device)
          .input<std::uint64_t>(WideFrameElements)
          .zip_input<std::uint64_t>(WideFrameElements)
          .zip_input<std::uint64_t>(WideFrameElements)
          .zip_input<std::uint64_t>(WideFrameElements)
          .zip_input<std::uint64_t>(WideFrameElements)
          .zip_input<std::uint64_t>(WideFrameElements)
          .zip_input<std::uint64_t>(WideFrameElements)
          .map("vulkan-graph-admission-wide-input",
               [](auto a, auto b, auto c, auto d, auto e, auto f, auto g) {
                 return add_stage<1u, WideInputLeafCount>(a) +
                        add_stage<WideInputLeafCount + 1u, WideInputLeafCount>(
                            b) +
                        add_stage<2u * WideInputLeafCount + 1u,
                                  WideInputLeafCount>(c) +
                        add_stage<3u * WideInputLeafCount + 1u,
                                  WideInputLeafCount>(d) +
                        add_stage<4u * WideInputLeafCount + 1u,
                                  WideInputLeafCount>(e) +
                        add_stage<5u * WideInputLeafCount + 1u,
                                  WideInputLeafCount>(f) +
                        add_stage<6u * WideInputLeafCount + 1u,
                                  WideInputLeafCount>(g);
               })
          .map("vulkan-graph-admission-wide-output",
               [](auto value) {
                 return add_stage<1u, WideStageLeafCount>(value);
               })
          .compile();
  if (!program) {
    std::fprintf(stderr, "graph admission wide compile reason=%u\n",
                 static_cast<unsigned>(program.reason()));
    return false;
  }
  std::array<
      std::shared_ptr<rund_node_test_virtual::product::MemoryVirtualBacking>,
      WideInputs>
      backings{};
  for (auto &backing : backings) {
    backing =
        std::make_shared<rund_node_test_virtual::product::MemoryVirtualBacking>(
            WideElements * sizeof(std::uint64_t), WidePageBytes);
  }
  auto output_backing =
      std::make_shared<rund_node_test_virtual::product::MemoryVirtualBacking>(
          WideElements * sizeof(std::uint64_t), WidePageBytes);
  auto first = virtual_buffer<std::uint64_t>(WideElements, backings[0u]);
  auto second = virtual_buffer<std::uint64_t>(WideElements, backings[1u]);
  auto third = virtual_buffer<std::uint64_t>(WideElements, backings[2u]);
  auto fourth = virtual_buffer<std::uint64_t>(WideElements, backings[3u]);
  auto fifth = virtual_buffer<std::uint64_t>(WideElements, backings[4u]);
  auto sixth = virtual_buffer<std::uint64_t>(WideElements, backings[5u]);
  auto seventh = virtual_buffer<std::uint64_t>(WideElements, backings[6u]);
  auto output = virtual_buffer<std::uint64_t>(WideElements, output_backing);
  if (!first || !second || !third || !fourth || !fifth || !sixth || !seventh ||
      !output) {
    std::fprintf(stderr, "graph admission wide buffer construction failed\n");
    return false;
  }
  auto pipeline =
      virtual_pipeline(*program, *first, *second, *third, *fourth, *fifth,
                       *sixth, *seventh, *output, ResidencyConfig{});
  if (!pipeline) {
    std::fprintf(stderr, "graph admission wide pipeline reason=%u text=%.*s\n",
                 static_cast<unsigned>(pipeline.reason()),
                 static_cast<int>(pipeline.error().size()),
                 pipeline.error().data());
    return false;
  }
  const auto state =
      rund::compute::detail::VirtualPipelineAccess::state(*pipeline);
  if (state == nullptr ||
      state->graph_pipelines.size() != WideStageCount * BankCount) {
    std::fprintf(stderr, "graph admission wide state shape failed\n");
    return false;
  }
  using namespace rund::node::accel::detail;
  for (std::size_t index = 0u; index < state->graph_pipelines.size(); ++index) {
    const auto &graph_pipeline = state->graph_pipelines[index];
    VulkanResidencyAdmissionSnapshot snapshot{};
    if (graph_pipeline == nullptr ||
        !InspectVulkanResidencyAdmission(graph_pipeline->prepared, snapshot) ||
        !snapshot.final_selected ||
        snapshot.final_candidate !=
            VulkanResidencyAdmissionCandidate::GraphStageGeneratedIndirect ||
        snapshot.selected_mode !=
            VulkanResidencyMode::GraphStageGeneratedIndirect) {
      std::fprintf(stderr, "graph admission wide selection failed index=%zu\n",
                   index);
      return false;
    }
    const std::uint32_t expected_data = index / BankCount == 0u ? 8u : 2u;
    bool generated_accepted = false;
    bool direct_rejected = true;
    const auto count = std::min<std::size_t>(snapshot.candidate_count,
                                             snapshot.candidates.size());
    for (std::size_t candidate_index = 0u; candidate_index < count;
         ++candidate_index) {
      const auto &candidate = snapshot.candidates[candidate_index];
      if (candidate.candidate ==
              VulkanResidencyAdmissionCandidate::GraphStageGeneratedIndirect &&
          candidate.reason ==
              VulkanResidencyAdmissionReason::PredicateAccepted &&
          candidate.entry_count == 2u && candidate.active_step_count == 2u &&
          candidate.data_binding_count == expected_data &&
          candidate.control_binding_count == 1u) {
        generated_accepted = true;
      }
      if (candidate.candidate ==
              VulkanResidencyAdmissionCandidate::GraphDirect &&
          candidate.reason ==
              VulkanResidencyAdmissionReason::PredicateAccepted) {
        direct_rejected = false;
      }
    }
    if (!generated_accepted || !direct_rejected) {
      std::fprintf(stderr,
                   "graph admission wide candidate failed index=%zu data=%u\n",
                   index, expected_data);
      return false;
    }
  }
  return true;
}

int RunComputePipelineVulkanGraphAdmissionContract() {
  using namespace rund::compute;
  using rund::compute::detail::accel_diagnostic::Event;
  using rund::compute::detail::accel_diagnostic::ScopedAccelCompileDiagnostic;
  using rund::node::accel::detail::InspectVulkanResidencyAdmission;
  using rund::node::accel::detail::VulkanResidencyAdmissionSnapshot;

  auto opened = open(rund::node::test_contract::target_for(Backend::Vulkan));
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable ? 0 : 1;
  }
  ScopedAccelCompileDiagnostic diagnostics;
  const auto print_diagnostics = [&diagnostics] {
    const auto &snapshot = diagnostics.snapshot();
    std::fprintf(stderr, "graph admission diagnostics count=%u overflow=%u\n",
                 snapshot.count, snapshot.overflow ? 1u : 0u);
    for (std::size_t index = 0u; index < snapshot.count; ++index) {
      const Event &event = snapshot.events[index];
      std::fprintf(
          stderr,
          " graph diagnostic seq=%llu phase=%u context=%llu graph=%llu/%llu "
          "nodes=%llu tiles=%llu binds=%llu mode=%u failed=%u check=%u "
          "raw=%u inner=%.*s inner_code=%llu default=%.*s "
          "default_code=%llu projected=%.*s projected_code=%llu\n",
          static_cast<unsigned long long>(event.sequence),
          static_cast<unsigned>(event.phase),
          static_cast<unsigned long long>(event.context_id),
          static_cast<unsigned long long>(event.graph_id_hi),
          static_cast<unsigned long long>(event.graph_id_lo),
          static_cast<unsigned long long>(event.node_count),
          static_cast<unsigned long long>(event.tile_count),
          static_cast<unsigned long long>(event.binding_count), event.mode,
          event.failed_node, event.check_ok ? 1u : 0u,
          event.raw_reason_present ? 1u : 0u,
          static_cast<int>(event.inner_reason_length),
          event.inner_reason == nullptr ? "" : event.inner_reason,
          static_cast<unsigned long long>(event.inner_reason_code),
          static_cast<int>(event.effective_default_length),
          event.effective_default == nullptr ? "" : event.effective_default,
          static_cast<unsigned long long>(event.effective_default_code),
          static_cast<int>(event.projected_reason_length),
          event.projected_reason == nullptr ? "" : event.projected_reason,
          static_cast<unsigned long long>(event.projected_reason_code));
    }
  };
  auto program =
      on(*opened)
          .input<std::uint64_t>(FrameElements)
          .zip_input<std::uint64_t>(FrameElements)
          .zip_input<std::uint64_t>(FrameElements)
          .branch([](auto first, auto second, auto third) {
            const auto prefix =
                first.map("vulkan-graph-admission-prefix", [](auto value) {
                  return add_stage<1u, StageLeafCount>(value);
                });
            const auto left =
                second.map("vulkan-graph-admission-left", [](auto value) {
                  return add_stage<StageLeafCount + 1u, StageLeafCount>(value);
                });
            const auto right =
                third.map("vulkan-graph-admission-right", [](auto value) {
                  return add_stage<2u * StageLeafCount + 1u, StageLeafCount>(
                      value);
                });
            return zip(prefix, left, right)
                .map("vulkan-graph-admission-terminal",
                     [](auto first_value, auto left_value, auto right_value) {
                       return first_value + left_value + right_value;
                     });
          })
          .compile();
  if (!program) {
    std::fprintf(stderr, "graph admission compile reason=%.*s\n",
                 static_cast<int>(program.error().size()),
                 program.error().data());
    print_diagnostics();
    return 2;
  }

  auto first_backing =
      std::make_shared<rund_node_test_virtual::product::MemoryVirtualBacking>(
          ElementCount * sizeof(std::uint64_t), PageBytes);
  auto second_backing =
      std::make_shared<rund_node_test_virtual::product::MemoryVirtualBacking>(
          ElementCount * sizeof(std::uint64_t), PageBytes);
  auto third_backing =
      std::make_shared<rund_node_test_virtual::product::MemoryVirtualBacking>(
          ElementCount * sizeof(std::uint64_t), PageBytes);
  auto output_backing =
      std::make_shared<rund_node_test_virtual::product::MemoryVirtualBacking>(
          ElementCount * sizeof(std::uint64_t), PageBytes);
  auto first = virtual_buffer<std::uint64_t>(ElementCount, first_backing);
  auto second = virtual_buffer<std::uint64_t>(ElementCount, second_backing);
  auto third = virtual_buffer<std::uint64_t>(ElementCount, third_backing);
  auto output = virtual_buffer<std::uint64_t>(ElementCount, output_backing);
  const auto report_buffer = [](const char *name, const auto &buffer) {
    if (buffer) {
      return true;
    }
    std::fprintf(stderr, "graph admission buffer=%s reason=%u text=%.*s\n",
                 name, static_cast<unsigned>(buffer.reason()),
                 static_cast<int>(buffer.error().size()),
                 buffer.error().data());
    return false;
  };
  if (!report_buffer("first", first) || !report_buffer("second", second) ||
      !report_buffer("third", third) || !report_buffer("output", output)) {
    return 3;
  }
  auto pipeline =
      virtual_pipeline(*program, first.value(), second.value(), third.value(),
                       output.value(), ResidencyConfig{});
  if (!pipeline) {
    std::fprintf(stderr, "graph admission pipeline reason=%u text=%.*s\n",
                 static_cast<unsigned>(pipeline.reason()),
                 static_cast<int>(pipeline.error().size()),
                 pipeline.error().data());
    const auto location = pipeline.location();
    std::fprintf(
        stderr,
        "graph admission location native_reason_key=%s native_text=%.*s "
        "template_index=%u occurrence_index=%u node=%u step=%u\n",
        location.native_reason_key == nullptr ? "" : location.native_reason_key,
        static_cast<int>(pipeline.error().size()), pipeline.error().data(),
        location.template_index, location.occurrence_index, location.node,
        location.step);
    print_diagnostics();
    return 4;
  }
  const auto state =
      rund::compute::detail::VirtualPipelineAccess::state(*pipeline);
  if (state == nullptr ||
      state->geometry.route !=
          rund::compute::detail::VirtualRoute::GraphPointwise ||
      state->geometry.device_vsm_required || state->input_count != 3u ||
      state->graph_input_resource_count != 3u ||
      state->graph_pipelines.size() != StageCount * BankCount) {
    std::fprintf(
        stderr,
        "graph admission shape state=%u route=%u required=%u "
        "inputs=%llu graph_inputs=%llu pipelines=%llu\n",
        state != nullptr ? 1u : 0u,
        state == nullptr ? 255u : static_cast<unsigned>(state->geometry.route),
        state != nullptr && state->geometry.device_vsm_required ? 1u : 0u,
        state == nullptr ? 0ull
                         : static_cast<unsigned long long>(state->input_count),
        state == nullptr ? 0ull
                         : static_cast<unsigned long long>(
                               state->graph_input_resource_count),
        state == nullptr
            ? 0ull
            : static_cast<unsigned long long>(state->graph_pipelines.size()));
    return 5;
  }

  if (state->graph_pipelines.empty() ||
      !node_compute_pipeline_vulkan_graph_admission::CheckGenerationSeedDomains(
          state->graph_pipelines[0u])) {
    std::fprintf(stderr, "graph admission generation seed domains failed\n");
    return 8;
  }

  for (std::size_t index = 0u; index < state->graph_pipelines.size(); ++index) {
    const auto &graph_pipeline = state->graph_pipelines[index];
    VulkanResidencyAdmissionSnapshot snapshot{};
    if (graph_pipeline == nullptr ||
        !InspectVulkanResidencyAdmission(graph_pipeline->prepared, snapshot)) {
      std::fprintf(stderr, "graph admission unavailable index=%llu\n",
                   static_cast<unsigned long long>(index));
      return 5;
    }
    if (snapshot.candidate_count != snapshot.candidates.size() ||
        snapshot.generation_tag == 0u) {
      return 6;
    }
    std::fprintf(stderr,
                 "graph admission index=%llu final=%u selected=%u tag=%llu\n",
                 static_cast<unsigned long long>(index),
                 static_cast<unsigned>(snapshot.final_candidate),
                 snapshot.final_selected ? 1u : 0u,
                 static_cast<unsigned long long>(snapshot.generation_tag));
    for (const auto &candidate : snapshot.candidates) {
      std::fprintf(
          stderr,
          " graph candidate=%s reason=%u entry=%u local=%u predicate=%u "
          "E=%u W=%u data=%u control=%u fail_pred=%u fail_ordinal=%u "
          "fail_reason=%s tag=%llu\n",
          candidate_name(candidate.candidate),
          static_cast<unsigned>(candidate.reason), candidate.entry_ordinal,
          candidate.local_ordinal, candidate.predicate_ordinal,
          candidate.entry_count, candidate.active_step_count,
          candidate.data_binding_count, candidate.control_binding_count,
          static_cast<unsigned>(candidate.first_failure_predicate),
          candidate.first_failure_ordinal,
          candidate.first_failure_reason == nullptr
              ? ""
              : candidate.first_failure_reason,
          static_cast<unsigned long long>(candidate.generation_tag));
    }
    const std::uint32_t expected_data =
        index / BankCount == StageCount - 1u ? 4u : 2u;
    if (snapshot.candidates[1].reason !=
            rund::node::accel::detail::VulkanResidencyAdmissionReason::
                PredicateAccepted ||
        snapshot.candidates[1].data_binding_count != expected_data ||
        snapshot.candidates[1].control_binding_count != 1u ||
        !snapshot.final_selected ||
        snapshot.final_candidate !=
            rund::node::accel::detail::VulkanResidencyAdmissionCandidate::
                GraphStageGeneratedIndirect ||
        snapshot.selected_mode !=
            rund::node::accel::detail::VulkanResidencyMode::
                GraphStageGeneratedIndirect) {
      const auto &candidate = snapshot.candidates[1];
      std::fprintf(
          stderr,
          "graph admission generated expected accepted/final-selected "
          "candidate_reason=%u final_selected=%u final_candidate=%u "
          "selected_mode=%u fail_pred=%u fail_ordinal=%u fail_reason=%s\n",
          static_cast<unsigned>(candidate.reason),
          snapshot.final_selected ? 1u : 0u,
          static_cast<unsigned>(snapshot.final_candidate),
          static_cast<unsigned>(snapshot.selected_mode),
          static_cast<unsigned>(candidate.first_failure_predicate),
          candidate.first_failure_ordinal,
          candidate.first_failure_reason == nullptr
              ? ""
              : candidate.first_failure_reason);
      return 7;
    }
  }
  if (state->graph_pipelines.size() < 2u ||
      !node_compute_pipeline_vulkan_graph_admission::
          PreparedPipelineReleaseUnlocks(state->graph_pipelines[0u]) ||
      !node_compute_pipeline_vulkan_graph_admission::RejectRecordedGraphReuse(
          state->graph_pipelines[0u]) ||
      !node_compute_pipeline_vulkan_graph_admission::
          RejectRecordedGraphReprepare(state->graph_pipelines[1u])) {
    std::fprintf(stderr,
                 "graph admission recorded graph reuse/reprepare accepted\n");
    return 8;
  }
  return check_wide_graph_generated(*opened) ? 0 : 9;
}

#endif
