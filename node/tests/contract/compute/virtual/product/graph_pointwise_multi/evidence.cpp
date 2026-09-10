#include "internal.hpp"

#include "src/compute/virtual/backing.hpp"
#include "src/compute/virtual/run/device_vsm/model.hpp"

#include <array>
#include <cstdio>
#include <limits>

namespace rund_node_test_virtual::product::graph_pointwise_multi {
namespace {

using Owner =
    rund::compute::detail::device_vsm_product_detail::DeviceVsmProductOwner;

[[nodiscard]] std::shared_ptr<Owner> owner(const Case &test_case) noexcept {
  return test_case.state == nullptr
             ? std::shared_ptr<Owner>{}
             : std::static_pointer_cast<Owner>(
                   test_case.state->device_vsm_product_cache);
}

[[nodiscard]] bool proof_valid(const Owner &value) noexcept {
  using namespace rund::node::accel::detail;
  const auto &proof = value.proof;
  return proof != nullptr &&
         proof->topology == DeviceVsmTopology::GraphPointwise &&
         proof->plan.input_buffer_count == InputCount &&
         proof->plan.output_buffer_count == 1u &&
         proof->residents.input_count == InputCount &&
         proof->residents.output_count == 1u &&
         proof->residents.count == InputCount + 1u &&
         proof->graph_pointwise.stage_count == StageCount &&
         proof->graph_pointwise.topology.external_input_count == InputCount &&
         proof->graph_pointwise.topology.stages[0u].input_count == 1u &&
         proof->graph_pointwise.topology.stages[0u].inputs[0u].kind ==
             DeviceVsmGraphValueSourceKind::ExternalInput &&
         proof->graph_pointwise.topology.stages[0u].inputs[0u].index == 0u &&
         proof->graph_pointwise.topology.stages[1u].input_count == 2u &&
         proof->graph_pointwise.topology.stages[1u].inputs[0u].kind ==
             DeviceVsmGraphValueSourceKind::StageOutput &&
         proof->graph_pointwise.topology.stages[1u].inputs[0u].index == 0u &&
         proof->graph_pointwise.topology.stages[1u].inputs[1u].kind ==
             DeviceVsmGraphValueSourceKind::ExternalInput &&
         proof->graph_pointwise.topology.stages[1u].inputs[1u].index == 1u &&
         proof->graph_pointwise.topology.stages[2u].input_count == 2u &&
         proof->graph_pointwise.topology.stages[2u].inputs[0u].kind ==
             DeviceVsmGraphValueSourceKind::StageOutput &&
         proof->graph_pointwise.topology.stages[2u].inputs[0u].index == 1u &&
         proof->graph_pointwise.topology.stages[2u].inputs[1u].kind ==
             DeviceVsmGraphValueSourceKind::ExternalInput &&
         proof->graph_pointwise.topology.stages[2u].inputs[1u].index == 2u;
}

[[nodiscard]] bool native_valid(const Owner &value,
                                const std::size_t page_count) noexcept {
  const auto &evidence = value.evidence;
  return evidence != nullptr && evidence->public_handoff_count == 1u &&
         evidence->authority_accept_count == 1u &&
         evidence->pipeline_terminal_count == StageCount &&
         evidence->backing_publication_count == 1u &&
         evidence->final_received && !evidence->quarantined &&
         evidence->native.page_count == page_count &&
         evidence->native.generated_epochs == page_count &&
         evidence->native.completed_epochs == page_count &&
         evidence->native.native_submit_count == 1u &&
         evidence->native.epoch_native_submit_count == 0u &&
         evidence->native.payload_dispatch_count == 1u &&
         evidence->native.host_service_turn_count == 0u &&
         evidence->native.host_epoch_callback_count == 0u &&
         evidence->native.final_callback_count == 1u;
}

[[nodiscard]] bool backing_valid(const Case &test_case,
                                 const std::uint64_t initial_version) noexcept {
  using rund::compute::detail::VirtualBackingAccess;
  const BackingFacts first = test_case.first_backing->facts();
  const BackingFacts second = test_case.second_backing->facts();
  const BackingFacts third = test_case.third_backing->facts();
  const BackingFacts output = test_case.output_backing->facts();
  const std::size_t logical_bytes =
      test_case.expected.size() * sizeof(std::uint64_t);
  return first.read_count == test_case.page_count &&
         first.read_bytes == logical_bytes &&
         second.read_count == test_case.page_count &&
         second.read_bytes == logical_bytes &&
         third.read_count == test_case.page_count &&
         third.read_bytes == logical_bytes &&
         output.write_count == test_case.page_count &&
         output.write_bytes == logical_bytes &&
         VirtualBackingAccess::version(*test_case.output_backing) ==
             initial_version + 1u &&
         VirtualBackingAccess::recovery_bytes(*test_case.output_backing) == 0u;
}

} // namespace

namespace {

enum Leaf : std::size_t {
  Backend = 0u,
  Status = 1u,
  RetainedOwner = 2u,
  Proof = 3u,
  Native = 4u,
  Backing = 5u,
  Frame = 6u,
  Handoff = 7u,
  Batch = 8u,
  Queue = 9u,
  PageIn = 10u,
  ReadBytes = 11u,
  PageOut = 12u,
  WriteBytes = 13u,
  Submit = 14u,
  Dispatch = 15u,
  FinalDispatch = 16u,
  Output = 17u,
  Tail = 18u,
  Published = 19u,
  WarmBackend = 20u,
  WarmStatus = 21u,
  WarmOwner = 22u,
  WarmProof = 23u,
  WarmNative = 24u,
  WarmVersion = 25u,
  WarmWriteCount = 26u,
  WarmWriteBytes = 27u,
  WarmStages = 28u,
  WarmHandoff = 29u,
  WarmBatch = 30u,
  WarmQueue = 31u,
};

void note(SuccessReport &report, const std::size_t bit,
          const char *const name) noexcept {
  if (bit >= FailedLeafCapacity) {
    return;
  }
  report.failed_mask |= std::uint64_t{1u} << bit;
  if (report.failed_count < report.failed_names.size()) {
    report.failed_names[report.failed_count++] = name;
  }
}

void capture(Case &test_case, SuccessReport &report) noexcept {
  if (!report.output_reached) {
    report.output = observe_output(test_case);
    report.output_reached = true;
  }
}

[[nodiscard]] bool fail(Case &test_case, SuccessReport &report,
                        const std::size_t bit,
                        const char *const name) noexcept {
  (void)test_case;
  note(report, bit, name);
  return false;
}

} // namespace

void report_failure(const SuccessReport &report,
                    const rund::compute::Backend backend,
                    const rund::compute::Status &status,
                    const std::size_t page_count,
                    const char *const phase) noexcept {
  std::fprintf(stderr,
               "Graph multi %s backend=%u pages=%zu reason=%.*s failed=0x%llx "
               "leaves=",
               phase, static_cast<unsigned>(backend), page_count,
               static_cast<int>(status.error().size()), status.error().data(),
               static_cast<unsigned long long>(report.failed_mask));
  for (std::size_t index = 0u; index < report.failed_count; ++index) {
    std::fprintf(stderr, "%s%s", index == 0u ? "" : ",",
                 report.failed_names[index]);
  }
  std::fprintf(
      stderr,
      " published=%u output_readable=%u output_matches=%u "
      "first_bad_index=%zu actual=%llu expected=%llu owner=%u submit=%llu "
      "dispatch=%llu page=%llu/%llu write=%llu/%llu stages=",
      static_cast<unsigned>(report.published),
      static_cast<unsigned>(report.output.readable),
      static_cast<unsigned>(report.output.matches),
      report.output.first_bad_index,
      static_cast<unsigned long long>(report.output.actual),
      static_cast<unsigned long long>(report.output.expected),
      static_cast<unsigned>(report.owner),
      static_cast<unsigned long long>(report.command_submits),
      static_cast<unsigned long long>(report.dispatches),
      static_cast<unsigned long long>(report.page_in),
      static_cast<unsigned long long>(report.page_out),
      static_cast<unsigned long long>(report.write_count),
      static_cast<unsigned long long>(report.write_bytes));
  for (std::size_t stage = 0u; stage < StageCount; ++stage) {
    std::fprintf(
        stderr, "%s%llu->%llu", stage == 0u ? "" : ",",
        static_cast<unsigned long long>(report.before_generations[stage]),
        static_cast<unsigned long long>(report.after_generations[stage]));
  }
  std::fputc('\n', stderr);
}

bool validate_success(Case &test_case, const rund::compute::Backend backend,
                      const rund::compute::Status &status,
                      const std::uint64_t initial_version, const bool published,
                      SuccessReport &report) noexcept {
  using namespace rund::compute;
  const Stats stats = test_case.pipeline.stats();
  const ResidencyStats residency = stats.pipeline.residency;
  const std::shared_ptr<Owner> retained = owner(test_case);
  const std::size_t logical_bytes =
      test_case.expected.size() * sizeof(std::uint64_t);
  report.command_submits = stats.command_submits;
  report.dispatches = stats.dispatches;
  report.page_in = residency.page_in_count;
  report.page_out = residency.page_out_count;
  report.owner = retained != nullptr;
  report.published = published;
  if (backend == Backend::Cpu) {
    return fail(test_case, report, Leaf::Backend, "backend");
  }
  if (!status) {
    return fail(test_case, report, Leaf::Status, "status");
  }
  if (retained == nullptr) {
    return fail(test_case, report, Leaf::RetainedOwner, "owner");
  }
  if (!proof_valid(*retained)) {
    return fail(test_case, report, Leaf::Proof, "proof");
  }
  if (!native_valid(*retained, test_case.page_count)) {
    return fail(test_case, report, Leaf::Native, "native");
  }
  if (!backing_valid(test_case, initial_version)) {
    return fail(test_case, report, Leaf::Backing, "backing");
  }
  if (test_case.pipeline.plan().residency.frame_capacity != 2u) {
    return fail(test_case, report, Leaf::Frame, "frame");
  }
  if (residency.window_handoff_count != 1u) {
    return fail(test_case, report, Leaf::Handoff, "handoff");
  }
  if (residency.window_batch_count != 1u) {
    return fail(test_case, report, Leaf::Batch, "batch");
  }
  if (residency.window_queue_call_count != 1u) {
    return fail(test_case, report, Leaf::Queue, "queue");
  }
  if (residency.page_in_count != InputCount * test_case.page_count) {
    return fail(test_case, report, Leaf::PageIn, "page_in");
  }
  if (residency.backing_read_bytes != InputCount * logical_bytes) {
    return fail(test_case, report, Leaf::ReadBytes, "read_bytes");
  }
  if (residency.page_out_count != test_case.page_count) {
    return fail(test_case, report, Leaf::PageOut, "page_out");
  }
  if (residency.backing_write_bytes != logical_bytes) {
    return fail(test_case, report, Leaf::WriteBytes, "write_bytes");
  }
  if (stats.command_submits != 1u) {
    return fail(test_case, report, Leaf::Submit, "submit");
  }
  if (stats.dispatches != 1u) {
    return fail(test_case, report, Leaf::Dispatch, "dispatch");
  }
  if (stats.final_dispatches != 1u) {
    return fail(test_case, report, Leaf::FinalDispatch, "final_dispatch");
  }
  capture(test_case, report);
  if (!report.output.matches) {
    return fail(test_case, report, Leaf::Output, "output");
  }
  if (!test_case.output_backing->tail_poisoned()) {
    return fail(test_case, report, Leaf::Tail, "tail");
  }
  if (!published) {
    return fail(test_case, report, Leaf::Published, "published");
  }
  return true;
}

bool validate_warm(Case &test_case, const rund::compute::Backend backend,
                   const rund::compute::Status &status,
                   const std::uint64_t initial_version,
                   const BackingFacts &before,
                   const std::array<std::uint64_t, StageCount> &generations,
                   const std::array<std::uint64_t, StageCount> &published,
                   SuccessReport &report) noexcept {
  using namespace rund::compute;
  using rund::compute::detail::VirtualBackingAccess;
  const Stats stats = test_case.pipeline.stats();
  const ResidencyStats residency = stats.pipeline.residency;
  const std::shared_ptr<Owner> retained = owner(test_case);
  const BackingFacts output = test_case.output_backing->facts();
  const std::size_t logical_bytes =
      test_case.expected.size() * sizeof(std::uint64_t);
  report.command_submits = stats.command_submits;
  report.dispatches = stats.dispatches;
  report.page_in = residency.page_in_count;
  report.page_out = residency.page_out_count;
  report.write_count = output.write_count;
  report.write_bytes = output.write_bytes;
  report.owner = retained != nullptr;
  bool stage_ok = true;
  for (std::size_t stage = 0u; stage < StageCount; ++stage) {
    stage_ok =
        stage_ok &&
        generations[stage] != std::numeric_limits<std::uint64_t>::max() &&
        published[stage] == generations[stage] + 1u;
  }
  if (backend == Backend::Cpu) {
    return fail(test_case, report, Leaf::WarmBackend, "warm_backend");
  }
  if (!status) {
    return fail(test_case, report, Leaf::WarmStatus, "warm_status");
  }
  if (retained == nullptr) {
    return fail(test_case, report, Leaf::WarmOwner, "warm_owner");
  }
  if (!proof_valid(*retained)) {
    return fail(test_case, report, Leaf::WarmProof, "warm_proof");
  }
  if (!native_valid(*retained, test_case.page_count)) {
    return fail(test_case, report, Leaf::WarmNative, "warm_native");
  }
  if (VirtualBackingAccess::version(*test_case.output_backing) !=
      initial_version + 1u) {
    return fail(test_case, report, Leaf::WarmVersion, "warm_version");
  }
  if (output.write_count != before.write_count + test_case.page_count) {
    return fail(test_case, report, Leaf::WarmWriteCount, "warm_write_count");
  }
  if (output.write_bytes != before.write_bytes + logical_bytes) {
    return fail(test_case, report, Leaf::WarmWriteBytes, "warm_write_bytes");
  }
  if (!stage_ok) {
    return fail(test_case, report, Leaf::WarmStages, "warm_stages");
  }
  if (residency.window_handoff_count != 1u) {
    return fail(test_case, report, Leaf::WarmHandoff, "warm_handoff");
  }
  if (residency.window_batch_count != 1u) {
    return fail(test_case, report, Leaf::WarmBatch, "warm_batch");
  }
  if (residency.window_queue_call_count != 1u) {
    return fail(test_case, report, Leaf::WarmQueue, "warm_queue");
  }
  capture(test_case, report);
  if (!report.output.matches) {
    return fail(test_case, report, Leaf::Output, "warm_output");
  }
  if (!test_case.output_backing->tail_poisoned()) {
    return fail(test_case, report, Leaf::Tail, "warm_tail");
  }
  return true;
}

} // namespace rund_node_test_virtual::product::graph_pointwise_multi
