#include "internal.hpp"

#include "src/compute/virtual/backing.hpp"

#include <cstdio>

namespace rund_node_test_virtual::product::graph_pointwise {

int check_recovery(const rund::compute::Device &device,
                   const rund::compute::Backend backend) {
  using namespace rund::compute;
  Preparation prepared = prepare_case(device, 5u);
  if (!prepared.value) {
    return prepared.reason;
  }
  Case &test_case = *prepared.value;
  const std::uint64_t first_version =
      detail::VirtualBackingAccess::version(*test_case.output_backing);
  const Status first = test_case.pipeline.run();
  if (!validate_success(test_case, backend, first, first_version)) {
    return 5;
  }

  const BackingFacts before_retry = test_case.output_backing->facts();
  const auto before_failure_generations = stage_generations(test_case);
  const std::uint64_t retry_version =
      detail::VirtualBackingAccess::version(*test_case.output_backing);
  test_case.output_backing->fail_next_write_after(sizeof(std::uint64_t),
                                                  Reason::BackendFailed);
  const Status failed = test_case.pipeline.run();
  const auto failed_generations = stage_generations(test_case);
  const BackingFacts after_failure = test_case.output_backing->facts();
  const std::uint64_t failed_recovery =
      detail::VirtualBackingAccess::recovery_bytes(*test_case.output_backing);
  const Status retried = test_case.pipeline.run();
  const auto retried_generations = stage_generations(test_case);
  const BackingFacts after_retry = test_case.output_backing->facts();
  const std::size_t logical_bytes =
      test_case.expected.size() * sizeof(std::uint64_t);
  bool generation_valid = backend == Backend::Cpu;
  if (backend != Backend::Cpu) {
    generation_valid = true;
    for (std::size_t stage = 0u; stage < StageCount; ++stage) {
      generation_valid =
          generation_valid &&
          failed_generations[stage] == before_failure_generations[stage] &&
          retried_generations[stage] == failed_generations[stage] + 1u;
    }
  }
  const bool valid =
      failed.reason() == Reason::BackendFailed && retried && generation_valid &&
      after_failure.write_failure_count ==
          before_retry.write_failure_count + 1u &&
      after_failure.partial_write_bytes ==
          before_retry.partial_write_bytes + sizeof(std::uint64_t) &&
      detail::VirtualBackingAccess::version(*test_case.output_backing) ==
          retry_version + 1u &&
      failed_recovery == logical_bytes &&
      detail::VirtualBackingAccess::recovery_bytes(*test_case.output_backing) ==
          0u &&
      after_retry.write_count == after_failure.write_count + 5u &&
      after_retry.write_bytes == after_failure.write_bytes + logical_bytes &&
      observe_output(test_case) && test_case.output_backing->tail_poisoned();
  if (!valid) {
    std::fprintf(
        stderr,
        "Graph pointwise retry backend=%u failed=%.*s retried=%.*s "
        "writes=%llu/%llu failure=%llu partial=%llu recovery=%llu\n",
        static_cast<unsigned>(backend), static_cast<int>(failed.error().size()),
        failed.error().data(), static_cast<int>(retried.error().size()),
        retried.error().data(),
        static_cast<unsigned long long>(after_failure.write_count),
        static_cast<unsigned long long>(after_retry.write_count),
        static_cast<unsigned long long>(after_failure.write_failure_count),
        static_cast<unsigned long long>(after_failure.partial_write_bytes),
        static_cast<unsigned long long>(failed_recovery));
    return 6;
  }
  return 0;
}

} // namespace rund_node_test_virtual::product::graph_pointwise
