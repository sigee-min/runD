#include "internal.hpp"

#include "src/compute/virtual/backing.hpp"

#include <limits>

namespace rund_node_test_virtual::product::graph_pointwise_multi {

int check_scale(const rund::compute::Device &device,
                const rund::compute::Backend backend,
                const std::size_t page_count) {
  Preparation prepared = prepare_case(device, page_count);
  if (!prepared.value) {
    return prepared.reason;
  }
  Case &test_case = *prepared.value;
  const std::uint64_t initial_version =
      rund::compute::detail::VirtualBackingAccess::version(
          *test_case.output_backing);
  const auto before = stage_generations(test_case);
  const rund::compute::Status status = test_case.pipeline.run();
  const auto after = stage_generations(test_case);
  bool published = true;
  for (std::size_t stage = 0u; stage < StageCount; ++stage) {
    published = published &&
                before[stage] != std::numeric_limits<std::uint64_t>::max() &&
                after[stage] == before[stage] + 1u;
  }
  SuccessReport report{.before_generations = before,
                       .after_generations = after,
                       .published = published};
  if (!validate_success(test_case, backend, status, initial_version, published,
                        report)) {
    report_failure(report, backend, status, page_count, "pointwise");
    return 5;
  }
  const std::uint64_t warm_version =
      rund::compute::detail::VirtualBackingAccess::version(
          *test_case.output_backing);
  const BackingFacts warm_before = test_case.output_backing->facts();
  const auto warm_generations = after;
  const rund::compute::Status warm = test_case.pipeline.run();
  const auto warm_after = stage_generations(test_case);
  SuccessReport warm_report{.before_generations = warm_generations,
                            .after_generations = warm_after,
                            .published = true};
  if (!validate_warm(test_case, backend, warm, warm_version, warm_before,
                     warm_generations, warm_after, warm_report)) {
    report_failure(warm_report, backend, warm, page_count, "warm");
    return 6;
  }
  return 0;
}

} // namespace rund_node_test_virtual::product::graph_pointwise_multi
