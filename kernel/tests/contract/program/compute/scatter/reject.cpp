#include "local.hpp"

namespace program_compute_contract {

int ScatterReject() {
  rund::kernel::ScatterDesc zero_count = U32Scatter();
  zero_count.element_count = 0u;
  const rund::kernel::ScatterPlan zero_count_plan =
      rund::kernel::PlanScatter(zero_count);
  TEST_ASSERT(!zero_count_plan.ok);
  TEST_ASSERT(std::string_view{zero_count_plan.reason} ==
              "compute_scatter_count_zero");

  rund::kernel::ScatterDesc encoded = U32Scatter();
  encoded.element_count =
      rund::kernel::scatter_plan_detail::kMaxEncodedCount + 1u;
  const rund::kernel::ScatterPlan encoded_plan =
      rund::kernel::PlanScatter(encoded);
  TEST_ASSERT(!encoded_plan.ok);
  TEST_ASSERT(std::string_view{encoded_plan.reason} ==
              "compute_scatter_count_unsupported");

  rund::kernel::ScatterDesc zero_output = U32Scatter();
  zero_output.output_count = 0u;
  const rund::kernel::ScatterPlan zero_output_plan =
      rund::kernel::PlanScatter(zero_output);
  TEST_ASSERT(!zero_output_plan.ok);
  TEST_ASSERT(std::string_view{zero_output_plan.reason} ==
              "compute_scatter_output_zero");

  rund::kernel::ScatterDesc unaddressable = U32Scatter();
  unaddressable.output_count =
      rund::kernel::scatter_plan_detail::kMaxOutputCount + 1u;
  const rund::kernel::ScatterPlan unaddressable_plan =
      rund::kernel::PlanScatter(unaddressable);
  TEST_ASSERT(!unaddressable_plan.ok);
  TEST_ASSERT(std::string_view{unaddressable_plan.reason} ==
              "compute_scatter_output_unsupported");

  rund::kernel::ScatterDesc unknown_element = U32Scatter();
  unknown_element.element = static_cast<rund::kernel::ScatterElement>(0u);
  const rund::kernel::ScatterPlan unknown_element_plan =
      rund::kernel::PlanScatter(unknown_element);
  TEST_ASSERT(!unknown_element_plan.ok);
  TEST_ASSERT(std::string_view{unknown_element_plan.reason} ==
              "compute_scatter_element_unsupported");

  return 0;
}

} // namespace program_compute_contract
