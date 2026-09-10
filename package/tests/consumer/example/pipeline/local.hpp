#pragma once

#include <rund/compute.hpp>

namespace rund::package_example::pipeline {

[[nodiscard]] int CheckDependentWide(rund::compute::Device &);
[[nodiscard]] int CheckMultiInput(rund::compute::Device &);
[[nodiscard]] int CheckBounded(rund::compute::Device &);
[[nodiscard]] int CheckRecordInSession(rund::compute::Device &);
[[nodiscard]] int CheckMultiOutputAlias(rund::compute::Device &);
[[nodiscard]] int CheckRecurrence(rund::compute::Device &);
[[nodiscard]] int CheckHostFeedback(rund::compute::Device &);
[[nodiscard]] int CheckActionFreeWindowOutput(rund::compute::Device &);
[[nodiscard]] int CheckReusableCheckpoint(rund::compute::Device &);
[[nodiscard]] int CheckNestedResidentRecurrence(rund::compute::Device &);

} // namespace rund::package_example::pipeline
