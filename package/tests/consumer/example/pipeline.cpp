#include "pipeline/local.hpp"

#include <rund/compute.hpp>

int main() {
  auto opened = rund::compute::open(rund::compute::Target::cpu(2u));
  if (!opened) {
    return opened.exit_code();
  }
  if (const int result =
          rund::package_example::pipeline::CheckDependentWide(*opened);
      result != 0) {
    return result;
  }
  if (const int result =
          rund::package_example::pipeline::CheckMultiInput(*opened);
      result != 0) {
    return result;
  }
  if (const int result = rund::package_example::pipeline::CheckBounded(*opened);
      result != 0) {
    return result;
  }
  if (const int result =
          rund::package_example::pipeline::CheckRecordInSession(*opened);
      result != 0) {
    return result;
  }
  if (const int result =
          rund::package_example::pipeline::CheckMultiOutputAlias(*opened);
      result != 0) {
    return result;
  }
  if (const int result =
          rund::package_example::pipeline::CheckRecurrence(*opened);
      result != 0) {
    return result;
  }
  if (const int result =
          rund::package_example::pipeline::CheckHostFeedback(*opened);
      result != 0) {
    return result;
  }
  if (const int result =
          rund::package_example::pipeline::CheckActionFreeWindowOutput(*opened);
      result != 0) {
    return result;
  }
  if (const int result =
          rund::package_example::pipeline::CheckReusableCheckpoint(*opened);
      result != 0) {
    return result;
  }
  if (const int result =
          rund::package_example::pipeline::CheckNestedResidentRecurrence(
              *opened);
      result != 0) {
    return result;
  }
  return 0;
}
