#include "local.hpp"

namespace program_skeleton_contract {

int ExpectReason(const char *const expected,
                 const rund::kernel::SkeletonResult result) {
  TEST_ASSERT(!result.ok);
  TEST_ASSERT(std::string_view{result.reason} == expected);
  return 0;
}

} // namespace program_skeleton_contract
