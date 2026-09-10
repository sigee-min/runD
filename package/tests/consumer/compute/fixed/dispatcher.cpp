#include "local.hpp"

#include <rund/compute.hpp>

namespace package_compute {

int Fixed() {
  using namespace fixed_contract;
  if (const int result = CheckArithmetic(); result != 0) {
    return result;
  }
  if (const int result = CheckDeclaredMultiply(); result != 0) {
    return result;
  }
  if (const int result = CheckMixedPolicy(); result != 0) {
    return result;
  }
  if (const int result = CheckPolicyArithmetic(); result != 0) {
    return result;
  }
  if (const int result = CheckRejections(); result != 0) {
    return result;
  }
  return CheckRescale();
}

} // namespace package_compute
