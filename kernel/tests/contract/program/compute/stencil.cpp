#include "stencil/local.hpp"

namespace program_compute_contract {

int RunStencilContract() {
  if (StencilReject() != 0) {
    return 1;
  }
  if (StencilIdentity() != 0) {
    return 1;
  }
  if (StencilShape() != 0) {
    return 1;
  }
  if (StencilReference() != 0) {
    return 1;
  }
  return 0;
}

}  // namespace program_compute_contract
