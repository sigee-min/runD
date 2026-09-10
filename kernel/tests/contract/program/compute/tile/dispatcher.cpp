#include "contract/program/compute/local.hpp"

#include "local.hpp"

namespace program_compute_contract {

int RunComputeTileContract() {
  if (tile_contract::CheckPositiveDispatch() != 0) {
    return 1;
  }
  if (tile_contract::CheckBoundaries() != 0) {
    return 1;
  }
  if (tile_contract::CheckConstCallback() != 0) {
    return 1;
  }
  if (tile_contract::CheckIndependentRunState() != 0) {
    return 1;
  }
  if (tile_contract::CheckBorrowedStorage() != 0) {
    return 1;
  }
  if (tile_contract::CheckBoundedBind() != 0) {
    return 1;
  }
  if (tile_contract::CheckRetainedMemory() != 0) {
    return 1;
  }
  return tile_contract::CheckExplicitBackend();
}

} // namespace program_compute_contract
