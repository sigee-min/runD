#include "context/local.hpp"
#include "context/forgery/run.hpp"

namespace node_accel_contract {

bool PublicContextApiContract() {
  return ContextRejectsInvalidInputs() &&
         PublicContextApiOpensRealBackend() &&
         PublicContextApiCreatesAndTransfers() &&
         PublicContextApiRejectsForgedBuffers();
}

}  // namespace node_accel_contract
