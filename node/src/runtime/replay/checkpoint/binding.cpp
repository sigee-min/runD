#include "local.hpp"

namespace rund::replay {

Resume Binding::resume(const Checkpoint &checkpoint) const noexcept {
  Code binding = code();
  if (binding == Code::Ok && !checkpointable_) {
    binding = Code::StateSchemaInvalid;
  }
  if (binding == Code::Ok) {
    binding = checkpoint.code();
  }
  if (binding == Code::Ok && checkpoint.schema() != schema_) {
    binding = Code::StateSchemaInvalid;
  }
  return Resume{checkpoint, restore_, binding};
}

} // namespace rund::replay
