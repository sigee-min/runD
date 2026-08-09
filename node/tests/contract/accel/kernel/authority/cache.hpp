#pragma once

namespace node_accel_contract {

[[nodiscard]] bool PreparedTemplateRegistryIsColdAndCollisionSafe();
[[nodiscard]] bool RecurrenceRouteCopiesDoNotCloneTemplates();
[[nodiscard]] bool MetalPointerIdentityIndexIsExactAndOneShot();

} // namespace node_accel_contract
