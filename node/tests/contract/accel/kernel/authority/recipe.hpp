#pragma once

namespace node_accel_contract {

[[nodiscard]] bool BackendSourceRecipeIsCheckedAndCanonical();
[[nodiscard]] bool VulkanMapAndResetSourceRecipesAreExact();

} // namespace node_accel_contract
