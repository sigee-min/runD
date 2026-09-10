#pragma once

namespace package_compute::fixed_contract {

[[nodiscard]] int CheckArithmetic();
[[nodiscard]] int CheckDeclaredMultiply();
[[nodiscard]] int CheckMixedPolicy();
[[nodiscard]] int CheckPolicyArithmetic();
[[nodiscard]] int CheckRejections();
[[nodiscard]] int CheckRescale();

} // namespace package_compute::fixed_contract
