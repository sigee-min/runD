#pragma once

#include <string>

namespace rund::node::accel::detail {

[[nodiscard]] std::string NumericBaseSource();
[[nodiscard]] std::string NumericBaseSource64();
[[nodiscard]] std::string MatrixSource();
[[nodiscard]] std::string TransformSource();
[[nodiscard]] std::string FactorSource();
[[nodiscard]] std::string SolveSource();
[[nodiscard]] std::string SpectrumSource();
[[nodiscard]] std::string MatrixSource64();
[[nodiscard]] std::string TransformSource64();
[[nodiscard]] std::string FactorSource64();
[[nodiscard]] std::string SolveSource64();
[[nodiscard]] std::string SpectrumSource64();

} // namespace rund::node::accel::detail
