#pragma once

#include "../../kernel/backend/source_recipe.hpp"

#include <cstdint>
#include <string>

namespace rund::node::accel::detail {

[[nodiscard]] std::string NumericBaseSource();
[[nodiscard]] std::string NumericBaseSource64();
[[nodiscard]] bool EmitNumericBaseSource(
    backend_source_recipe::CountSink &sink, bool wide) noexcept;
[[nodiscard]] bool EmitNumericBaseSource(
    backend_source_recipe::StringSink &sink, bool wide);
[[nodiscard]] bool NumericBaseSourceBytes(bool wide,
                                          std::uint64_t &bytes) noexcept;
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
[[nodiscard]] bool MatrixSourceBytes(std::uint64_t &bytes) noexcept;
[[nodiscard]] bool TransformSourceBytes(std::uint64_t &bytes) noexcept;
[[nodiscard]] bool FactorSourceBytes(std::uint64_t &bytes) noexcept;
[[nodiscard]] bool SolveSourceBytes(std::uint64_t &bytes) noexcept;
[[nodiscard]] bool SpectrumSourceBytes(std::uint64_t &bytes) noexcept;
[[nodiscard]] bool MatrixSource64Bytes(std::uint64_t &bytes) noexcept;
[[nodiscard]] bool TransformSource64Bytes(std::uint64_t &bytes) noexcept;
[[nodiscard]] bool FactorSource64Bytes(std::uint64_t &bytes) noexcept;
[[nodiscard]] bool SolveSource64Bytes(std::uint64_t &bytes) noexcept;
[[nodiscard]] bool SpectrumSource64Bytes(std::uint64_t &bytes) noexcept;

} // namespace rund::node::accel::detail
