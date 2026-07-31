#pragma once

#include "../state.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace rund::node::accel::detail {

[[nodiscard]] std::shared_ptr<void> LookupMetalNamedPipeline(
    MetalAdapter& adapter,
    std::string_view key);

void StoreMetalNamedPipeline(MetalAdapter& adapter,
                             std::string key,
                             std::shared_ptr<void> pipeline,
                             std::uint64_t create_ns);

[[nodiscard]] std::shared_ptr<void> LookupMetalSourceLibrary(
    MetalAdapter& adapter,
    std::string_view source);

[[nodiscard]] std::shared_ptr<void> PublishMetalSourceLibrary(
    MetalAdapter& adapter,
    std::string source,
    std::shared_ptr<void> library,
    std::uint64_t compile_ns);

}  // namespace rund::node::accel::detail
