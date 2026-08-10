#pragma once

#include "../state.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace rund::node::accel::detail {

[[nodiscard]] std::shared_ptr<void>
LookupMetalNamedPipeline(MetalAdapter &adapter, std::string_view key);
[[nodiscard]] std::shared_ptr<void>
LookupMetalNamedPipeline(MetalAdapter &adapter, std::string_view key,
                         rund::AccelRunFacts *local);

enum class MetalNamedPipelinePublishStatus : std::uint8_t {
  Inserted,
  Existing,
  Failed,
};

struct MetalNamedPipelinePublishResult final {
  MetalNamedPipelinePublishStatus status{
      MetalNamedPipelinePublishStatus::Failed};
  std::shared_ptr<void> pipeline{};
};

[[nodiscard]] MetalNamedPipelinePublishResult PublishMetalNamedPipeline(
    MetalAdapter &adapter, std::string key, std::shared_ptr<void> pipeline,
    std::uint64_t create_ns, rund::AccelRunFacts *local = nullptr) noexcept;

void StoreMetalNamedPipeline(MetalAdapter &adapter, std::string key,
                             std::shared_ptr<void> pipeline,
                             std::uint64_t create_ns,
                             rund::AccelRunFacts *local = nullptr);

void RecordMetalUncachedPipelineCompile(MetalAdapter &adapter,
                                        std::uint64_t create_ns) noexcept;

[[nodiscard]] std::shared_ptr<void>
LookupMetalSourceLibrary(MetalAdapter &adapter, std::string_view source);

enum class MetalSourceLibraryPublishStatus : std::uint8_t {
  Inserted,
  Existing,
  Failed,
};

struct MetalSourceLibraryPublishResult final {
  MetalSourceLibraryPublishStatus status{
      MetalSourceLibraryPublishStatus::Failed};
  std::shared_ptr<void> library{};
};

[[nodiscard]] MetalSourceLibraryPublishResult PublishMetalSourceLibrary(
    MetalAdapter &adapter, std::string source, std::shared_ptr<void> library,
    std::uint64_t compile_ns, rund::AccelRunFacts *local = nullptr) noexcept;

void RecordMetalUncachedLibraryCompile(MetalAdapter &adapter,
                                       std::uint64_t compile_ns) noexcept;

} // namespace rund::node::accel::detail
