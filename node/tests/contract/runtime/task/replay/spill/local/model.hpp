#pragma once

#include "src/runtime/replay/host/payload/backend.hpp"
#include "src/runtime/replay/host/payload/store.hpp"

#include <rund/host/event.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string_view>
#include <vector>

namespace replay_spill {

using rund::StableHash;
using rund::host::EventKind;
using rund::node::replay_detail::payload::Binding;
using rund::node::replay_detail::payload::ResolveResult;
using rund::node::replay_detail::payload::Store;
using ::rund::replay::Storage;

[[nodiscard]] std::vector<std::byte> Bytes(std::string_view text);
[[nodiscard]] std::filesystem::path TempDir(std::string_view name);
[[nodiscard]] std::vector<std::filesystem::path>
GenerationDirectories(const std::filesystem::path &root);
[[nodiscard]] std::filesystem::path
SegmentPath(const std::filesystem::path &dir, std::uint32_t index);
[[nodiscard]] ::rund::replay::Storage
Storage(const std::filesystem::path &dir, std::uint64_t segment_bytes = 80u);
[[nodiscard]] ::rund::replay::Storage
CacheStorage(const std::filesystem::path &dir);
[[nodiscard]] Store Prepared(::rund::replay::Storage storage,
                             std::uint32_t hosts = 4096u,
                             std::uint32_t inputs = 1024u);
[[nodiscard]] bool CorruptLastByte(const std::filesystem::path &path);
[[nodiscard]] std::uint64_t
SegmentBytes(const std::filesystem::path &generation);
[[nodiscard]] std::uint64_t
AllocatedBytes(const std::filesystem::path &generation);
int CanonicalSegment(const std::filesystem::path &segment,
                     std::span<const std::byte> payload, StableHash hash);

int RunSegmentsContract();
int RunRejectContract();
int RunCacheContract();
int GenerationContract();
int LifetimeContract();
int BudgetContract();
int AppendContract();
int ArtifactContract();
int LayoutContract();

} // namespace replay_spill
