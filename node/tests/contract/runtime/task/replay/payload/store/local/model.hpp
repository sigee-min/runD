#pragma once

#include "src/runtime/replay/host/payload/backend.hpp"
#include "src/runtime/replay/host/payload/materialize.hpp"

#include <rund/host/event.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace replay_payload_store {

using Contract = int (*)();
using rund::StableHash;
using rund::host::Event;
using rund::host::EventKind;
using rund::host::hash_bytes;
using rund::host::Status;
using rund::node::replay_detail::payload::Archive;
using rund::node::replay_detail::payload::Build;
using rund::node::replay_detail::payload::Capture;
using rund::node::replay_detail::payload::Codec;
using rund::node::replay_detail::payload::Store;

[[nodiscard]] std::vector<std::byte> Payload(std::string_view text);
[[nodiscard]] Store Prepared(std::uint32_t hosts = 4096u,
                             std::uint32_t inputs = 1024u,
                             std::uint64_t bytes = 4u * 1024u * 1024u,
                             ::rund::replay::Storage storage = {},
                             ::rund::replay::Diagnostic diagnostic = {});
[[nodiscard]] std::uint64_t Identity(const Archive &archive);
[[nodiscard]] int Run(std::span<const Contract> contracts);

int MemoryContract();
int ArchiveContract();
int PublishContract();
int InputContract();
int DiagnosticContract();
int MaterializationContract();

} // namespace replay_payload_store
