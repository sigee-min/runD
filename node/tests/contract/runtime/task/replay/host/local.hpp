#pragma once

#include "../local.hpp"
#include "src/runtime/replay/host/payload/materialize.hpp"

[[nodiscard]] inline rund::node::replay_detail::payload::Materialization
MaterializePayloadArchive(const rund::node::replay_detail::payload::Archive &archive,
                          const ::rund::replay::Storage storage = {}) {
  rund::node::replay_detail::payload::BuildResult built =
      rund::node::replay_detail::payload::Build(archive, storage);
  if (!built.ok()) {
    return rund::node::replay_detail::payload::Materialization{};
  }
  return rund::node::replay_detail::payload::Materialize(built.store);
}
