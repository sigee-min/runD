#include "local.hpp"

#include "src/compute/device/residency/registry.hpp"

#include <array>
#include <cstddef>
#include <limits>

namespace rund_node_test_pipeline_residency {

int CheckAuthorityMigration() {
  using namespace rund::compute::detail::residency;
  const auto never = std::numeric_limits<std::uint64_t>::max();
  // A Device dirty source and Host-output destination publish as one
  // migration transaction. Write-only admission maps without a fake fetch,
  // and a wrong token mutates neither side.
  Authority migration;
  std::uint32_t migration_device = 99u;
  std::uint32_t migration_output = 99u;
  if (!migration.register_frames(FrameTier::Device, FrameRole::Output, 1u,
                                 migration_device) ||
      !migration.register_frames(FrameTier::Host, FrameRole::Output, 1u,
                                 migration_output)) {
    return 30;
  }
  const std::array source_use{CacheUse{
      .key = {.backing = 40u, .version = 1u, .page = 0u},
      .access = Access::Write,
      .next_use = never,
      .dirty = {.offset = 0u, .bytes = 4096u},
  }};
  const AuthorityResult source = migration.begin(
      source_use, FrameTier::Device, FrameRole::Output, migration_device, 1u);
  if (!source || !migration.activate(source.lease.token) ||
      !migration.complete(source.lease.token, true)) {
    return 31;
  }
  const AuthorityResult source_migration = migration.begin_migration(
      std::array{source_use[0].key}, migration_device, 1u);
  const std::array output_use{CacheUse{
      .key = source_use[0].key,
      .access = Access::Write,
      .next_use = never,
      .dirty = source_use[0].dirty,
  }};
  if (!source_migration ||
      migration
              .begin(output_use, FrameTier::Host, FrameRole::Input,
                     migration_output, 1u)
              .failure != AuthorityFailure::Invalid) {
    return 32;
  }
  const std::array wrong_output_use{CacheUse{
      .key = {.backing = 40u, .version = 1u, .page = 1u},
      .access = Access::Write,
      .next_use = never,
      .dirty = {.offset = 4096u, .bytes = 4096u},
  }};
  const AuthorityResult wrong_destination =
      migration.begin(wrong_output_use, FrameTier::Host, FrameRole::Output,
                      migration_output, 1u);
  if (!wrong_destination ||
      !migration.activate(wrong_destination.lease.token) ||
      migration.complete_migration(source_migration.lease.token,
                                   wrong_destination.lease.token) ||
      !migration.complete(wrong_destination.lease.token, false, true)) {
    return 33;
  }
  const AuthorityResult destination = migration.begin(
      output_use, FrameTier::Host, FrameRole::Output, migration_output, 1u);
  if (!destination || destination.lease.bindings.front().fetch ||
      !destination.lease.bindings.front().prior_dirty.empty() ||
      destination.lease.transitions.size() != 1u ||
      destination.lease.transitions.front().kind != TransitionKind::Map ||
      !migration.activate(destination.lease.token) ||
      migration.complete_migration(source_migration.lease.token,
                                   destination.lease.token + 1u) ||
      !migration.complete_migration(source_migration.lease.token,
                                    destination.lease.token)) {
    return 34;
  }
  const AuthorityResult migrated_dirty = migration.begin_writeback(
      std::array{output_use[0].key}, migration_output, 1u);
  const std::array migration_regions{
      FrameRegion{.tier = FrameTier::Device,
                  .role = FrameRole::Output,
                  .first = migration_device,
                  .count = 1u},
      FrameRegion{.tier = FrameTier::Host,
                  .role = FrameRole::Output,
                  .first = migration_output,
                  .count = 1u},
  };
  if (migration.release_frames(migration_regions) || !migrated_dirty ||
      !migration.discard(migrated_dirty.lease.token)) {
    return 35;
  }

  // An existing dirty Host-output frame must be written back or discarded
  // before another migration can overwrite it, even when the CacheKey is the
  // same and admission therefore emits no replacement transition.
  const AuthorityResult old_output = migration.begin(
      output_use, FrameTier::Host, FrameRole::Output, migration_output, 1u);
  if (!old_output || !migration.activate(old_output.lease.token) ||
      !migration.complete(old_output.lease.token, true)) {
    return 36;
  }
  const AuthorityResult source_again = migration.begin(
      source_use, FrameTier::Device, FrameRole::Output, migration_device, 1u);
  if (!source_again || !migration.activate(source_again.lease.token) ||
      !migration.complete(source_again.lease.token, true)) {
    return 36;
  }
  const AuthorityResult source_again_migration = migration.begin_migration(
      std::array{source_use[0].key}, migration_device, 1u);
  const AuthorityResult overwrite = migration.begin(
      output_use, FrameTier::Host, FrameRole::Output, migration_output, 1u);
  if (!source_again_migration || !overwrite ||
      overwrite.lease.bindings.front().prior_dirty.empty() ||
      !migration.activate(overwrite.lease.token) ||
      migration.complete_migration(source_again_migration.lease.token,
                                   overwrite.lease.token) ||
      !migration.complete(overwrite.lease.token, false) ||
      !migration.discard(source_again_migration.lease.token)) {
    return 37;
  }
  const AuthorityResult old_output_writeback = migration.begin_writeback(
      std::array{output_use[0].key}, migration_output, 1u);
  if (!old_output_writeback ||
      !migration.discard(old_output_writeback.lease.token) ||
      !migration.release_frames(migration_regions)) {
    return 38;
  }
  return 0;
}

} // namespace rund_node_test_pipeline_residency
