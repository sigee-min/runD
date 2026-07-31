#include "local.hpp"

#include "../../../../../src/compute/graph/describe/model.hpp"

#include <cstdint>
#include <vector>

namespace rund_node_graph_services {
namespace {

using rund::compute::Status;
using rund::compute::detail::graph_detail::describe_detail::build_hazards;
using rund::compute::graph::Access;
using rund::compute::graph::Info;
using rund::compute::graph::Node;
using rund::compute::graph::Operation;
using rund::compute::graph::Resource;
using rund::compute::resource::AccessMode;

[[nodiscard]] bool dense_boundary() {
  constexpr std::uint32_t WriterCount = 660u;
  constexpr std::uint64_t ElementBytes = sizeof(std::uint32_t);
  Info info;
  info.resources.push_back(Resource{
      .id = 1u,
      .elements = WriterCount,
      .element_bytes = ElementBytes,
      .bytes = WriterCount * ElementBytes,
      .alias_group = 1u,
  });
  info.nodes.reserve(WriterCount + 1u);
  for (std::uint32_t node = 0u; node < WriterCount; ++node) {
    info.nodes.push_back(Node{
        .index = node,
        .operation = Operation::Map,
        .elements = 1u,
        .accesses = {Access{
            .resource = 1u,
            .mode = AccessMode::Write,
            .offset_bytes = node * ElementBytes,
            .size_bytes = ElementBytes,
            .element_bytes = ElementBytes,
            .element_count = 1u,
            .stride_bytes = ElementBytes,
        }},
    });
  }
  info.nodes.push_back(Node{
      .index = WriterCount,
      .operation = Operation::Map,
      .elements = WriterCount,
      .accesses = {Access{
          .resource = 1u,
          .mode = AccessMode::Read,
          .offset_bytes = 0u,
          .size_bytes = WriterCount * ElementBytes,
          .element_bytes = ElementBytes,
          .element_count = WriterCount,
          .stride_bytes = ElementBytes,
      }},
  });

  const Status status = build_hazards(info);
  return status && info.nodes.size() == 661u &&
         info.nodes.back().dependencies.size() == WriterCount &&
         info.barriers.size() == 1u &&
         info.barriers.front().before_node == WriterCount - 1u &&
         info.barriers.front().after_node == WriterCount;
}

[[nodiscard]] bool alias_boundary() {
  constexpr std::uint64_t ElementBytes = sizeof(std::uint32_t);
  Info info;
  info.resources = {
      Resource{
          .id = 1u,
          .elements = 1u,
          .element_bytes = ElementBytes,
          .bytes = ElementBytes,
          .alias_group = 1u,
      },
      Resource{
          .id = 2u,
          .elements = 2u,
          .element_bytes = ElementBytes,
          .bytes = 2u * ElementBytes,
          .alias_group = 1u,
      },
  };
  info.nodes = {
      Node{
          .index = 0u,
          .operation = Operation::Map,
          .elements = 1u,
          .accesses = {Access{
              .resource = 1u,
              .mode = AccessMode::Write,
              .offset_bytes = 0u,
              .size_bytes = ElementBytes,
              .element_bytes = ElementBytes,
              .element_count = 1u,
              .stride_bytes = ElementBytes,
          }},
      },
      Node{
          .index = 1u,
          .operation = Operation::Map,
          .elements = 1u,
          .accesses = {Access{
              .resource = 2u,
              .mode = AccessMode::Write,
              .offset_bytes = ElementBytes,
              .size_bytes = ElementBytes,
              .element_bytes = ElementBytes,
              .element_count = 1u,
              .stride_bytes = ElementBytes,
          }},
      },
      Node{
          .index = 2u,
          .operation = Operation::Map,
          .elements = 2u,
          .accesses = {Access{
              .resource = 2u,
              .mode = AccessMode::Read,
              .offset_bytes = 0u,
              .size_bytes = 2u * ElementBytes,
              .element_bytes = ElementBytes,
              .element_count = 2u,
              .stride_bytes = ElementBytes,
          }},
      },
  };

  const Status status = build_hazards(info);
  return status && info.nodes.back().dependencies.size() == 2u &&
         info.barriers.size() == 1u &&
         info.barriers.front().before_resource == 1u &&
         info.barriers.front().after_resource == 2u &&
         info.barriers.front().before_node == 0u &&
         info.barriers.front().after_node == 2u;
}

[[nodiscard]] bool paired_alias_boundary() {
  constexpr std::uint64_t ElementBytes = sizeof(std::uint32_t);
  Info info;
  info.resources = {
      Resource{.id = 1u,
               .elements = 1u,
               .element_bytes = ElementBytes,
               .bytes = ElementBytes,
               .alias_group = 1u},
      Resource{.id = 2u,
               .elements = 2u,
               .element_bytes = ElementBytes,
               .bytes = 2u * ElementBytes,
               .alias_group = 1u},
  };
  info.nodes = {
      Node{
          .index = 0u,
          .operation = Operation::Map,
          .elements = 2u,
          .accesses =
              {
                  Access{.resource = 1u,
                         .mode = AccessMode::Write,
                         .offset_bytes = 0u,
                         .size_bytes = ElementBytes,
                         .element_bytes = ElementBytes,
                         .element_count = 1u,
                         .stride_bytes = ElementBytes},
                  Access{.resource = 2u,
                         .mode = AccessMode::Write,
                         .offset_bytes = ElementBytes,
                         .size_bytes = ElementBytes,
                         .element_bytes = ElementBytes,
                         .element_count = 1u,
                         .stride_bytes = ElementBytes},
              },
      },
      Node{
          .index = 1u,
          .operation = Operation::Map,
          .elements = 2u,
          .accesses = {Access{
              .resource = 2u,
              .mode = AccessMode::Read,
              .offset_bytes = 0u,
              .size_bytes = 2u * ElementBytes,
              .element_bytes = ElementBytes,
              .element_count = 2u,
              .stride_bytes = ElementBytes,
          }},
      },
  };

  const Status status = build_hazards(info);
  return status && info.nodes.back().dependencies.size() == 1u &&
         info.barriers.size() == 1u &&
         info.barriers.front().before_resource == 1u &&
         info.barriers.front().after_resource == 2u &&
         info.barriers.front().before_node == 0u &&
         info.barriers.front().after_node == 1u;
}

} // namespace

bool ValidBoundaryPlan() {
  return dense_boundary() && alias_boundary() && paired_alias_boundary();
}

} // namespace rund_node_graph_services
