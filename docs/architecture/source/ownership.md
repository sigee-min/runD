# Source Ownership

Source layout follows semantic ownership, not measured line or include counts.
A file should own one behavior, state transition, or integration boundary. A
split is useful only when the new files have independent responsibilities;
routers, anchor functions, and forwarding layers do not establish ownership.

Public interfaces and private implementations remain separate. Private sibling
declarations may use a local header, but local headers are not package surface.
Dependency direction is enforced by compilable interfaces, link boundaries,
package consumers, and runtime contracts.

Large files are reviewed by meaning. A change should split an owner when it
mixes unrelated lifecycle, scheduling, validation, execution, or evidence
decisions.

## Path Meaning

Code leaves name one direct concept. When two independent concepts need to
identify an owner, their order is expressed by directories rather than by
concatenating the words or appending a width or phase number. Established
domain tokens such as `checkpoint`, `overflow`, and `u64` remain valid. New
names receive semantic review in the owning subsystem; a byte-identity tool
does not infer natural-language boundaries.

## Header Dependency Ownership

Internal headers include the narrow header that owns every value in their
interface. A type aggregate is not an implementation dependency: a consumer of
`ComputePlan`, for example, includes its type owner rather than the planning,
lowering, reference, and validation aggregate. Complete by-value members,
arrays, and template element types use their real definitions; forward
declarations may not stand in for an owning include.

A scheduler-wide state header owns stored runtime state, not every public
operation value named by a member function. Public request, result, budget, and
configuration values that appear only in function declarations are forward
declared; the operation owner includes their definitions. In particular, the
network many-readiness surface is not an edge of the scheduler state storage
graph. A readiness API edit therefore dirties only the reactor and host-I/O
closure, not unrelated lanes, channels, or task execution owners.

This rule makes invalidation semantic. For the include graph `G = (V, E)`, a
changed header `h` dirties the translation units in its reverse-reachable set
`R(h)`. An aggregate edge makes that set the union of every imported owner's
set. Leaf-owner edges limit it to
`R(h) = union(R(o) for o in actual owners used by h)`. Rebuild reduction claims
must compare those exact dirty sets, and timing claims require the same source
manifest and build route; the graph law alone is not performance evidence.
