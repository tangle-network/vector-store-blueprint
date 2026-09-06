# Vector Store Blueprint

Read [README.md](README.md) for setup and [operator/Cargo.toml](operator/Cargo.toml) for supported dependencies.
Keep common payment validation, health, and metrics in `tangle-inference-core`.

For backend behavior, inspect [store.rs](operator/src/store.rs).
For pricing, limits, and subscription tiers, inspect [config.rs](operator/src/config.rs) and [VectorStoreBSM.sol](contracts/src/VectorStoreBSM.sol).
[server.rs](operator/src/server.rs) owns HTTP authorization and cost calculations, including minimum charges and administrative operations.
Do not infer complete storage settlement from configuration or a successful write alone.

Verify API changes through the actual server, including access rejection, failed operations, and payment handling.
Use the in-memory backend only for its documented development or test role.
Exercise contract changes with actual deployments under `contracts/`.
