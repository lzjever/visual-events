# No Motion SDK Audit

This audit pins the Visual Events Step 1 boundary: the CLI and DDS bridge may move camera facts and visual target facts, but they must not depend on robot motion-control SDKs or publish robot control commands.

## Forbidden dependencies and calls

The CLI and DDS bridge must not link, import, instantiate, or call APIs, message types, clients, topics, or helpers associated with motion control. The explicit blacklist is:

- `rt/lowcmd`
- `rt/arm_sdk`
- `LowCmd`
- `MotorCmd`
- `SportModeCmd`
- `MotionSwitcherClient`
- `look_at`
- `head_position`
- `yaw_velocity`
- `pitch_velocity`
- `motor_command`

These tokens are forbidden in runtime CLI/DDS bridge implementation paths except inside audits, tests, or documentation that enforce this boundary.

## Allowed contracts

The allowed Step 1 DDS contract names are:

- `CameraFrame_`
- `HeadStateV1_`
- `GazeTargetV1_`

`CameraFrame_` is read as the JPEG camera source. `HeadStateV1_` is a read-only observation of current head state. `GazeTargetV1_` is a target fact derived from fresh `visual_state.attention`; it is not a command.

## Runtime boundary

The CLI and DDS bridge must not publish motor or head control command topics. They must not send low-level motor commands, arm SDK commands, sport mode commands, look-at requests, head position setpoints, yaw velocity setpoints, pitch velocity setpoints, or any other motion-control request.

The only permitted head-related data flow is read-only `HeadStateV1_` input and `GazeTargetV1_` target fact output. Any downstream component that chooses to move hardware must live outside this Visual Events CLI/DDS bridge contract and must consume these facts through its own separately audited safety boundary.

## Native bridge audit commands

Run the repo-local native bridge audit without scanning Unitree SDK includes, because SDK headers contain unrelated motion-control IDL names:

```bash
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv \
  uv run --group dev pytest -q tests/unit/test_native_dds_bridge_foundation.py
```

The native source/binary boundary can also be inspected directly:

```bash
rg -n 'LowCmd|MotorCmd|SportModeCmd|MotionSwitcherClient|look_at|head_position|yaw_velocity|pitch_velocity|motor_command|rt/lowcmd|rt/arm_sdk' native/dds_bridge
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv \
  uv run --group dev python tools/build_dds_bridge.py \
  --check --build --probe \
  --unitree-sdk-root "$UNITREE_SDK_ROOT" \
  --video-dds-publisher-dir "$VIDEO_DDS_PUBLISHER_DIR" \
  --out artifacts/dds_bridge/foundation-report.json
ldd build/dds_bridge/visual_events_dds_bridge_probe
readelf -d build/dds_bridge/visual_events_dds_bridge_probe
```

The foundation build/probe gate does not require an IDL generator and its report must distinguish `foundation_ready` from `visual_events_codegen_ready`. The dry-run codegen check must not download, build, write files, or claim that generation/oracle passed:

```bash
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv \
  uv run --group dev python tools/prepare_dds_codegen_toolchain.py \
  --check --dry-run \
  --idlc "$VISUAL_EVENTS_IDLC"
```

The generator oracle hardening check is explicit and writes only under ignored repo `build/`. It runs real C++ idlc codegen and requires the expected `.hpp` and `.cpp` probe outputs:

```bash
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv \
  uv run --group dev python tools/prepare_dds_codegen_toolchain.py \
  --probe-codegen \
  --idlc "$VISUAL_EVENTS_IDLC"
```

The full bridge type-support/codegen gate is explicit, does not search `PATH`, and reuses the same codegen probe; pass the same pinned `idlc` with `--idlc` or `VISUAL_EVENTS_IDLC`:

```bash
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv \
  uv run --group dev python tools/build_dds_bridge.py \
  --check --check-full-bridge \
  --idlc "$VISUAL_EVENTS_IDLC" \
  --unitree-sdk-root "$UNITREE_SDK_ROOT" \
  --video-dds-publisher-dir "$VIDEO_DDS_PUBLISHER_DIR" \
  --out artifacts/dds_bridge/full-bridge-toolchain-report.json
```

If neither `--idlc` nor `VISUAL_EVENTS_IDLC` is provided, if the generator is not pinned to 0.10.2, if stdout/stderr contains `cannot load generator` or `cannot load generator cxx`, or if the probe does not write the expected `.hpp` and `.cpp`, the full bridge gate must fail fast even when idlc returns 0. The foundation gate can still pass.

This only proves generator oracle hardening. It does not prove that a real CycloneDDS 0.10.2 toolchain is prepared, does not generate or connect `HeadStateV1_`/`GazeTargetV1_` type support, and does not complete full bridge/runtime/PC E2E/RK/real-device validation.

The `ldd`/`readelf` commands are only meaningful after a local CMake probe build. Build output must stay under ignored `build/`; reports must stay under ignored `artifacts/`.
