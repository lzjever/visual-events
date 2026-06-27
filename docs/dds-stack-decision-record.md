# DDS Stack Decision Record

File: `docs/dds-stack-decision-record.md`

Status: frozen for GA implementation handoff
Date: 2026-06-27
Owner: Visual Events CLI/DDS implementation handoff owner

## Decision

Use a small C++ native DDS helper/bridge for GA DDS runtime access. Runtime choice: C++ native DDS helper/bridge using Unitree SDK2 Channel API + CycloneDDS/CycloneDDS-CXX. The bridge owns all direct DDS SDK linkage. The Python CLI talks to that helper through a controlled local process/IPC boundary; Python must not directly import or link the DDS SDK.

This is a runtime stack decision only. Real DDS factories/adapters, generated type support, bridge binary, PC E2E, and board compatibility are not implemented by this decision record.

## Version Evidence

- Unitree SDK2 2.0.0 is the selected SDK version. Local evidence: `/home/galbot/works/et1/third_party/unitree_sdk2_install/lib/cmake/unitree_sdk2/unitree_sdk2ConfigVersion.cmake` contains `PACKAGE_VERSION "2.0.0"`.
- CycloneDDS 0.10.2 is the selected DDS runtime version. Local evidence: `/home/galbot/works/et1/third_party/unitree_sdk2_install/include/dds/version.h` contains `DDS_VERSION "0.10.2"` and `DDS_PROJECT_NAME "CycloneDDS"`.
- The aarch64 package exists at `/home/galbot/works/et1/third_party/unitree_sdk2_install_aarch64/...` with the same Unitree SDK2 and CycloneDDS version facts. Local `file` output shows ARM aarch64 libraries for `libddsc.so` and `libddscxx.so`. This is package evidence only, not board validation.

## CameraFrame_ Source

`CameraFrame_` comes from the existing video DDS publisher artifacts:

- `/home/galbot/works/video_dds_publisher/idl/CameraFrame_.idl`
- `/home/galbot/works/video_dds_publisher/include/unitree_camera/msg/dds/CameraFrame_.hpp`
- `/home/galbot/works/video_dds_publisher/src/CameraFrame_.cpp`

`/home/galbot/works/image-capture/src/main.cpp` already subscribes to the camera topic using `unitree::robot::ChannelSubscriber<DdsCameraFrame>`, equivalent to `ChannelSubscriber<CameraFrame_>` after the local alias where `DdsCameraFrame` is `unitree_camera::msg::dds_::CameraFrame_`.

## Topics And Environment

Topics:

- camera input: `/camera/image/jpeg`
- gaze output: `/visual_events/gaze_target`
- head state input: `/robot/head_state`

Environment variables:

- `VISUAL_EVENTS_DDS_DOMAIN`
- `VISUAL_EVENTS_DDS_NETWORK`
- `VISUAL_EVENTS_UNITREE_SDK_ROOT`
- `VISUAL_EVENTS_DDS_BRIDGE_BIN`
- `LD_LIBRARY_PATH`

Production defaults are domain `0` and network `eth0`. PC smoke defaults are domain `57` and network `lo`; test runners must still pass domain/network explicitly.

## IDL Codegen Gate

This repo already has authoritative IDL for Visual Events output/input contracts:

- `common/schema/dds/gaze_target_v1.idl`
- `common/schema/dds/head_state_v1.idl`

Current local PATH does not provide `idlc`, `cyclonedds-idlc`, or `fastddsgen`. Therefore IDL codegen is a required next adapter-step input/toolchain gate, not evidence that codegen is already runnable or complete.

Required next-step command shape:

```bash
<cyclonedds-idlc-or-approved-generator> \
  common/schema/dds/gaze_target_v1.idl \
  common/schema/dds/head_state_v1.idl \
  --output <bridge-generated-type-support-dir>
```

The adapter step must record the actual generator binary, version, command, generated files, and compatibility test result.

## Native Bridge ABI

Use line-delimited JSON over stdin/stdout as the first stable ABI. Each message is one single-line UTF-8 JSON object terminated by `\n`; raw bytes are forbidden on stdin/stdout. Keep the boundary small; do not design a broad RPC protocol.

Required rules:

- `protocol_version=1` is present on every message.
- Message types are `camera_jpeg`, `head_state`, `gaze_target`, `status`, and `error`.
- Bridge stdout emits `camera_jpeg`, `head_state`, `status`, and `error` messages for the Python CLI.
- Bridge stdin accepts `gaze_target` messages from the Python CLI.
- Message timestamps include monotonic timestamps/ns for freshness decisions. Wall-clock timestamps may be carried only when the DDS contract requires them.
- Camera/head inputs use latest-only semantics with bounded queue/backpressure. Old samples may be dropped; unbounded buffering is forbidden.
- Logs go to stderr only. stdout is reserved for protocol frames.
- Fatal DDS init/runtime errors exit nonzero and emit a final `error` message when possible.
- Unitree Channel API is the only DDS pub/sub API used by the bridge. CycloneDDS/CycloneDDS-CXX is the Unitree SDK2 runtime/type support dependency, not a second direct pub/sub API for bridge code.

Minimum `camera_jpeg` stdout fields:

- `protocol_version`: integer, fixed `1`.
- `type`: string, fixed `camera_jpeg`.
- `dds_timestamp_ns`: integer, copied from `CameraFrame_.timestamp_ns`.
- `received_monotonic_ns`: integer, bridge monotonic receive timestamp.
- `camera_name`: string, copied from DDS source `CameraFrame_.camera_name`.
- `width`: integer, copied from `CameraFrame_.width`.
- `height`: integer, copied from `CameraFrame_.height`.
- `encoding`: string, must be `JPEG`.
- `step`: integer, copied from `CameraFrame_.step`.
- `data_size_bytes`: integer byte length of the JPEG payload before base64 encoding.
- `data_base64`: string base64 encoding of the JPEG bytes.

The Python side must decode `data_base64` and compare the decoded byte length with `data_size_bytes`; mismatch means the sample is dropped and an error/status counter is incremented. Raw JPEG bytes are never written directly to stdout.

Minimum `gaze_target` stdin fields align directly to `GazeTargetPayload` / `GazeTargetV1_` canonical fields:

- `protocol_version`: integer, fixed `1`.
- `type`: string, fixed `gaze_target`.
- `schema_version`: integer, copied from canonical gaze payload, fixed `1` for GA.
- `camera`: string, CLI logical camera name from canonical gaze payload.
- `frame_id`: integer, visual state frame id from canonical gaze payload.
- `frame_timestamp_ms`: integer, visual state frame timestamp from canonical gaze payload.
- `publish_timestamp_ms`: integer, CLI wall-clock publish timestamp from canonical gaze payload.
- `valid`: boolean.
- `state`: string, one of `tracking`, `lost`, `stale`, or `disabled`.
- `target_track_id`: integer, valid target track id or `-1` for invalid samples.
- `target_u`: number, target pixel u/x coordinate.
- `target_v`: number, target pixel v/y coordinate.
- `target_norm_x`: number, normalized horizontal target position.
- `target_norm_y`: number, normalized vertical target position.
- `image_width`: integer, source image width for the target payload.
- `image_height`: integer, source image height for the target payload.
- `confidence`: number, canonical confidence field.
- `reason`: string, canonical target reason field.
- `stale_after_ms`: integer, downstream freshness hint.

The bridge publishes `gaze_target` to DDS without renaming these canonical fields. `received_monotonic_ns` is only for inbound `camera_jpeg` and `head_state`; outbound gaze samples do not add a bridge-specific monotonic timestamp field.

Minimum `head_state` stdout fields are `protocol_version`, `type`, `dds_timestamp_ns`, `received_monotonic_ns`, `valid`, `state`, and the head angle/velocity fields needed by the existing CLI head-motion mapper. Minimum `status` fields are `protocol_version`, `type`, `code`, and `message`. Minimum `error` fields are `protocol_version`, `type`, `code`, `message`, and `fatal`.

## Tests Required Before PC E2E

Existing fake/unit DDS tests are not enough for GA runtime confidence. Before PC E2E can be counted, Step 4 must add:

- real serialization/type compatibility tests for `CameraFrame_`, `GazeTargetV1_`, and `HeadStateV1_`;
- Unitree Channel QoS construction tests for camera, head state, and gaze topics;
- no-motion-SDK audit on the native bridge binary via dependency inspection and topic allowlist.

## PC Install, Build, And Smoke Plan

Expected bridge build inputs:

- `UNITREE_SDK_ROOT` or `VISUAL_EVENTS_UNITREE_SDK_ROOT` pointing to `/home/galbot/works/et1/third_party/unitree_sdk2_install`
- `VIDEO_DDS_PUBLISHER_DIR` pointing to `/home/galbot/works/video_dds_publisher`
- `CMAKE_PREFIX_PATH` including the Unitree SDK2 install root when needed

Expected build shape:

```bash
cmake -S <bridge-src> -B <bridge-build> \
  -DUNITREE_SDK_ROOT="$VISUAL_EVENTS_UNITREE_SDK_ROOT" \
  -DVIDEO_DDS_PUBLISHER_DIR=/home/galbot/works/video_dds_publisher
cmake --build <bridge-build>
```

Runtime must set `VISUAL_EVENTS_DDS_BRIDGE_BIN`, `VISUAL_EVENTS_DDS_DOMAIN`, `VISUAL_EVENTS_DDS_NETWORK`, and `LD_LIBRARY_PATH` so the bridge can load Unitree SDK2/CycloneDDS libraries. The PC loopback smoke plan uses domain `57` and network `lo`.

This section is a required plan for the next adapter step, not passed evidence.

## RK3588 / Board Probe Gate

The aarch64 SDK package exists and local `file` evidence shows ARM aarch64 DDS libraries. GA handoff still requires an aarch64/RK3588 build/probe, or an explicit unsupported fail-fast path, before claiming board compatibility. Do not mark RK3588 supported or GA from package presence alone.

## No-Motion Boundary

The CLI/bridge publishes gaze DDS only. It must not publish motor/head command topics, and it must not contain fields or APIs that directly command head velocity, head position, or motor movement.

The Python CLI and native bridge must not link, import, or call motion/head-control SDK code. Review wording: `visual-events-cli` 和 DDS bridge 不链接、import 或调用运控 SDK；只允许读取 camera/head state DDS and publish gaze target DDS.

## Non-Goals

- No real DDS factories/adapters are implemented by this decision record.
- No native bridge binary is implemented by this decision record.
- No PC E2E pass is claimed by this decision record.
- No board compatibility pass is claimed by this decision record.
- No motion/head-control adapter is introduced by this decision record.
