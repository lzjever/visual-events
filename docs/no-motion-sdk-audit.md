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
