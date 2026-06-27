from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
NATIVE_BRIDGE = REPO_ROOT / "native" / "dds_bridge"
TOOLS_BUILD = REPO_ROOT / "tools" / "build_dds_bridge.py"
TOOLS_PREPARE_CODEGEN = REPO_ROOT / "tools" / "prepare_dds_codegen_toolchain.py"
GITIGNORE = REPO_ROOT / ".gitignore"
HEAD_STATE_IDL = REPO_ROOT / "common" / "schema" / "dds" / "head_state_v1.idl"
GAZE_TARGET_IDL = REPO_ROOT / "common" / "schema" / "dds" / "gaze_target_v1.idl"
CYCLONEDDS_COMMIT = "9995905bce6c4cf9f740d6438bbf7fcfd1c83dfd"
CYCLONEDDS_CXX_COMMIT = "2a372d2c4597faea54543b925755fa2d7cdd4232"

ALLOWED_TOPICS = {
    "/camera/image/jpeg",
    "/robot/head_state",
    "/visual_events/gaze_target",
}
ALLOWED_TYPES = {
    "unitree_camera::msg::dds_::CameraFrame_",
    "visual_events::msg::dds_::HeadStateV1_",
    "visual_events::msg::dds_::GazeTargetV1_",
}
DENIED_MOTION_TOKENS = {
    "LowCmd",
    "MotorCmd",
    "SportModeCmd",
    "MotionSwitcherClient",
    "look_at",
    "head_position",
    "yaw_velocity",
    "pitch_velocity",
    "motor_command",
    "rt/lowcmd",
    "rt/arm_sdk",
}


def _repo_native_sources() -> list[Path]:
    assert NATIVE_BRIDGE.is_dir(), "expected native/dds_bridge foundation"
    return sorted(
        path
        for path in NATIVE_BRIDGE.rglob("*")
        if path.suffix in {".cmake", ".txt", ".hpp", ".cpp", ".h", ".cc"}
    )


def _combined_native_source_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in _repo_native_sources())


def _make_minimal_unitree_sdk_root(tmp_path: Path) -> Path:
    root = tmp_path / "unitree-sdk"
    (root / "lib" / "cmake" / "unitree_sdk2").mkdir(parents=True)
    channel_include = root / "include" / "unitree" / "robot" / "channel"
    channel_include.mkdir(parents=True)
    (root / "lib" / "cmake" / "unitree_sdk2" / "unitree_sdk2Config.cmake").write_text(
        "# fake unitree_sdk2 package\n",
        encoding="utf-8",
    )
    for name in ["libunitree_sdk2.a", "libddsc.so", "libddscxx.so"]:
        (root / "lib" / name).write_text("", encoding="utf-8")
    (channel_include / "channel_factory.hpp").write_text(
        "#pragma once\n"
        "\n"
        "#include <cstdint>\n"
        "#include <functional>\n"
        "#include <memory>\n"
        "#include <string>\n"
        "\n"
        "namespace unitree { namespace robot {\n"
        "template<typename MSG>\n"
        "class FakeChannel {\n"
        "public:\n"
        "  bool Write(const MSG&, int64_t = 0) { return true; }\n"
        "  int64_t GetLastDataAvailableTime() const { return -1; }\n"
        "};\n"
        "template<typename MSG>\n"
        "using ChannelPtr = std::shared_ptr<FakeChannel<MSG>>;\n"
        "class ChannelFactory {\n"
        "public:\n"
        "  static ChannelFactory* Instance() { static ChannelFactory inst; return &inst; }\n"
        "  void Init(int32_t domain_id, const std::string& network = \"\") {\n"
        "    domain_id_ = domain_id;\n"
        "    network_ = network;\n"
        "    inited_ = true;\n"
        "  }\n"
        "  void Release() { inited_ = false; }\n"
        "  template<typename MSG>\n"
        "  ChannelPtr<MSG> CreateSendChannel(const std::string&) {\n"
        "    return std::make_shared<FakeChannel<MSG>>();\n"
        "  }\n"
        "  template<typename MSG>\n"
        "  ChannelPtr<MSG> CreateRecvChannel(const std::string&, std::function<void(const void*)>, int32_t = 0) {\n"
        "    return std::make_shared<FakeChannel<MSG>>();\n"
        "  }\n"
        "private:\n"
        "  bool inited_ = false;\n"
        "  int32_t domain_id_ = 0;\n"
        "  std::string network_;\n"
        "};\n"
        "}}\n",
        encoding="utf-8",
    )
    (channel_include / "channel_subscriber.hpp").write_text(
        "#pragma once\n"
        "\n"
        "#include <functional>\n"
        "#include <string>\n"
        "\n"
        "#include \"unitree/robot/channel/channel_factory.hpp\"\n"
        "\n"
        "namespace unitree { namespace robot {\n"
        "template<typename MSG>\n"
        "class ChannelSubscriber {\n"
        "public:\n"
        "  explicit ChannelSubscriber(const std::string& channel_name) : channel_name_(channel_name) {}\n"
        "  void InitChannel(const std::function<void(const void*)>& handler, int64_t queuelen = 0) {\n"
        "    channel_ = ChannelFactory::Instance()->CreateRecvChannel<MSG>(channel_name_, handler, static_cast<int32_t>(queuelen));\n"
        "  }\n"
        "  void CloseChannel() { channel_.reset(); }\n"
        "private:\n"
        "  std::string channel_name_;\n"
        "  ChannelPtr<MSG> channel_;\n"
        "};\n"
        "}}\n",
        encoding="utf-8",
    )
    (channel_include / "channel_publisher.hpp").write_text(
        "#pragma once\n"
        "\n"
        "#include <string>\n"
        "\n"
        "#include \"unitree/robot/channel/channel_factory.hpp\"\n"
        "\n"
        "namespace unitree { namespace robot {\n"
        "template<typename MSG>\n"
        "class ChannelPublisher {\n"
        "public:\n"
        "  explicit ChannelPublisher(const std::string& channel_name) : channel_name_(channel_name) {}\n"
        "  void InitChannel() { channel_ = ChannelFactory::Instance()->CreateSendChannel<MSG>(channel_name_); }\n"
        "  void CloseChannel() { channel_.reset(); }\n"
        "private:\n"
        "  std::string channel_name_;\n"
        "  ChannelPtr<MSG> channel_;\n"
        "};\n"
        "}}\n",
        encoding="utf-8",
    )
    return root


def _make_minimal_video_dds_publisher_dir(tmp_path: Path) -> Path:
    root = tmp_path / "video-dds-publisher"
    header = root / "include" / "unitree_camera" / "msg" / "dds" / "CameraFrame_.hpp"
    source = root / "src" / "CameraFrame_.cpp"
    header.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    header.write_text(
        "#pragma once\n"
        "\n"
        "#include <cstdint>\n"
        "#include <string>\n"
        "#include <utility>\n"
        "#include <vector>\n"
        "\n"
        "namespace unitree_camera { namespace msg { namespace dds_ {\n"
        "class CameraFrame_ {\n"
        "public:\n"
        "  uint64_t timestamp_ns() const { return timestamp_ns_; }\n"
        "  uint64_t& timestamp_ns() { return timestamp_ns_; }\n"
        "  void timestamp_ns(uint64_t value) { timestamp_ns_ = value; }\n"
        "  const std::string& camera_name() const { return camera_name_; }\n"
        "  std::string& camera_name() { return camera_name_; }\n"
        "  void camera_name(const std::string& value) { camera_name_ = value; }\n"
        "  void camera_name(std::string&& value) { camera_name_ = std::move(value); }\n"
        "  uint32_t width() const { return width_; }\n"
        "  uint32_t& width() { return width_; }\n"
        "  void width(uint32_t value) { width_ = value; }\n"
        "  uint32_t height() const { return height_; }\n"
        "  uint32_t& height() { return height_; }\n"
        "  void height(uint32_t value) { height_ = value; }\n"
        "  const std::string& encoding() const { return encoding_; }\n"
        "  std::string& encoding() { return encoding_; }\n"
        "  void encoding(const std::string& value) { encoding_ = value; }\n"
        "  void encoding(std::string&& value) { encoding_ = std::move(value); }\n"
        "  uint32_t step() const { return step_; }\n"
        "  uint32_t& step() { return step_; }\n"
        "  void step(uint32_t value) { step_ = value; }\n"
        "  const std::vector<uint8_t>& data() const { return data_; }\n"
        "  std::vector<uint8_t>& data() { return data_; }\n"
        "  void data(const std::vector<uint8_t>& value) { data_ = value; }\n"
        "  void data(std::vector<uint8_t>&& value) { data_ = std::move(value); }\n"
        "private:\n"
        "  uint64_t timestamp_ns_ = 0;\n"
        "  std::string camera_name_;\n"
        "  uint32_t width_ = 0;\n"
        "  uint32_t height_ = 0;\n"
        "  std::string encoding_;\n"
        "  uint32_t step_ = 0;\n"
        "  std::vector<uint8_t> data_;\n"
        "};\n"
        "}}}\n",
        encoding="utf-8",
    )
    source.write_text(
        '#include "unitree_camera/msg/dds/CameraFrame_.hpp"\n',
        encoding="utf-8",
    )
    return root


def _make_minimal_generated_head_gaze_dir(tmp_path: Path) -> Path:
    root = tmp_path / "generated-head-gaze"
    root.mkdir()
    (root / "head_state_v1.hpp").write_text(
        "#pragma once\n"
        "namespace visual_events { namespace msg { namespace dds_ {\n"
        "class HeadStateV1_ {};\n"
        "}}}\n",
        encoding="utf-8",
    )
    (root / "head_state_v1.cpp").write_text(
        '#include "head_state_v1.hpp"\n',
        encoding="utf-8",
    )
    (root / "gaze_target_v1.hpp").write_text(
        "#pragma once\n"
        "namespace visual_events { namespace msg { namespace dds_ {\n"
        "class GazeTargetV1_ {};\n"
        "}}}\n",
        encoding="utf-8",
    )
    (root / "gaze_target_v1.cpp").write_text(
        '#include "gaze_target_v1.hpp"\n',
        encoding="utf-8",
    )
    return root


def _make_fake_idlc(
    tmp_path: Path,
    *,
    version: str,
    backends: str,
    codegen: str = "success",
) -> Path:
    script = tmp_path / f"fake-idlc-{version}-{backends.replace(' ', '-')}-{codegen}"
    if codegen not in {"success", "missing_cxx_rc0", "hpp_only"}:
        raise ValueError(codegen)
    script.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = '--version' ] || [ \"$1\" = '-v' ]; then\n"
        f"    printf '%s\\n' 'CycloneDDS idlc {version}'\n"
        "    exit 0\n"
        "fi\n"
        "if [ \"$1\" = '--help' ] || { [ \"$1\" = '-l' ] && [ \"$#\" -eq 1 ]; }; then\n"
        f"    printf '%s\\n' 'available backends: {backends}'\n"
        "    exit 0\n"
        "fi\n"
        "lang=''\n"
        "out_dir=''\n"
        "idl=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "    case \"$1\" in\n"
        "      -l)\n"
        "        shift\n"
        "        lang=\"$1\"\n"
        "        ;;\n"
        "      -o)\n"
        "        shift\n"
        "        out_dir=\"$1\"\n"
        "        ;;\n"
        "      *.idl)\n"
        "        idl=\"$1\"\n"
        "        ;;\n"
        "    esac\n"
        "    shift\n"
        "done\n"
        "if [ \"$lang\" != 'cxx' ] || [ -z \"$out_dir\" ] || [ -z \"$idl\" ]; then\n"
        "    printf '%s\\n' 'fake idlc only supports --version, --help, -l, and -l cxx -o OUT IDL' >&2\n"
        "    exit 64\n"
        "fi\n"
        "base=${idl##*/}\n"
        "base=${base%.idl}\n"
        f"case '{codegen}' in\n"
        "  success)\n"
        "    printf '%s\\n' '// fake generated header' > \"$out_dir/$base.hpp\"\n"
        "    printf '%s\\n' '// fake generated source' > \"$out_dir/$base.cpp\"\n"
        "    ;;\n"
        "  hpp_only)\n"
        "    printf '%s\\n' '// fake generated header' > \"$out_dir/$base.hpp\"\n"
        "    ;;\n"
        "  missing_cxx_rc0)\n"
        "    printf '%s\\n' 'Cannot load generator libcycloneddsidlcxx.so' >&2\n"
        "    printf '%s\\n' 'idlc: cannot load generator cxx' >&2\n"
        "    ;;\n"
        "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _make_fake_idlc_hpp_only_for_stem(
    tmp_path: Path,
    *,
    version: str,
    backends: str,
    hpp_only_stem: str,
) -> Path:
    script = tmp_path / f"fake-idlc-{version}-{backends.replace(' ', '-')}-hpp-only-{hpp_only_stem}"
    script.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = '--version' ] || [ \"$1\" = '-v' ]; then\n"
        f"    printf '%s\\n' 'CycloneDDS idlc {version}'\n"
        "    exit 0\n"
        "fi\n"
        "if [ \"$1\" = '--help' ] || { [ \"$1\" = '-l' ] && [ \"$#\" -eq 1 ]; }; then\n"
        f"    printf '%s\\n' 'available backends: {backends}'\n"
        "    exit 0\n"
        "fi\n"
        "lang=''\n"
        "out_dir=''\n"
        "idl=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "    case \"$1\" in\n"
        "      -l) shift; lang=\"$1\" ;;\n"
        "      -o) shift; out_dir=\"$1\" ;;\n"
        "      *.idl) idl=\"$1\" ;;\n"
        "    esac\n"
        "    shift\n"
        "done\n"
        "if [ \"$lang\" != 'cxx' ] || [ -z \"$out_dir\" ] || [ -z \"$idl\" ]; then\n"
        "    printf '%s\\n' 'fake idlc only supports --version, --help, -l, and -l cxx -o OUT IDL' >&2\n"
        "    exit 64\n"
        "fi\n"
        "base=${idl##*/}\n"
        "base=${base%.idl}\n"
        "printf '%s\\n' '// fake generated header' > \"$out_dir/$base.hpp\"\n"
        f"if [ \"$base\" != '{hpp_only_stem}' ]; then\n"
        "    printf '%s\\n' '// fake generated source' > \"$out_dir/$base.cpp\"\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _make_fake_idlc_v_h_only(tmp_path: Path, *, version: str, backends: str) -> Path:
    script = tmp_path / f"fake-idlc-v-h-only-{version}-{backends.replace(' ', '-')}"
    script.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        "  -v)\n"
        f"    printf '%s\\n' 'CycloneDDS idlc {version}'\n"
        "    ;;\n"
        "  -h)\n"
        f"    printf '%s\\n' 'available backends: {backends}'\n"
        "    ;;\n"
        "  *)\n"
        "    printf '%s\\n' 'fake idlc only supports -v and -h' >&2\n"
        "    exit 64\n"
        "    ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _make_fake_idlc_backend_help_only(tmp_path: Path, *, version: str) -> Path:
    script = tmp_path / f"fake-idlc-backend-help-only-{version}"
    script.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = '--version' ] || [ \"$1\" = '-v' ]; then\n"
        f"    printf '%s\\n' 'CycloneDDS idlc {version}'\n"
        "    exit 0\n"
        "fi\n"
        "if [ \"$1\" = '--help' ] || [ \"$1\" = '-h' ]; then\n"
        "    printf '%s\\n' 'Usage: idlc [OPTIONS] IDL'\n"
        "    exit 0\n"
        "fi\n"
        "if [ \"$1\" = '-l' ] && [ \"$2\" = 'cxx' ] && [ \"$3\" = '-h' ]; then\n"
        "    printf '%s\\n' '--bounded-sequence-template TEMPLATE'\n"
        "    exit 0\n"
        "fi\n"
        "if [ \"$1\" = '-l' ] && [ \"$#\" -eq 1 ]; then\n"
        "    printf '%s\\n' 'available backends: c'\n"
        "    exit 0\n"
        "fi\n"
        "printf '%s\\n' 'fake idlc unsupported args' >&2\n"
        "exit 64\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _make_fake_idlc_codegen_only_from_output_cwd(tmp_path: Path, *, version: str) -> Path:
    script = tmp_path / f"fake-idlc-codegen-output-cwd-only-{version}"
    script.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = '--version' ] || [ \"$1\" = '-v' ]; then\n"
        f"    printf '%s\\n' 'CycloneDDS idlc {version}'\n"
        "    exit 0\n"
        "fi\n"
        "if [ \"$1\" = '--help' ] || { [ \"$1\" = '-l' ] && [ \"$#\" -eq 1 ]; }; then\n"
        "    printf '%s\\n' 'available backends: c cxx'\n"
        "    exit 0\n"
        "fi\n"
        "lang=''\n"
        "out_dir=''\n"
        "idl=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "    case \"$1\" in\n"
        "      -l) shift; lang=\"$1\" ;;\n"
        "      -o) shift; out_dir=\"$1\" ;;\n"
        "      *.idl) idl=\"$1\" ;;\n"
        "    esac\n"
        "    shift\n"
        "done\n"
        "if [ \"$lang\" != 'cxx' ] || [ -z \"$out_dir\" ] || [ -z \"$idl\" ]; then\n"
        "    printf '%s\\n' 'fake idlc only supports --version, --help, -l, and -l cxx -o OUT IDL' >&2\n"
        "    exit 64\n"
        "fi\n"
        "if [ \"$(pwd -P)\" != \"$(cd \"$out_dir\" && pwd -P)\" ]; then\n"
        "    exit 0\n"
        "fi\n"
        "base=${idl##*/}\n"
        "base=${base%.idl}\n"
        "printf '%s\\n' '// fake generated header' > \"$out_dir/$base.hpp\"\n"
        "printf '%s\\n' '// fake generated source' > \"$out_dir/$base.cpp\"\n"
        "exit 0\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _make_fake_idlc_version_generator_error(tmp_path: Path) -> Path:
    script = tmp_path / "fake-idlc-version-generator-error"
    script.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        "  --version)\n"
        "    printf '%s\\n' 'CycloneDDS idlc 0.10.2'\n"
        "    printf '%s\\n' 'Cannot load generator libcycloneddsidlcxx.so' >&2\n"
        "    ;;\n"
        "  --help|-h|-l)\n"
        "    printf '%s\\n' 'available backends: c cxx'\n"
        "    ;;\n"
        "  *)\n"
        "    printf '%s\\n' 'unsupported fake idlc arg' >&2\n"
        "    exit 64\n"
        "    ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _make_probe_idl(tmp_path: Path, stem: str = "CameraFrame_") -> Path:
    idl = tmp_path / f"{stem}.idl"
    idl.write_text(
        "module visual_events { module msg { module dds_ { struct CameraFrame_ { long frame_id; }; }; }; };\n",
        encoding="utf-8",
    )
    return idl


def _repo_build_probe_dir(tmp_path: Path, name: str) -> Path:
    probe_dir = REPO_ROOT / "build" / "test-dds-codegen" / f"{tmp_path.name}-{name}"
    shutil.rmtree(probe_dir, ignore_errors=True)
    return probe_dir


def _repo_build_toolchain_dir(tmp_path: Path, name: str) -> Path:
    toolchain_dir = REPO_ROOT / "build" / "test-dds-codegen" / f"{tmp_path.name}-{name}"
    shutil.rmtree(toolchain_dir, ignore_errors=True)
    return toolchain_dir


def _repo_idlc_cxx() -> Path | None:
    candidate = REPO_ROOT / "build" / "tools" / "cyclonedds-cxx-idlc-0.10.2" / "bin" / "idlc-cxx"
    if candidate.exists():
        return candidate
    resolved = shutil.which("idlc-cxx")
    return Path(resolved) if resolved else None


def _repo_cyclonedds_cxx_include_dir() -> Path | None:
    candidate = (
        REPO_ROOT
        / "build"
        / "tools"
        / "cyclonedds-cxx-idlc-0.10.2"
        / "install"
        / "include"
        / "ddscxx"
    )
    if (candidate / "dds" / "topic" / "TopicTraits.hpp").exists():
        return candidate
    return None


def _generate_repo_head_gaze_dds(tmp_path: Path) -> Path:
    idlc = _repo_idlc_cxx()
    if idlc is None:
        pytest.skip("idlc-cxx is required for native full-bridge mapping tests")
    generated_dir = _repo_build_probe_dir(tmp_path, "mapping-generated")
    generated_dir.mkdir(parents=True)
    for idl in [HEAD_STATE_IDL, GAZE_TARGET_IDL]:
        result = subprocess.run(
            [os.fspath(idlc), "-l", "cxx", "-o", os.fspath(generated_dir), os.fspath(idl)],
            cwd=generated_dir,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
    return generated_dir


def _write_executable(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def _make_fake_prepare_tool_path(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "fake-prepare-bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "git",
        "#!/bin/sh\n"
        "set -eu\n"
        "log=${FAKE_TOOL_LOG:-}\n"
        "if [ -n \"$log\" ]; then printf '%s\\n' \"git $*\" >> \"$log\"; fi\n"
        f"cyclonedds_hash='{CYCLONEDDS_COMMIT}'\n"
        f"cyclonedds_cxx_hash='{CYCLONEDDS_CXX_COMMIT}'\n"
        "repo='cyclonedds'\n"
        "expected=\"$cyclonedds_hash\"\n"
        "case \"$*\" in\n"
        "  *cyclonedds-cxx*) repo='cyclonedds_cxx'; expected=\"$cyclonedds_cxx_hash\" ;;\n"
        "esac\n"
        "if [ \"$1\" = 'ls-remote' ]; then\n"
        "  if [ \"${FAKE_LS_REMOTE_BAD:-}\" = \"$repo\" ] || [ \"${FAKE_LS_REMOTE_BAD:-}\" = '1' ]; then\n"
        "    expected='0000000000000000000000000000000000000000'\n"
        "  fi\n"
        "  printf '%s\\trefs/tags/0.10.2\\n' \"$expected\"\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = 'clone' ]; then\n"
        "  if [ \"${FAKE_GIT_CLONE_FAIL:-}\" = \"$repo\" ] || [ \"${FAKE_GIT_CLONE_FAIL:-}\" = '1' ]; then\n"
        "    printf '%s\\n' 'fake clone failure' >&2\n"
        "    exit 42\n"
        "  fi\n"
        "  target=''\n"
        "  for arg in \"$@\"; do target=\"$arg\"; done\n"
        "  /bin/mkdir -p \"$target\"\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = '-C' ] && [ \"$3\" = 'rev-parse' ]; then\n"
        "  if [ \"${FAKE_REV_PARSE_BAD:-}\" = \"$repo\" ] || [ \"${FAKE_REV_PARSE_BAD:-}\" = '1' ]; then\n"
        "    printf '%s\\n' '1111111111111111111111111111111111111111'\n"
        "    exit 0\n"
        "  fi\n"
        "  case \"$2\" in\n"
        "    *cyclonedds-cxx*) printf '%s\\n' \"$cyclonedds_cxx_hash\" ;;\n"
        "    *) printf '%s\\n' \"$cyclonedds_hash\" ;;\n"
        "  esac\n"
        "  exit 0\n"
        "fi\n"
        "printf '%s\\n' \"unexpected fake git invocation: $*\" >&2\n"
        "exit 64\n",
    )
    _write_executable(
        bin_dir / "cmake",
        "#!/bin/sh\n"
        "set -eu\n"
        "log=${FAKE_TOOL_LOG:-}\n"
        "if [ -n \"$log\" ]; then printf '%s\\n' \"cmake $*\" >> \"$log\"; fi\n"
        "install=${FAKE_INSTALL_DIR:?}\n"
        "if [ \"$1\" = '--build' ]; then\n"
        "  build_dir=\"$2\"\n"
        "  /bin/mkdir -p \"$install/bin\" \"$install/lib\"\n"
        "  case \"$build_dir\" in\n"
        "    */build/cyclonedds)\n"
        "      /usr/bin/touch \"$install/lib/libcycloneddsidl.so\" \"$install/lib/libddsc.so\"\n"
        "      /bin/cat > \"$install/bin/idlc\" <<'IDLC'\n"
        "#!/bin/sh\n"
        "if [ \"$1\" = '--version' ] || [ \"$1\" = '-v' ]; then\n"
        "  printf '%s\\n' 'CycloneDDS idlc 0.10.2'\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = '--help' ] || { [ \"$1\" = '-l' ] && [ \"$#\" -eq 1 ]; }; then\n"
        "  printf '%s\\n' 'available backends: c cxx'\n"
        "  exit 0\n"
        "fi\n"
        "lang=''\n"
        "out_dir=''\n"
        "idl=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  case \"$1\" in\n"
        "    -l) shift; lang=\"$1\" ;;\n"
        "    -o) shift; out_dir=\"$1\" ;;\n"
        "    *.idl) idl=\"$1\" ;;\n"
        "  esac\n"
        "  shift\n"
        "done\n"
        "if [ \"$lang\" != 'cxx' ] || [ -z \"$out_dir\" ] || [ -z \"$idl\" ]; then\n"
        "  printf '%s\\n' 'fake installed idlc unsupported args' >&2\n"
        "  exit 64\n"
        "fi\n"
        "base=${idl##*/}\n"
        "base=${base%.idl}\n"
        "case \"${FAKE_INSTALLED_IDLC_CODEGEN:-success}\" in\n"
        "  success)\n"
        "    printf '%s\\n' '// fake generated header' > \"$out_dir/$base.hpp\"\n"
        "    printf '%s\\n' '// fake generated source' > \"$out_dir/$base.cpp\"\n"
        "    ;;\n"
        "  hpp_only)\n"
        "    printf '%s\\n' '// fake generated header' > \"$out_dir/$base.hpp\"\n"
        "    ;;\n"
        "  missing_cxx_rc0)\n"
        "    printf '%s\\n' 'idlc: cannot load generator cxx' >&2\n"
        "    ;;\n"
        "esac\n"
        "exit 0\n"
        "IDLC\n"
        "      /bin/chmod +x \"$install/bin/idlc\"\n"
        "      ;;\n"
        "    */build/cyclonedds-cxx)\n"
        "      if [ \"${FAKE_SKIP_IDLCXX:-}\" != '1' ]; then\n"
        "        /usr/bin/touch \"$install/lib/libcycloneddsidlcxx.so\"\n"
        "      fi\n"
        "      ;;\n"
        "  esac\n"
        "  exit 0\n"
        "fi\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '-B' ]; then\n"
        "    shift\n"
        "    /bin/mkdir -p \"$1\"\n"
        "  fi\n"
        "  shift\n"
        "done\n"
        "exit 0\n",
    )
    for name in ["make", "gcc", "g++"]:
        _write_executable(bin_dir / name, "#!/bin/sh\nexit 0\n")
    return bin_dir


@pytest.fixture
def repo_report_path(tmp_path):
    report_dirs: list[Path] = []

    def make(name: str) -> Path:
        report_dir = REPO_ROOT / "artifacts" / "test-dds-bridge" / f"{tmp_path.name}-{name}"
        shutil.rmtree(report_dir, ignore_errors=True)
        report_dir.mkdir(parents=True)
        report_dirs.append(report_dir)
        return report_dir / "report.json"

    yield make

    for report_dir in report_dirs:
        shutil.rmtree(report_dir, ignore_errors=True)


def _run_build_tool(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, os.fspath(TOOLS_BUILD), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_prepare_codegen_tool(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, os.fspath(TOOLS_PREPARE_CODEGEN), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.fixture
def native_abi_build(tmp_path):
    if shutil.which("cmake") is None:
        pytest.skip("cmake is required for native ABI target tests")
    if (
        shutil.which("c++") is None
        and shutil.which("g++") is None
        and shutil.which("clang++") is None
    ):
        pytest.skip("a C++ compiler is required for native ABI target tests")

    build_dir = REPO_ROOT / "build" / "test-dds-bridge" / f"{tmp_path.name}-abi"
    shutil.rmtree(build_dir, ignore_errors=True)
    configure = subprocess.run(
        [
            "cmake",
            "-S",
            os.fspath(NATIVE_BRIDGE),
            "-B",
            os.fspath(build_dir),
            "-DVISUAL_EVENTS_DDS_BRIDGE_BUILD_PROBE=OFF",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert configure.returncode == 0, configure.stderr

    build = subprocess.run(
        [
            "cmake",
            "--build",
            os.fspath(build_dir),
            "--target",
            "visual_events_dds_bridge",
            "visual_events_dds_bridge_abi_harness",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr

    yield build_dir

    shutil.rmtree(build_dir, ignore_errors=True)


@pytest.fixture
def native_full_bridge_mapping_build(tmp_path):
    if shutil.which("cmake") is None:
        pytest.skip("cmake is required for native full-bridge mapping target tests")
    if (
        shutil.which("c++") is None
        and shutil.which("g++") is None
        and shutil.which("clang++") is None
    ):
        pytest.skip("a C++ compiler is required for native full-bridge mapping target tests")
    dds_include_dir = _repo_cyclonedds_cxx_include_dir()
    if dds_include_dir is None:
        pytest.skip("CycloneDDS C++ headers are required for generated Head/Gaze headers")

    unitree_root = _make_minimal_unitree_sdk_root(tmp_path)
    video_dir = _make_minimal_video_dds_publisher_dir(tmp_path)
    generated_dir = _generate_repo_head_gaze_dds(tmp_path)
    build_dir = REPO_ROOT / "build" / "test-dds-bridge" / f"{tmp_path.name}-mapping"
    shutil.rmtree(build_dir, ignore_errors=True)
    configure = subprocess.run(
        [
            "cmake",
            "-S",
            os.fspath(NATIVE_BRIDGE),
            "-B",
            os.fspath(build_dir),
            f"-DUNITREE_SDK_ROOT={unitree_root}",
            f"-DVIDEO_DDS_PUBLISHER_DIR={video_dir}",
            "-DVISUAL_EVENTS_DDS_BRIDGE_FULL_BRIDGE=ON",
            f"-DVISUAL_EVENTS_GENERATED_DDS_DIR={generated_dir}",
            f"-DVISUAL_EVENTS_DDS_BRIDGE_CYCLONEDDS_INCLUDE_DIR={dds_include_dir}",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert configure.returncode == 0, configure.stderr

    build = subprocess.run(
        [
            "cmake",
            "--build",
            os.fspath(build_dir),
            "--target",
            "visual_events_dds_bridge_mapping_harness",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr

    yield build_dir

    shutil.rmtree(build_dir, ignore_errors=True)
    shutil.rmtree(generated_dir, ignore_errors=True)


@pytest.fixture
def native_full_bridge_construction_build(tmp_path):
    if shutil.which("cmake") is None:
        pytest.skip("cmake is required for native construction harness target tests")
    if (
        shutil.which("c++") is None
        and shutil.which("g++") is None
        and shutil.which("clang++") is None
    ):
        pytest.skip("a C++ compiler is required for native construction harness target tests")

    unitree_root = _make_minimal_unitree_sdk_root(tmp_path)
    video_dir = _make_minimal_video_dds_publisher_dir(tmp_path)
    generated_dir = _make_minimal_generated_head_gaze_dir(tmp_path)
    build_dir = REPO_ROOT / "build" / "test-dds-bridge" / f"{tmp_path.name}-construction"
    shutil.rmtree(build_dir, ignore_errors=True)
    configure = subprocess.run(
        [
            "cmake",
            "-S",
            os.fspath(NATIVE_BRIDGE),
            "-B",
            os.fspath(build_dir),
            f"-DUNITREE_SDK_ROOT={unitree_root}",
            f"-DVIDEO_DDS_PUBLISHER_DIR={video_dir}",
            "-DVISUAL_EVENTS_DDS_BRIDGE_FULL_BRIDGE=ON",
            f"-DVISUAL_EVENTS_GENERATED_DDS_DIR={generated_dir}",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert configure.returncode == 0, configure.stderr

    build = subprocess.run(
        [
            "cmake",
            "--build",
            os.fspath(build_dir),
            "--target",
            "visual_events_dds_bridge_construction_harness",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr

    yield build_dir

    shutil.rmtree(build_dir, ignore_errors=True)


def test_native_bridge_source_allowlist_has_only_camera_head_gaze_and_no_motion_tokens():
    text = _combined_native_source_text()

    for topic in ALLOWED_TOPICS:
        assert topic in text
    for type_name in ALLOWED_TYPES:
        assert type_name in text

    topic_literals = set()
    for part in text.split('"')[1::2]:
        if part.startswith("/"):
            topic_literals.add(part)
    assert topic_literals <= ALLOWED_TOPICS

    offenders = [token for token in sorted(DENIED_MOTION_TOKENS) if token in text]
    assert offenders == []


def test_native_bridge_cmake_uses_unitree_and_camera_frame_inputs_without_python_or_motion_sdks():
    cmake = NATIVE_BRIDGE / "CMakeLists.txt"
    assert cmake.exists()
    text = cmake.read_text(encoding="utf-8")

    required = [
        "find_package(unitree_sdk2 REQUIRED)",
        "unitree_sdk2",
        "VIDEO_DDS_PUBLISHER_DIR",
        "CameraFrame_.cpp",
        "unitree_camera/msg/dds/CameraFrame_.hpp",
        "visual_events_dds_bridge_probe",
    ]
    missing = [item for item in required if item not in text]
    assert missing == []

    forbidden = ["python_dds", "cyclonedds-python", "fastdds", "SportMode", "LowCmd", "MotorCmd"]
    offenders = [item for item in forbidden if item in text]
    assert offenders == []


def test_native_bridge_cmake_has_optional_full_bridge_generated_type_support_inputs():
    cmake = NATIVE_BRIDGE / "CMakeLists.txt"
    text = cmake.read_text(encoding="utf-8")

    required = [
        "VISUAL_EVENTS_DDS_BRIDGE_FULL_BRIDGE",
        "VISUAL_EVENTS_GENERATED_DDS_DIR",
        "VISUAL_EVENTS_HEAD_STATE_HEADER",
        "VISUAL_EVENTS_HEAD_STATE_SOURCE",
        "VISUAL_EVENTS_GAZE_TARGET_HEADER",
        "VISUAL_EVENTS_GAZE_TARGET_SOURCE",
        "head_state_v1.hpp",
        "head_state_v1.cpp",
        "gaze_target_v1.hpp",
        "gaze_target_v1.cpp",
    ]
    missing = [item for item in required if item not in text]
    assert missing == []


def test_native_bridge_cmake_declares_runtime_abi_core_and_test_harness_targets():
    text = (NATIVE_BRIDGE / "CMakeLists.txt").read_text(encoding="utf-8")

    required = [
        "VISUAL_EVENTS_DDS_BRIDGE_BUILD_PROBE",
        "visual_events_dds_bridge_abi",
        "src/bridge_abi.cpp",
        "visual_events_dds_bridge",
        "src/runtime_main.cpp",
        "visual_events_dds_bridge_abi_harness",
        "src/abi_harness_main.cpp",
    ]
    missing = [item for item in required if item not in text]
    assert missing == []


def test_native_bridge_declares_dds_type_mapping_core_and_full_bridge_harness():
    text = _combined_native_source_text()

    required = [
        "bridge_dds_types.hpp",
        "src/bridge_dds_types.cpp",
        "visual_events_dds_bridge_dds_types",
        "visual_events_dds_bridge_mapping_harness",
        "src/mapping_harness_main.cpp",
        "CameraFrameToAbi",
        "HeadStateToAbi",
        "GazeTargetFrameToDds",
        "VISUAL_EVENTS_DDS_BRIDGE_FULL_BRIDGE",
    ]
    missing = [item for item in required if item not in text]
    assert missing == []


def test_native_bridge_declares_unitree_channel_construction_harness_without_direct_cyclonedds_pubsub():
    text = _combined_native_source_text()

    required = [
        "visual_events_dds_bridge_construction_harness",
        "src/construction_harness_main.cpp",
        "unitree_channel_runtime.hpp",
        "src/unitree_channel_runtime.cpp",
        "ChannelFactory::Instance()->Init",
        "unitree::robot::ChannelSubscriber<unitree_camera::msg::dds_::CameraFrame_>",
        "unitree::robot::ChannelSubscriber<visual_events::msg::dds_::HeadStateV1_>",
        "unitree::robot::ChannelPublisher<visual_events::msg::dds_::GazeTargetV1_>",
    ]
    missing = [item for item in required if item not in text]
    assert missing == []

    forbidden = [
        "dds::pub::Publisher",
        "dds::sub::Subscriber",
        "dds::topic::Topic",
        "dds_create_participant",
        "dds_create_writer",
        "dds_create_reader",
        "ddsc/",
    ]
    offenders = [item for item in forbidden if item in text]
    assert offenders == []


def test_native_bridge_full_bridge_construction_target_includes_camera_head_gaze_type_support():
    text = (NATIVE_BRIDGE / "CMakeLists.txt").read_text(encoding="utf-8")

    required = [
        "visual_events_dds_bridge_construction_harness",
        "src/construction_harness_main.cpp",
        "src/unitree_channel_runtime.cpp",
        "${CAMERA_FRAME_SOURCE}",
        "${VISUAL_EVENTS_GENERATED_DDS_SOURCES}",
        "VISUAL_EVENTS_DDS_BRIDGE_FULL_BRIDGE=1",
    ]
    missing = [item for item in required if item not in text]
    assert missing == []


def test_native_runtime_options_are_pure_env_parser_without_unitree_dependency():
    header = NATIVE_BRIDGE / "include" / "visual_events" / "dds_bridge" / "runtime_options.hpp"
    source = NATIVE_BRIDGE / "src" / "runtime_options.cpp"
    assert header.exists()
    assert source.exists()
    text = header.read_text(encoding="utf-8") + source.read_text(encoding="utf-8")

    required = [
        "VISUAL_EVENTS_DDS_DOMAIN",
        "VISUAL_EVENTS_DDS_NETWORK",
        "VISUAL_EVENTS_CAMERA_TOPIC",
        "VISUAL_EVENTS_HEAD_STATE_TOPIC",
        "VISUAL_EVENTS_GAZE_TOPIC",
        "kCameraTopic.name",
        "kHeadTopic.name",
        "kGazeTopic.name",
    ]
    missing = [item for item in required if item not in text]
    assert missing == []
    assert "unitree/" not in text


def test_native_construction_runtime_closes_channels_before_releasing_factory():
    source = NATIVE_BRIDGE / "src" / "unitree_channel_runtime.cpp"
    assert source.exists()
    text = source.read_text(encoding="utf-8")

    assert "CloseChannel" in text
    assert "ChannelFactory::Instance()->Release" in text
    assert text.rfind("CloseChannel") < text.rfind("ChannelFactory::Instance()->Release")
    assert "Write(" not in text
    assert "std::cout" not in text


def test_native_dds_type_mapping_foundation_target_builds_without_generated_head_gaze(tmp_path):
    if shutil.which("cmake") is None:
        pytest.skip("cmake is required for native DDS mapping target tests")
    if (
        shutil.which("c++") is None
        and shutil.which("g++") is None
        and shutil.which("clang++") is None
    ):
        pytest.skip("a C++ compiler is required for native DDS mapping target tests")

    unitree_root = _make_minimal_unitree_sdk_root(tmp_path)
    video_dir = _make_minimal_video_dds_publisher_dir(tmp_path)
    build_dir = REPO_ROOT / "build" / "test-dds-bridge" / f"{tmp_path.name}-mapping-foundation"
    shutil.rmtree(build_dir, ignore_errors=True)
    configure = subprocess.run(
        [
            "cmake",
            "-S",
            os.fspath(NATIVE_BRIDGE),
            "-B",
            os.fspath(build_dir),
            f"-DUNITREE_SDK_ROOT={unitree_root}",
            f"-DVIDEO_DDS_PUBLISHER_DIR={video_dir}",
            "-DVISUAL_EVENTS_DDS_BRIDGE_FULL_BRIDGE=OFF",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert configure.returncode == 0, configure.stderr

    build = subprocess.run(
        [
            "cmake",
            "--build",
            os.fspath(build_dir),
            "--target",
            "visual_events_dds_bridge_dds_types",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        assert build.returncode == 0, build.stderr
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)


def test_native_probe_source_references_generated_head_and_gaze_type_props():
    probe = NATIVE_BRIDGE / "src" / "probe_main.cpp"
    text = probe.read_text(encoding="utf-8")

    required = [
        '#include "head_state_v1.hpp"',
        '#include "gaze_target_v1.hpp"',
        "get_type_props<visual_events::msg::dds_::HeadStateV1_>()",
        "get_type_props<visual_events::msg::dds_::GazeTargetV1_>()",
        "HeadStateV1_ type properties are unavailable",
        "GazeTargetV1_ type properties are unavailable",
    ]
    missing = [item for item in required if item not in text]
    assert missing == []


def test_native_probe_status_source_contract_uses_existing_jsonl_bridge_status_frame():
    text = _combined_native_source_text()
    assert '"protocol_version":1' in text
    assert '"type":"status"' in text
    assert '"code":"probe_ok"' in text
    assert '"message":"' in text
    assert '"mode":"probe"' in text
    assert '"status":"ok"' not in text
    assert "std::cout" in text
    assert "std::cerr" in text


def test_native_probe_binary_emits_single_jsonl_status_frame_without_stdout_logs():
    binary = REPO_ROOT / "build" / "dds_bridge" / "visual_events_dds_bridge_probe"
    if not binary.exists():
        pytest.skip("probe binary not built")

    result = subprocess.run(
        [os.fspath(binary), "--probe"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    stdout_lines = result.stdout.splitlines()
    assert len(stdout_lines) == 1
    status = json.loads(stdout_lines[0])
    assert status["protocol_version"] == 1
    assert status["type"] == "status"
    assert status["code"] == "probe_ok"
    assert isinstance(status["message"], str)
    assert status["message"]
    assert status["mode"] == "probe"
    assert "log" not in status


def test_native_runtime_probe_and_no_args_fail_fast_as_jsonl(native_abi_build):
    from visual_events_cli.dds.bridge_protocol import BridgeErrorFrame, BridgeStatusFrame
    from visual_events_cli.dds.bridge_protocol import decode_bridge_line

    binary = native_abi_build / "visual_events_dds_bridge"

    probe = subprocess.run(
        [os.fspath(binary), "--probe"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert probe.returncode == 0
    assert len(probe.stdout.splitlines()) == 1
    status = decode_bridge_line(probe.stdout, logical_camera_name="front")
    assert isinstance(status, BridgeStatusFrame)
    assert status.code == "probe_ok"
    assert "log" not in json.loads(probe.stdout)

    runtime = subprocess.run(
        [os.fspath(binary)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert runtime.returncode != 0
    stdout_lines = runtime.stdout.splitlines()
    assert len(stdout_lines) == 1
    error = decode_bridge_line(stdout_lines[0], logical_camera_name="front")
    assert isinstance(error, BridgeErrorFrame)
    assert error.fatal is True
    assert error.code == "dds_runtime_not_implemented"
    assert runtime.stderr


def test_native_construction_harness_print_options_defaults_and_env_contract(
    native_full_bridge_construction_build,
):
    binary = native_full_bridge_construction_build / "visual_events_dds_bridge_construction_harness"

    default_env = os.environ.copy()
    for name in [
        "VISUAL_EVENTS_DDS_DOMAIN",
        "VISUAL_EVENTS_DDS_NETWORK",
        "VISUAL_EVENTS_CAMERA_TOPIC",
        "VISUAL_EVENTS_HEAD_STATE_TOPIC",
        "VISUAL_EVENTS_GAZE_TOPIC",
    ]:
        default_env.pop(name, None)

    defaults = subprocess.run(
        [os.fspath(binary), "--print-options"],
        cwd=REPO_ROOT,
        env=default_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert defaults.returncode == 0
    assert len(defaults.stdout.splitlines()) == 1
    default_status = json.loads(defaults.stdout)
    assert default_status == {
        "protocol_version": 1,
        "type": "status",
        "code": "options_ok",
        "message": "native Unitree channel construction options ok",
        "mode": "print_options",
        "domain": 0,
        "network": "eth0",
        "camera_topic": "/camera/image/jpeg",
        "head_state_topic": "/robot/head_state",
        "gaze_topic": "/visual_events/gaze_target",
    }

    env = default_env | {
        "VISUAL_EVENTS_DDS_DOMAIN": "7",
        "VISUAL_EVENTS_DDS_NETWORK": "enp4s0",
        "VISUAL_EVENTS_CAMERA_TOPIC": "/camera/image/jpeg",
        "VISUAL_EVENTS_HEAD_STATE_TOPIC": "/robot/head_state",
        "VISUAL_EVENTS_GAZE_TOPIC": "/visual_events/gaze_target",
    }
    overridden = subprocess.run(
        [os.fspath(binary), "--print-options"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert overridden.returncode == 0
    assert len(overridden.stdout.splitlines()) == 1
    status = json.loads(overridden.stdout)
    assert status["domain"] == 7
    assert status["network"] == "enp4s0"
    assert status["camera_topic"] == "/camera/image/jpeg"
    assert status["head_state_topic"] == "/robot/head_state"
    assert status["gaze_topic"] == "/visual_events/gaze_target"


@pytest.mark.parametrize(
    ("env_name", "env_value", "expected_fragment"),
    [
        ("VISUAL_EVENTS_DDS_DOMAIN", "not-an-int", "VISUAL_EVENTS_DDS_DOMAIN"),
        ("VISUAL_EVENTS_DDS_DOMAIN", "-1", "VISUAL_EVENTS_DDS_DOMAIN"),
        ("VISUAL_EVENTS_DDS_NETWORK", "", "VISUAL_EVENTS_DDS_NETWORK"),
        ("VISUAL_EVENTS_CAMERA_TOPIC", "", "VISUAL_EVENTS_CAMERA_TOPIC"),
        ("VISUAL_EVENTS_HEAD_STATE_TOPIC", "", "VISUAL_EVENTS_HEAD_STATE_TOPIC"),
        ("VISUAL_EVENTS_GAZE_TOPIC", "", "VISUAL_EVENTS_GAZE_TOPIC"),
    ],
)
def test_native_construction_harness_print_options_invalid_env_is_single_fatal_jsonl(
    native_full_bridge_construction_build,
    env_name: str,
    env_value: str,
    expected_fragment: str,
):
    binary = native_full_bridge_construction_build / "visual_events_dds_bridge_construction_harness"
    env = os.environ.copy() | {env_name: env_value}
    result = subprocess.run(
        [os.fspath(binary), "--print-options"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    stdout_lines = result.stdout.splitlines()
    assert len(stdout_lines) == 1
    error = json.loads(stdout_lines[0])
    assert error["protocol_version"] == 1
    assert error["type"] == "error"
    assert error["code"] == "invalid_runtime_options"
    assert error["fatal"] is True
    assert expected_fragment in error["message"]
    assert "log" not in error
    assert result.stderr


@pytest.mark.parametrize("gaze_state", ["tracking", "lost", "stale", "disabled"])
def test_native_abi_harness_outputs_camera_head_and_accepts_python_gaze(
    native_abi_build,
    gaze_state: str,
):
    from visual_events_cli.dds.bridge_protocol import BridgeHeadStateFrame
    from visual_events_cli.dds.bridge_protocol import decode_bridge_line, encode_gaze_target_line
    from visual_events_cli.dds.types import CameraJpegMessage
    from visual_events_cli.target_mapper import GazeTargetPayload

    payload = GazeTargetPayload(
        schema_version=1,
        camera="front",
        frame_id=42,
        frame_timestamp_ms=1_710_000_000_000,
        publish_timestamp_ms=1_710_000_000_082,
        valid=gaze_state == "tracking",
        state=gaze_state,
        target_track_id=7,
        target_u=640.0,
        target_v=360.0,
        target_norm_x=0.0,
        target_norm_y=0.0,
        image_width=1280,
        image_height=720,
        confidence=0.91,
        reason="nearest",
        stale_after_ms=250,
    )

    result = subprocess.run(
        [os.fspath(native_abi_build / "visual_events_dds_bridge_abi_harness")],
        cwd=REPO_ROOT,
        input=encode_gaze_target_line(payload),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "accepted_gaze_target=1" in result.stderr
    stdout_lines = result.stdout.splitlines()
    assert len(stdout_lines) == 2
    raw_frames = [json.loads(line) for line in stdout_lines]
    assert [frame["type"] for frame in raw_frames] == ["camera_jpeg", "head_state"]
    assert "data" not in raw_frames[0]
    assert raw_frames[0]["data_size_bytes"] > 0
    assert raw_frames[0]["data_base64"]

    camera = decode_bridge_line(stdout_lines[0], logical_camera_name="front")
    head = decode_bridge_line(stdout_lines[1], logical_camera_name="front")
    assert isinstance(camera, CameraJpegMessage)
    assert camera.camera == "front"
    assert camera.data
    assert isinstance(head, BridgeHeadStateFrame)
    assert head.state in {"stationary", "moving", "unknown"}


@pytest.mark.parametrize(
    "malformed_gaze",
    [
        json.dumps(
            {
                "protocol_version": 1,
                "type": "gaze_target",
                "schema_version": 1,
                "camera": "front",
            },
            separators=(",", ":"),
        ),
        json.dumps(
            {
                "protocol_version": 1,
                "type": "gaze_target",
                "schema_version": 1,
                "camera": "front",
                "frame_id": "42",
                "frame_timestamp_ms": 1_710_000_000_000,
                "publish_timestamp_ms": 1_710_000_000_082,
                "valid": True,
                "state": "tracking",
                "target_track_id": 7,
                "target_u": 640.0,
                "target_v": 360.0,
                "target_norm_x": 0.0,
                "target_norm_y": 0.0,
                "image_width": 1280,
                "image_height": 720,
                "confidence": 0.91,
                "reason": "nearest",
                "stale_after_ms": 250,
            },
            separators=(",", ":"),
        ),
        (
            '{"protocol_version":1,"type":"gaze_target","schema_version":1,'
            '"camera":"front","frame_id":42,"frame_timestamp_ms":1710000000000,'
            '"publish_timestamp_ms":1710000000082,"valid":true,"state":"tracking",'
            '"target_track_id":7,"target_u":1e999,"target_v":360.0,'
            '"target_norm_x":0.0,"target_norm_y":0.0,"image_width":1280,'
            '"image_height":720,"confidence":0.91,"reason":"nearest",'
            '"stale_after_ms":250}'
        ),
        (
            '{"protocol_version":1,"type":"gaze_target","schema_version":1,'
            '"camera":"front","frame_id":42,"frame_timestamp_ms":1710000000000,'
            '"publish_timestamp_ms":1710000000082,"valid":true,"state":"tracking",'
            '"target_track_id":7,"target_u":NaN,"target_v":360.0,'
            '"target_norm_x":0.0,"target_norm_y":0.0,"image_width":1280,'
            '"image_height":720,"confidence":0.91,"reason":"nearest",'
            '"stale_after_ms":250}'
        ),
        (
            '{"protocol_version":1,"type":"gaze_target","schema_version":1,'
            '"camera":"front","frame_id":42,"frame_timestamp_ms":1710000000000,'
            '"publish_timestamp_ms":1710000000082,"valid":true,"state":"bogus",'
            '"target_track_id":7,"target_u":640.0,"target_v":360.0,'
            '"target_norm_x":0.0,"target_norm_y":0.0,"image_width":1280,'
            '"image_height":720,"confidence":0.91,"reason":"nearest",'
            '"stale_after_ms":250}'
        ),
    ],
)
def test_native_abi_harness_rejects_invalid_gaze_with_fatal_jsonl(
    native_abi_build,
    malformed_gaze: str,
):
    from visual_events_cli.dds.bridge_protocol import BridgeErrorFrame
    from visual_events_cli.dds.bridge_protocol import decode_bridge_line

    result = subprocess.run(
        [os.fspath(native_abi_build / "visual_events_dds_bridge_abi_harness")],
        cwd=REPO_ROOT,
        input=malformed_gaze + "\n",
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    stdout_lines = result.stdout.splitlines()
    assert len(stdout_lines) == 3
    decoded_error = decode_bridge_line(stdout_lines[-1], logical_camera_name="front")
    assert isinstance(decoded_error, BridgeErrorFrame)
    assert decoded_error.fatal is True
    assert decoded_error.code == "invalid_gaze_target"
    for line in stdout_lines:
        frame = json.loads(line)
        assert "log" not in frame
        assert "data" not in frame


def _canonical_gaze_target_jsonl(state: str = "tracking", valid: bool | None = None) -> str:
    from visual_events_cli.dds.bridge_protocol import encode_gaze_target_line
    from visual_events_cli.target_mapper import GazeTargetPayload

    if valid is None:
        valid = state == "tracking"

    payload = GazeTargetPayload(
        schema_version=1,
        camera="front",
        frame_id=42,
        frame_timestamp_ms=1_710_000_000_000,
        publish_timestamp_ms=1_710_000_000_082,
        valid=valid,
        state=state,
        target_track_id=7,
        target_u=640.0,
        target_v=360.0,
        target_norm_x=0.25,
        target_norm_y=-0.5,
        image_width=1280,
        image_height=720,
        confidence=0.91,
        reason="nearest",
        stale_after_ms=250,
    )
    return encode_gaze_target_line(payload)


def test_native_mapping_harness_maps_camera_and_head_dds_to_jsonl_abi(
    native_full_bridge_mapping_build,
):
    from visual_events_cli.dds.bridge_protocol import BridgeHeadStateFrame
    from visual_events_cli.dds.bridge_protocol import decode_bridge_line
    from visual_events_cli.dds.types import CameraJpegMessage

    result = subprocess.run(
        [os.fspath(native_full_bridge_mapping_build / "visual_events_dds_bridge_mapping_harness")],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout_lines = result.stdout.splitlines()
    assert len(stdout_lines) == 4
    raw_frames = [json.loads(line) for line in stdout_lines]
    assert [frame["type"] for frame in raw_frames] == [
        "camera_jpeg",
        "head_state",
        "head_state",
        "head_state",
    ]
    assert "data" not in raw_frames[0]
    assert raw_frames[0]["dds_timestamp_ns"] == 123456789
    assert raw_frames[0]["received_monotonic_ns"] == 987654321
    assert raw_frames[0]["camera_name"] == "front"
    assert raw_frames[0]["width"] == 1280
    assert raw_frames[0]["height"] == 720
    assert raw_frames[0]["encoding"] == "jpeg"
    assert raw_frames[0]["step"] == 4096
    assert raw_frames[0]["data_size_bytes"] == 4

    camera = decode_bridge_line(stdout_lines[0], logical_camera_name="front")
    assert isinstance(camera, CameraJpegMessage)
    assert camera.camera == "front"
    assert camera.data == b"\xff\xd8\xff\xd9"

    heads = [decode_bridge_line(line, logical_camera_name="front") for line in stdout_lines[1:]]
    assert all(isinstance(head, BridgeHeadStateFrame) for head in heads)
    assert [head.state for head in heads] == ["stationary", "moving", "unknown"]
    assert [head.valid for head in heads] == [True, True, False]
    assert raw_frames[3]["yaw_rad"] == 0.0
    assert raw_frames[3]["pitch_vel_rad_s"] == 0.0


@pytest.mark.parametrize(
    ("gaze_state", "valid"),
    [
        ("tracking", True),
        ("lost", False),
        ("stale", False),
        ("disabled", False),
    ],
)
def test_native_mapping_harness_constructs_generated_gaze_target_from_python_jsonl(
    native_full_bridge_mapping_build,
    gaze_state: str,
    valid: bool,
):
    line = _canonical_gaze_target_jsonl(gaze_state, valid=valid)
    result = subprocess.run(
        [os.fspath(native_full_bridge_mapping_build / "visual_events_dds_bridge_mapping_harness")],
        cwd=REPO_ROOT,
        input=line,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    constructed = json.loads(result.stdout.splitlines()[-1])
    assert constructed == {
        "protocol_version": 1,
        "type": "gaze_target_constructed",
        "schema_version": 1,
        "camera": "front",
        "frame_id": 42,
        "frame_timestamp_ms": 1_710_000_000_000,
        "publish_timestamp_ms": 1_710_000_000_082,
        "valid": valid,
        "state": gaze_state,
        "target_track_id": 7,
        "target_u": 640.0,
        "target_v": 360.0,
        "target_norm_x": 0.25,
        "target_norm_y": -0.5,
        "image_width": 1280,
        "image_height": 720,
        "confidence": pytest.approx(0.91),
        "reason": "nearest",
        "stale_after_ms": 250,
    }


@pytest.mark.parametrize(
    ("gaze_state", "valid"),
    [
        ("tracking", False),
        ("lost", True),
        ("stale", True),
        ("disabled", True),
    ],
)
def test_native_mapping_harness_rejects_gaze_target_valid_state_mismatch(
    native_full_bridge_mapping_build,
    gaze_state: str,
    valid: bool,
):
    line = _canonical_gaze_target_jsonl(gaze_state, valid=valid)

    result = subprocess.run(
        [os.fspath(native_full_bridge_mapping_build / "visual_events_dds_bridge_mapping_harness")],
        cwd=REPO_ROOT,
        input=line,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    error = json.loads(result.stdout.splitlines()[-1])
    assert error["type"] == "error"
    assert error["code"] == "invalid_gaze_target"
    assert error["fatal"] is True


@pytest.mark.parametrize(
    ("field", "value", "error_fragment"),
    [
        ("image_width", 2**32, "image_width"),
        ("target_u", 1e40, "target_u"),
        ("state", "bogus", "state"),
        ("schema_version", 2, "schema_version"),
    ],
)
def test_native_mapping_harness_rejects_gaze_target_construction_invalid_inputs(
    native_full_bridge_mapping_build,
    field: str,
    value: object,
    error_fragment: str,
):
    payload = json.loads(_canonical_gaze_target_jsonl())
    payload[field] = value
    line = json.dumps(payload, separators=(",", ":")) + "\n"

    result = subprocess.run(
        [os.fspath(native_full_bridge_mapping_build / "visual_events_dds_bridge_mapping_harness")],
        cwd=REPO_ROOT,
        input=line,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    error = json.loads(result.stdout.splitlines()[-1])
    assert error["type"] == "error"
    assert error["code"] == "invalid_gaze_target"
    assert error["fatal"] is True
    assert error_fragment in error["message"]


def test_build_tool_foundation_check_does_not_require_idl_generator(tmp_path, repo_report_path):
    assert TOOLS_BUILD.exists()
    unitree_root = _make_minimal_unitree_sdk_root(tmp_path)
    video_dir = _make_minimal_video_dds_publisher_dir(tmp_path)
    report_path = repo_report_path("foundation")

    env = os.environ.copy()
    env["PATH"] = ""
    result = _run_build_tool(
        [
            "--check",
            "--unitree-sdk-root",
            os.fspath(unitree_root),
            "--video-dds-publisher-dir",
            os.fspath(video_dir),
            "--out",
            os.fspath(report_path),
        ],
        env=env,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["foundation_ready"] is True
    assert report["visual_events_codegen_ready"] is False
    assert report["visual_events_codegen_error"] == "not required for foundation check"


def test_build_tool_validates_full_bridge_generated_head_gaze_files(tmp_path):
    import tools.build_dds_bridge as build_dds_bridge

    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    for name in [
        "head_state_v1.hpp",
        "head_state_v1.cpp",
        "gaze_target_v1.hpp",
    ]:
        (generated_dir / name).write_text("// fake generated file\n", encoding="utf-8")

    with pytest.raises(build_dds_bridge.CheckError, match="gaze_target_v1.cpp"):
        build_dds_bridge.validate_full_bridge_generated_type_support(generated_dir)


def test_build_tool_full_bridge_configure_passes_generated_inputs_to_cmake(
    tmp_path,
    monkeypatch,
):
    import tools.build_dds_bridge as build_dds_bridge

    unitree_root = _make_minimal_unitree_sdk_root(tmp_path)
    video_dir = _make_minimal_video_dds_publisher_dir(tmp_path)
    build_dir = REPO_ROOT / "build" / "test-dds-bridge" / f"{tmp_path.name}-full-configure"
    shutil.rmtree(build_dir, ignore_errors=True)
    generated_dir = REPO_ROOT / "build" / "test-dds-codegen" / f"{tmp_path.name}-generated"
    shutil.rmtree(generated_dir, ignore_errors=True)
    generated_dir.mkdir(parents=True)
    for name in [
        "head_state_v1.hpp",
        "head_state_v1.cpp",
        "gaze_target_v1.hpp",
        "gaze_target_v1.cpp",
    ]:
        (generated_dir / name).write_text("// fake generated file\n", encoding="utf-8")

    commands: list[list[str]] = []

    def fake_run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(build_dds_bridge, "_run", fake_run)

    try:
        report = build_dds_bridge.configure_and_build(
            unitree_sdk_root=unitree_root,
            video_dds_publisher_dir=video_dir,
            build_dir=build_dir,
            target="visual_events_dds_bridge_probe",
            full_bridge_generated_dir=generated_dir,
        )
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)
        shutil.rmtree(generated_dir, ignore_errors=True)

    configure = commands[0]
    assert f"-DVISUAL_EVENTS_DDS_BRIDGE_FULL_BRIDGE=ON" in configure
    assert f"-DVISUAL_EVENTS_GENERATED_DDS_DIR={generated_dir}" in configure
    assert f"-DVISUAL_EVENTS_HEAD_STATE_HEADER={generated_dir / 'head_state_v1.hpp'}" in configure
    assert f"-DVISUAL_EVENTS_HEAD_STATE_SOURCE={generated_dir / 'head_state_v1.cpp'}" in configure
    assert f"-DVISUAL_EVENTS_GAZE_TARGET_HEADER={generated_dir / 'gaze_target_v1.hpp'}" in configure
    assert f"-DVISUAL_EVENTS_GAZE_TARGET_SOURCE={generated_dir / 'gaze_target_v1.cpp'}" in configure
    assert report["full_bridge_generated_dir"] == os.fspath(generated_dir)


def test_build_tool_rejects_report_paths_outside_repo_artifacts_without_writing(tmp_path):
    unitree_root = _make_minimal_unitree_sdk_root(tmp_path)
    video_dir = _make_minimal_video_dds_publisher_dir(tmp_path)
    outside_report = tmp_path / "foundation-report.json"

    result = _run_build_tool(
        [
            "--check",
            "--unitree-sdk-root",
            os.fspath(unitree_root),
            "--video-dds-publisher-dir",
            os.fspath(video_dir),
            "--out",
            os.fspath(outside_report),
        ],
    )

    assert result.returncode != 0
    assert "report path must be under repo artifacts/" in result.stderr
    assert not outside_report.exists()


def test_build_tool_rejects_build_dir_outside_repo_build(tmp_path, repo_report_path):
    unitree_root = _make_minimal_unitree_sdk_root(tmp_path)
    video_dir = _make_minimal_video_dds_publisher_dir(tmp_path)
    outside_build_dir = tmp_path / "outside-build"
    report_path = repo_report_path("outside-build-dir")

    result = _run_build_tool(
        [
            "--check",
            "--build-dir",
            os.fspath(outside_build_dir),
            "--unitree-sdk-root",
            os.fspath(unitree_root),
            "--video-dds-publisher-dir",
            os.fspath(video_dir),
            "--out",
            os.fspath(report_path),
        ],
    )

    assert result.returncode != 0
    assert "build dir must be under repo build/" in result.stderr
    assert not outside_build_dir.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert "build dir must be under repo build/" in report["error"]


def test_prepare_dds_codegen_toolchain_check_accepts_pinned_fake_idlc_without_writes(tmp_path):
    assert TOOLS_PREPARE_CODEGEN.exists()
    fake_idlc = _make_fake_idlc(tmp_path, version="0.10.2", backends="c cxx")
    result = _run_prepare_codegen_tool(
        ["--check", "--dry-run", "--idlc", os.fspath(fake_idlc)]
    )

    assert result.returncode == 0
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["dry_run"] is True
    assert report["will_write"] is False
    assert report["cyclonedds_version"] == "0.10.2"
    assert report["cyclonedds_cxx_version"] == "0.10.2"
    assert report["toolchain_dir"] == os.fspath(
        REPO_ROOT / "build" / "tools" / "cyclonedds-cxx-idlc-0.10.2"
    )
    assert report["idlc"] == os.fspath(fake_idlc.resolve())
    assert report["idlc_version"] == "0.10.2"
    assert report["cxx_backend_available"] is True
    assert report["probe_codegen"] is False
    assert report["generated_files"] == []
    assert report["expected_generated_files_present"] is False
    assert report["oracle_ok"] is False


def test_prepare_dds_codegen_toolchain_check_accepts_idlc_with_short_version_and_help_only(
    tmp_path,
):
    fake_idlc = _make_fake_idlc_v_h_only(tmp_path, version="0.10.2", backends="c cxx")
    result = _run_prepare_codegen_tool(
        ["--check", "--dry-run", "--idlc", os.fspath(fake_idlc)]
    )

    assert result.returncode == 0
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["dry_run"] is True
    assert report["will_write"] is False
    assert report["idlc_version"] == "0.10.2"
    assert report["idlc_version_arg"] == "-v"
    assert report["cxx_backend_available"] is True
    assert "available backends: c cxx" in report["idlc_backend_inspection_stdout"]
    assert "--help:" in " ".join(report["idlc_backend_inspection_errors"])
    assert "-l:" in " ".join(report["idlc_backend_inspection_errors"])
    assert report["probe_codegen"] is False
    assert report["oracle_ok"] is False


def test_prepare_dds_codegen_toolchain_check_accepts_backend_specific_cxx_help(
    tmp_path,
):
    fake_idlc = _make_fake_idlc_backend_help_only(tmp_path, version="0.10.2")
    result = _run_prepare_codegen_tool(
        ["--check", "--dry-run", "--idlc", os.fspath(fake_idlc)]
    )

    assert result.returncode == 0
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["cxx_backend_available"] is True
    assert "--bounded-sequence-template TEMPLATE" in report["idlc_backend_inspection_stdout"]
    assert report["probe_codegen"] is False
    assert report["oracle_ok"] is False


def test_prepare_dds_codegen_toolchain_check_rejects_generator_load_error_in_version_output(
    tmp_path,
):
    fake_idlc = _make_fake_idlc_version_generator_error(tmp_path)
    result = _run_prepare_codegen_tool(
        ["--check", "--dry-run", "--idlc", os.fspath(fake_idlc)]
    )

    assert result.returncode != 0
    assert "cannot load generator" in result.stderr
    report = json.loads(result.stdout)
    assert report["ok"] is False
    assert report["cxx_backend_available"] is False
    assert report["oracle_ok"] is False


@pytest.mark.parametrize("check_flag", ["--check", "--dry-run"])
def test_prepare_dds_codegen_toolchain_probe_codegen_rejects_check_and_dry_run_flags(
    tmp_path,
    check_flag: str,
):
    fake_idlc = _make_fake_idlc(tmp_path, version="0.10.2", backends="c cxx")
    probe_idl = _make_probe_idl(tmp_path)
    probe_output_dir = _repo_build_probe_dir(tmp_path, f"prepare-mutual-{check_flag[2:]}")

    try:
        result = _run_prepare_codegen_tool(
            [
                check_flag,
                "--probe-codegen",
                "--idlc",
                os.fspath(fake_idlc),
                "--probe-idl",
                os.fspath(probe_idl),
                "--probe-output-dir",
                os.fspath(probe_output_dir),
            ]
        )

        assert result.returncode != 0
        assert "--probe-codegen cannot be combined with --check or --dry-run" in result.stderr
        report = json.loads(result.stdout)
        assert report["ok"] is False
        assert report["probe_codegen"] is True
        assert report["will_write"] is False
        assert not probe_output_dir.exists()
    finally:
        shutil.rmtree(probe_output_dir, ignore_errors=True)


def test_prepare_dds_codegen_toolchain_probe_codegen_accepts_fake_idlc_that_writes_expected_files(
    tmp_path,
):
    fake_idlc = _make_fake_idlc(tmp_path, version="0.10.2", backends="c cxx")
    probe_idl = _make_probe_idl(tmp_path)
    probe_output_dir = _repo_build_probe_dir(tmp_path, "prepare-success")

    try:
        result = _run_prepare_codegen_tool(
            [
                "--probe-codegen",
                "--idlc",
                os.fspath(fake_idlc),
                "--probe-idl",
                os.fspath(probe_idl),
                "--probe-output-dir",
                os.fspath(probe_output_dir),
            ]
        )

        assert result.returncode == 0
        assert result.stderr == ""
        report = json.loads(result.stdout)
        assert report["ok"] is True
        assert report["dry_run"] is False
        assert report["will_write"] is True
        assert report["probe_codegen"] is True
        assert report["probe_idl"] == os.fspath(probe_idl.resolve())
        assert report["probe_output_dir"] == os.fspath(probe_output_dir.resolve())
        assert set(report["generated_files"]) == {"CameraFrame_.hpp", "CameraFrame_.cpp"}
        assert report["expected_generated_files_present"] is True
        assert report["expected_generated_file_presence"] == {
            "CameraFrame_.hpp": True,
            "CameraFrame_.cpp": True,
        }
        assert report["cxx_backend_available"] is True
        assert report["oracle_ok"] is True
        assert (probe_output_dir / "CameraFrame_.hpp").is_file()
        assert (probe_output_dir / "CameraFrame_.cpp").is_file()
    finally:
        shutil.rmtree(probe_output_dir, ignore_errors=True)


def test_prepare_dds_codegen_toolchain_probe_codegen_runs_from_probe_output_dir(
    tmp_path,
):
    fake_idlc = _make_fake_idlc_codegen_only_from_output_cwd(tmp_path, version="0.10.2")
    probe_idl = _make_probe_idl(tmp_path)
    probe_output_dir = _repo_build_probe_dir(tmp_path, "prepare-output-cwd")

    try:
        result = _run_prepare_codegen_tool(
            [
                "--probe-codegen",
                "--idlc",
                os.fspath(fake_idlc),
                "--probe-idl",
                os.fspath(probe_idl),
                "--probe-output-dir",
                os.fspath(probe_output_dir),
            ]
        )

        assert result.returncode == 0
        assert result.stderr == ""
        report = json.loads(result.stdout)
        assert report["ok"] is True
        assert report["oracle_ok"] is True
        assert report["probe_output_dir"] == os.fspath(probe_output_dir.resolve())
        assert report["idlc_codegen_cwd"] == os.fspath(probe_output_dir.resolve())
        assert report["codegen_probes"][0]["idlc_codegen_cwd"] == os.fspath(
            probe_output_dir.resolve()
        )
        assert set(report["generated_files"]) == {"CameraFrame_.hpp", "CameraFrame_.cpp"}
    finally:
        shutil.rmtree(probe_output_dir, ignore_errors=True)


def test_prepare_dds_codegen_toolchain_probe_codegen_defaults_to_repo_head_and_gaze_idls(
    tmp_path,
):
    fake_idlc = _make_fake_idlc(tmp_path, version="0.10.2", backends="c cxx")
    probe_output_dir = _repo_build_probe_dir(tmp_path, "prepare-default-head-gaze")

    try:
        result = _run_prepare_codegen_tool(
            [
                "--probe-codegen",
                "--idlc",
                os.fspath(fake_idlc),
                "--probe-output-dir",
                os.fspath(probe_output_dir),
            ]
        )

        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert report["oracle_ok"] is True
        assert report["probe_idls"] == [
            os.fspath(HEAD_STATE_IDL.resolve()),
            os.fspath(GAZE_TARGET_IDL.resolve()),
        ]
        assert set(report["generated_files"]) == {
            "head_state_v1.hpp",
            "head_state_v1.cpp",
            "gaze_target_v1.hpp",
            "gaze_target_v1.cpp",
        }
        assert report["expected_generated_file_presence"] == {
            "head_state_v1.hpp": True,
            "head_state_v1.cpp": True,
            "gaze_target_v1.hpp": True,
            "gaze_target_v1.cpp": True,
        }
        assert [
            probe["expected_generated_files"]
            for probe in report["codegen_probes"]
        ] == [
            ["head_state_v1.hpp", "head_state_v1.cpp"],
            ["gaze_target_v1.hpp", "gaze_target_v1.cpp"],
        ]
    finally:
        shutil.rmtree(probe_output_dir, ignore_errors=True)


def test_prepare_dds_codegen_toolchain_probe_codegen_accepts_repeatable_probe_idls(
    tmp_path,
):
    fake_idlc = _make_fake_idlc(tmp_path, version="0.10.2", backends="c cxx")
    first_idl = _make_probe_idl(tmp_path, stem="FirstProbe_")
    second_idl = _make_probe_idl(tmp_path, stem="SecondProbe_")
    probe_output_dir = _repo_build_probe_dir(tmp_path, "prepare-repeatable")

    try:
        result = _run_prepare_codegen_tool(
            [
                "--probe-codegen",
                "--idlc",
                os.fspath(fake_idlc),
                "--probe-idl",
                os.fspath(first_idl),
                "--probe-idl",
                os.fspath(second_idl),
                "--probe-output-dir",
                os.fspath(probe_output_dir),
            ]
        )

        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert report["oracle_ok"] is True
        assert report["probe_idls"] == [
            os.fspath(first_idl.resolve()),
            os.fspath(second_idl.resolve()),
        ]
        assert set(report["generated_files"]) == {
            "FirstProbe_.hpp",
            "FirstProbe_.cpp",
            "SecondProbe_.hpp",
            "SecondProbe_.cpp",
        }
        assert len(report["codegen_probes"]) == 2
    finally:
        shutil.rmtree(probe_output_dir, ignore_errors=True)


@pytest.mark.parametrize(
    ("version", "backends", "expected_error"),
    [
        ("0.11.0", "c cxx", "expected pinned idlc version 0.10.2"),
        ("0.10.2", "c", "cxx backend"),
    ],
)
def test_prepare_dds_codegen_toolchain_check_rejects_unpinned_or_non_cxx_fake_idlc(
    tmp_path,
    version: str,
    backends: str,
    expected_error: str,
):
    fake_idlc = _make_fake_idlc(tmp_path, version=version, backends=backends)
    result = _run_prepare_codegen_tool(
        ["--check", "--dry-run", "--idlc", os.fspath(fake_idlc)]
    )

    assert result.returncode != 0
    assert expected_error in result.stderr
    report = json.loads(result.stdout)
    assert report["ok"] is False
    assert expected_error in report["error"]
    assert report["idlc"] == os.fspath(fake_idlc.resolve())


def test_prepare_dds_codegen_toolchain_probe_codegen_rejects_missing_cxx_generator_even_with_zero_rc(
    tmp_path,
):
    fake_idlc = _make_fake_idlc(
        tmp_path,
        version="0.10.2",
        backends="c cxx",
        codegen="missing_cxx_rc0",
    )
    probe_idl = _make_probe_idl(tmp_path)
    probe_output_dir = _repo_build_probe_dir(tmp_path, "prepare-missing-generator")

    try:
        result = _run_prepare_codegen_tool(
            [
                "--probe-codegen",
                "--idlc",
                os.fspath(fake_idlc),
                "--probe-idl",
                os.fspath(probe_idl),
                "--probe-output-dir",
                os.fspath(probe_output_dir),
            ]
        )

        assert result.returncode != 0
        assert "cannot load generator" in result.stderr
        report = json.loads(result.stdout)
        assert report["ok"] is False
        assert report["probe_codegen"] is True
        assert report["idlc_codegen_returncode"] == 0
        assert "cannot load generator cxx" in report["idlc_codegen_stderr"]
        assert report["cxx_backend_available"] is False
        assert report["oracle_ok"] is False
        assert report["expected_generated_files_present"] is False
    finally:
        shutil.rmtree(probe_output_dir, ignore_errors=True)


def test_prepare_dds_codegen_toolchain_probe_codegen_rejects_header_without_source(tmp_path):
    fake_idlc = _make_fake_idlc(
        tmp_path,
        version="0.10.2",
        backends="c cxx",
        codegen="hpp_only",
    )
    probe_idl = _make_probe_idl(tmp_path)
    probe_output_dir = _repo_build_probe_dir(tmp_path, "prepare-hpp-only")

    try:
        result = _run_prepare_codegen_tool(
            [
                "--probe-codegen",
                "--idlc",
                os.fspath(fake_idlc),
                "--probe-idl",
                os.fspath(probe_idl),
                "--probe-output-dir",
                os.fspath(probe_output_dir),
            ]
        )

        assert result.returncode != 0
        assert "missing expected generated files: CameraFrame_.cpp" in result.stderr
        report = json.loads(result.stdout)
        assert report["ok"] is False
        assert report["generated_files"] == ["CameraFrame_.hpp"]
        assert report["expected_generated_file_presence"] == {
            "CameraFrame_.hpp": True,
            "CameraFrame_.cpp": False,
        }
        assert report["expected_generated_files_present"] is False
        assert report["oracle_ok"] is False
    finally:
        shutil.rmtree(probe_output_dir, ignore_errors=True)


def test_prepare_dds_codegen_toolchain_rejects_output_paths_outside_repo_build(tmp_path):
    fake_idlc = _make_fake_idlc(tmp_path, version="0.10.2", backends="c cxx")
    result = _run_prepare_codegen_tool(
        [
            "--check",
            "--dry-run",
            "--idlc",
            os.fspath(fake_idlc),
            "--toolchain-dir",
            os.fspath(tmp_path / "outside-repo-build"),
        ]
    )

    assert result.returncode != 0
    assert "toolchain dir must be under repo build/" in result.stderr


def test_prepare_dds_codegen_toolchain_rejects_probe_output_paths_outside_repo_build(tmp_path):
    fake_idlc = _make_fake_idlc(tmp_path, version="0.10.2", backends="c cxx")
    probe_idl = _make_probe_idl(tmp_path)
    result = _run_prepare_codegen_tool(
        [
            "--probe-codegen",
            "--idlc",
            os.fspath(fake_idlc),
            "--probe-idl",
            os.fspath(probe_idl),
            "--probe-output-dir",
            os.fspath(tmp_path / "outside-repo-build"),
        ]
    )

    assert result.returncode != 0
    assert "probe output dir must be under repo build/" in result.stderr


def test_prepare_dds_codegen_toolchain_prepare_success_uses_repo_local_wrapper_and_oracle(
    tmp_path,
):
    toolchain_dir = _repo_build_toolchain_dir(tmp_path, "prepare-success")
    fake_bin = _make_fake_prepare_tool_path(tmp_path)
    probe_idl = _make_probe_idl(tmp_path)
    command_log = tmp_path / "prepare-success.log"
    env = os.environ.copy()
    env["PATH"] = os.fspath(fake_bin)
    env["FAKE_INSTALL_DIR"] = os.fspath(toolchain_dir / "install")
    env["FAKE_TOOL_LOG"] = os.fspath(command_log)
    env["VISUAL_EVENTS_IDLC"] = os.fspath(tmp_path / "poison-env-idlc")

    try:
        result = _run_prepare_codegen_tool(
            [
                "--prepare",
                "--toolchain-dir",
                os.fspath(toolchain_dir),
                "--probe-idl",
                os.fspath(probe_idl),
            ],
            env=env,
        )

        assert result.returncode == 0, result.stderr
        assert result.stderr == ""
        report = json.loads(result.stdout)
        install_dir = toolchain_dir / "install"
        wrapper = toolchain_dir / "bin" / "idlc-cxx"
        assert report["ok"] is True
        assert report["mode"] == "prepare"
        assert report["prepare_toolchain"] is True
        assert report["toolchain_ready"] is True
        assert report["source_dir"] == os.fspath((toolchain_dir / "src").resolve())
        assert report["cyclonedds_source_dir"] == os.fspath((toolchain_dir / "src" / "cyclonedds").resolve())
        assert report["cyclonedds_cxx_source_dir"] == os.fspath((toolchain_dir / "src" / "cyclonedds-cxx").resolve())
        assert report["build_dir"] == os.fspath((toolchain_dir / "build").resolve())
        assert report["cyclonedds_build_dir"] == os.fspath((toolchain_dir / "build" / "cyclonedds").resolve())
        assert report["cyclonedds_cxx_build_dir"] == os.fspath((toolchain_dir / "build" / "cyclonedds-cxx").resolve())
        assert report["install_dir"] == os.fspath(install_dir.resolve())
        assert report["wrapper_idlc"] == os.fspath(wrapper.resolve())
        assert report["idlc"] == os.fspath(wrapper.resolve())
        assert report["ld_library_path_prepend"] == os.fspath((install_dir / "lib").resolve())
        assert report["probe_output_dir"] == os.fspath((toolchain_dir / "codegen_probe").resolve())
        assert report["oracle_ok"] is True
        assert set(report["generated_files"]) == {"CameraFrame_.hpp", "CameraFrame_.cpp"}
        assert report["required_tools"]["git"]["found"] is True
        assert report["required_tools"]["g++"]["found"] is True
        assert report["optional_tools"]["ninja"]["required"] is False
        assert report["optional_tools"]["bison"]["required"] is False
        assert report["optional_tools"]["flex"]["required"] is False
        assert (install_dir / "bin" / "idlc").is_file()
        assert (install_dir / "lib" / "libcycloneddsidlcxx.so").is_file()
        assert (install_dir / "lib" / "libcycloneddsidl.so").is_file()
        assert (install_dir / "lib" / "libddsc.so").is_file()
        assert wrapper.is_file()
        assert os.access(wrapper, os.X_OK)
        steps = [command["step"] for command in report["commands"]]
        assert steps == [
            "cyclonedds_ls_remote",
            "cyclonedds_cxx_ls_remote",
            "cyclonedds_clone",
            "cyclonedds_source_head",
            "cyclonedds_cxx_clone",
            "cyclonedds_cxx_source_head",
            "cyclonedds_configure",
            "cyclonedds_build_install",
            "cyclonedds_cxx_configure",
            "cyclonedds_cxx_build_install",
        ]
        assert report["cyclonedds_expected_commit"] == CYCLONEDDS_COMMIT
        assert report["cyclonedds_cxx_expected_commit"] == CYCLONEDDS_CXX_COMMIT
        assert report["cyclonedds_commit"] == CYCLONEDDS_COMMIT
        assert report["cyclonedds_cxx_commit"] == CYCLONEDDS_CXX_COMMIT
        assert "cmake --build" in command_log.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(toolchain_dir, ignore_errors=True)


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--check"],
        ["--dry-run"],
        ["--probe-codegen"],
        ["--idlc", "fake-idlc"],
    ],
)
def test_prepare_dds_codegen_toolchain_prepare_mutual_exclusions_fail_before_writing(
    tmp_path,
    extra_args: list[str],
):
    toolchain_dir = _repo_build_toolchain_dir(tmp_path, "prepare-mutual")
    fake_bin = _make_fake_prepare_tool_path(tmp_path)
    probe_idl = _make_probe_idl(tmp_path)
    command_log = tmp_path / "prepare-mutual.log"
    env = os.environ.copy()
    env["PATH"] = os.fspath(fake_bin)
    env["FAKE_INSTALL_DIR"] = os.fspath(toolchain_dir / "install")
    env["FAKE_TOOL_LOG"] = os.fspath(command_log)

    try:
        result = _run_prepare_codegen_tool(
            [
                "--prepare",
                *extra_args,
                "--toolchain-dir",
                os.fspath(toolchain_dir),
                "--probe-idl",
                os.fspath(probe_idl),
            ],
            env=env,
        )

        assert result.returncode != 0
        assert not toolchain_dir.exists()
        assert not command_log.exists()
        report = json.loads(result.stdout)
        assert report["ok"] is False
        assert report["mode"] == "prepare"
    finally:
        shutil.rmtree(toolchain_dir, ignore_errors=True)


def test_prepare_dds_codegen_toolchain_prepare_rejects_toolchain_dir_outside_repo_build_before_commands(
    tmp_path,
):
    outside_toolchain_dir = tmp_path / "outside-toolchain"
    fake_bin = _make_fake_prepare_tool_path(tmp_path)
    probe_idl = _make_probe_idl(tmp_path)
    command_log = tmp_path / "prepare-outside.log"
    env = os.environ.copy()
    env["PATH"] = os.fspath(fake_bin)
    env["FAKE_INSTALL_DIR"] = os.fspath(outside_toolchain_dir / "install")
    env["FAKE_TOOL_LOG"] = os.fspath(command_log)

    result = _run_prepare_codegen_tool(
        [
            "--prepare",
            "--toolchain-dir",
            os.fspath(outside_toolchain_dir),
            "--probe-idl",
            os.fspath(probe_idl),
        ],
        env=env,
    )

    assert result.returncode != 0
    assert "toolchain dir must be under repo build/" in result.stderr
    assert not outside_toolchain_dir.exists()
    assert not command_log.exists()


def test_prepare_dds_codegen_toolchain_prepare_missing_required_tool_fails_before_clone(
    tmp_path,
):
    toolchain_dir = _repo_build_toolchain_dir(tmp_path, "prepare-missing-tool")
    fake_bin = _make_fake_prepare_tool_path(tmp_path)
    (fake_bin / "g++").unlink()
    probe_idl = _make_probe_idl(tmp_path)
    command_log = tmp_path / "prepare-missing-tool.log"
    env = os.environ.copy()
    env["PATH"] = os.fspath(fake_bin)
    env["FAKE_INSTALL_DIR"] = os.fspath(toolchain_dir / "install")
    env["FAKE_TOOL_LOG"] = os.fspath(command_log)

    try:
        result = _run_prepare_codegen_tool(
            [
                "--prepare",
                "--toolchain-dir",
                os.fspath(toolchain_dir),
                "--probe-idl",
                os.fspath(probe_idl),
            ],
            env=env,
        )

        assert result.returncode != 0
        report = json.loads(result.stdout)
        assert report["failed_step"] == "preflight_required_tools"
        assert report["required_tools"]["g++"]["found"] is False
        assert "missing required tools: g++" in report["error"]
        assert not toolchain_dir.exists()
        assert not command_log.exists()
    finally:
        shutil.rmtree(toolchain_dir, ignore_errors=True)


def test_prepare_dds_codegen_toolchain_prepare_wrong_ls_remote_hash_fails_before_clone_or_configure(
    tmp_path,
):
    toolchain_dir = _repo_build_toolchain_dir(tmp_path, "prepare-wrong-hash")
    fake_bin = _make_fake_prepare_tool_path(tmp_path)
    probe_idl = _make_probe_idl(tmp_path)
    env = os.environ.copy()
    env["PATH"] = os.fspath(fake_bin)
    env["FAKE_INSTALL_DIR"] = os.fspath(toolchain_dir / "install")
    env["FAKE_LS_REMOTE_BAD"] = "cyclonedds"

    try:
        result = _run_prepare_codegen_tool(
            [
                "--prepare",
                "--toolchain-dir",
                os.fspath(toolchain_dir),
                "--probe-idl",
                os.fspath(probe_idl),
            ],
            env=env,
        )

        assert result.returncode != 0
        report = json.loads(result.stdout)
        assert report["failed_step"] == "cyclonedds_ls_remote"
        assert CYCLONEDDS_COMMIT in report["error"]
        steps = [command["step"] for command in report["commands"]]
        assert steps == ["cyclonedds_ls_remote"]
        assert not (toolchain_dir / "src" / "cyclonedds").exists()
        assert not (toolchain_dir / "build").exists()
    finally:
        shutil.rmtree(toolchain_dir, ignore_errors=True)


def test_prepare_dds_codegen_toolchain_prepare_git_clone_failure_stops_before_cmake(
    tmp_path,
):
    toolchain_dir = _repo_build_toolchain_dir(tmp_path, "prepare-clone-fail")
    fake_bin = _make_fake_prepare_tool_path(tmp_path)
    probe_idl = _make_probe_idl(tmp_path)
    env = os.environ.copy()
    env["PATH"] = os.fspath(fake_bin)
    env["FAKE_INSTALL_DIR"] = os.fspath(toolchain_dir / "install")
    env["FAKE_GIT_CLONE_FAIL"] = "cyclonedds"

    try:
        result = _run_prepare_codegen_tool(
            [
                "--prepare",
                "--toolchain-dir",
                os.fspath(toolchain_dir),
                "--probe-idl",
                os.fspath(probe_idl),
            ],
            env=env,
        )

        assert result.returncode != 0
        report = json.loads(result.stdout)
        assert report["failed_step"] == "cyclonedds_clone"
        steps = [command["step"] for command in report["commands"]]
        assert steps == [
            "cyclonedds_ls_remote",
            "cyclonedds_cxx_ls_remote",
            "cyclonedds_clone",
        ]
        assert report["commands"][-1]["returncode"] == 42
        assert not any("configure" in step for step in steps)
    finally:
        shutil.rmtree(toolchain_dir, ignore_errors=True)


def test_prepare_dds_codegen_toolchain_prepare_bad_cloned_head_stops_before_cmake(
    tmp_path,
):
    toolchain_dir = _repo_build_toolchain_dir(tmp_path, "prepare-bad-cloned-head")
    fake_bin = _make_fake_prepare_tool_path(tmp_path)
    probe_idl = _make_probe_idl(tmp_path)
    env = os.environ.copy()
    env["PATH"] = os.fspath(fake_bin)
    env["FAKE_INSTALL_DIR"] = os.fspath(toolchain_dir / "install")
    env["FAKE_REV_PARSE_BAD"] = "cyclonedds"

    try:
        result = _run_prepare_codegen_tool(
            [
                "--prepare",
                "--toolchain-dir",
                os.fspath(toolchain_dir),
                "--probe-idl",
                os.fspath(probe_idl),
            ],
            env=env,
        )

        assert result.returncode != 0
        report = json.loads(result.stdout)
        assert report["failed_step"] == "cyclonedds_source_head"
        assert report["cyclonedds_expected_commit"] == CYCLONEDDS_COMMIT
        assert report["cyclonedds_commit"] == "1111111111111111111111111111111111111111"
        steps = [command["step"] for command in report["commands"]]
        assert steps == [
            "cyclonedds_ls_remote",
            "cyclonedds_cxx_ls_remote",
            "cyclonedds_clone",
            "cyclonedds_source_head",
        ]
        assert not any("configure" in step for step in steps)
    finally:
        shutil.rmtree(toolchain_dir, ignore_errors=True)


def test_prepare_dds_codegen_toolchain_prepare_missing_idlcxx_artifact_fails_before_oracle(
    tmp_path,
):
    toolchain_dir = _repo_build_toolchain_dir(tmp_path, "prepare-missing-idlcxx")
    fake_bin = _make_fake_prepare_tool_path(tmp_path)
    probe_idl = _make_probe_idl(tmp_path)
    env = os.environ.copy()
    env["PATH"] = os.fspath(fake_bin)
    env["FAKE_INSTALL_DIR"] = os.fspath(toolchain_dir / "install")
    env["FAKE_SKIP_IDLCXX"] = "1"

    try:
        result = _run_prepare_codegen_tool(
            [
                "--prepare",
                "--toolchain-dir",
                os.fspath(toolchain_dir),
                "--probe-idl",
                os.fspath(probe_idl),
            ],
            env=env,
        )

        assert result.returncode != 0
        report = json.loads(result.stdout)
        assert report["failed_step"] == "require_artifacts"
        assert "libcycloneddsidlcxx.so" in report["error"]
        assert "idlc_codegen_returncode" not in report
        assert not (toolchain_dir / "codegen_probe").exists()
    finally:
        shutil.rmtree(toolchain_dir, ignore_errors=True)


def test_prepare_dds_codegen_toolchain_prepare_installed_idlc_oracle_failure_propagates(
    tmp_path,
):
    toolchain_dir = _repo_build_toolchain_dir(tmp_path, "prepare-oracle-fail")
    fake_bin = _make_fake_prepare_tool_path(tmp_path)
    probe_idl = _make_probe_idl(tmp_path)
    env = os.environ.copy()
    env["PATH"] = os.fspath(fake_bin)
    env["FAKE_INSTALL_DIR"] = os.fspath(toolchain_dir / "install")
    env["FAKE_INSTALLED_IDLC_CODEGEN"] = "hpp_only"

    try:
        result = _run_prepare_codegen_tool(
            [
                "--prepare",
                "--toolchain-dir",
                os.fspath(toolchain_dir),
                "--probe-idl",
                os.fspath(probe_idl),
            ],
            env=env,
        )

        assert result.returncode != 0
        assert "missing expected generated files: CameraFrame_.cpp" in result.stderr
        report = json.loads(result.stdout)
        assert report["failed_step"] == "codegen_oracle"
        assert report["oracle_ok"] is False
        assert report["generated_files"] == ["CameraFrame_.hpp"]
        assert report["expected_generated_file_presence"] == {
            "CameraFrame_.hpp": True,
            "CameraFrame_.cpp": False,
        }
    finally:
        shutil.rmtree(toolchain_dir, ignore_errors=True)


def test_build_tool_missing_root_and_full_bridge_missing_generator_fail_fast(
    tmp_path,
    repo_report_path,
):
    assert TOOLS_BUILD.exists()
    missing_root = tmp_path / "missing-unitree"
    video_dir = _make_minimal_video_dds_publisher_dir(tmp_path)
    result = _run_build_tool(
        [
            "--check",
            "--unitree-sdk-root",
            os.fspath(missing_root),
            "--video-dds-publisher-dir",
            os.fspath(video_dir),
            "--out",
            os.fspath(repo_report_path("missing-root")),
        ],
    )
    assert result.returncode != 0
    assert "UNITREE_SDK_ROOT" in result.stderr
    assert os.fspath(missing_root) in result.stderr

    unitree_root = _make_minimal_unitree_sdk_root(tmp_path)
    report_path = repo_report_path("full-bridge-missing-generator")
    env = os.environ.copy()
    env["PATH"] = ""
    result = _run_build_tool(
        [
            "--check",
            "--check-full-bridge",
            "--unitree-sdk-root",
            os.fspath(unitree_root),
            "--video-dds-publisher-dir",
            os.fspath(video_dir),
            "--out",
            os.fspath(report_path),
        ],
        env=env,
    )
    assert result.returncode != 0
    assert "explicit --idlc or VISUAL_EVENTS_IDLC is required" in result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["foundation_ready"] is True
    assert report["visual_events_codegen_ready"] is False
    assert "explicit --idlc or VISUAL_EVENTS_IDLC is required" in report["visual_events_codegen_error"]


def test_build_tool_full_bridge_accepts_explicit_pinned_fake_idlc(tmp_path, repo_report_path):
    unitree_root = _make_minimal_unitree_sdk_root(tmp_path)
    video_dir = _make_minimal_video_dds_publisher_dir(tmp_path)
    fake_idlc = _make_fake_idlc(tmp_path, version="0.10.2", backends="c cxx")
    probe_idl = _make_probe_idl(tmp_path)
    probe_output_dir = _repo_build_probe_dir(tmp_path, "build-explicit-success")
    report_path = repo_report_path("full-bridge-explicit")
    env = os.environ.copy()
    env["PATH"] = ""
    try:
        result = _run_build_tool(
            [
                "--check",
                "--check-full-bridge",
                "--idlc",
                os.fspath(fake_idlc),
                "--codegen-probe-idl",
                os.fspath(probe_idl),
                "--codegen-probe-output-dir",
                os.fspath(probe_output_dir),
                "--unitree-sdk-root",
                os.fspath(unitree_root),
                "--video-dds-publisher-dir",
                os.fspath(video_dir),
                "--out",
                os.fspath(report_path),
            ],
            env=env,
        )

        assert result.returncode == 0
        assert result.stderr == ""
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["ok"] is True
        assert report["foundation_ready"] is True
        assert report["visual_events_codegen_ready"] is True
        assert report["visual_events_codegen_error"] == ""
        assert report["idl_generator"] == os.fspath(fake_idlc.resolve())
        assert report["idl_generator_version"] == "0.10.2"
        assert report["idl_generator_cxx_backend"] is True
        assert report["probe_codegen"] is True
        assert report["oracle_ok"] is True
        assert set(report["generated_files"]) == {"CameraFrame_.hpp", "CameraFrame_.cpp"}
    finally:
        shutil.rmtree(probe_output_dir, ignore_errors=True)


def test_build_tool_full_bridge_default_probes_repo_head_and_gaze_idls(
    tmp_path,
    repo_report_path,
):
    unitree_root = _make_minimal_unitree_sdk_root(tmp_path)
    video_dir = _make_minimal_video_dds_publisher_dir(tmp_path)
    fake_idlc = _make_fake_idlc(tmp_path, version="0.10.2", backends="c cxx")
    probe_output_dir = _repo_build_probe_dir(tmp_path, "build-default-head-gaze")
    report_path = repo_report_path("full-bridge-default-head-gaze")
    env = os.environ.copy()
    env["PATH"] = ""

    try:
        result = _run_build_tool(
            [
                "--check",
                "--check-full-bridge",
                "--idlc",
                os.fspath(fake_idlc),
                "--codegen-probe-output-dir",
                os.fspath(probe_output_dir),
                "--unitree-sdk-root",
                os.fspath(unitree_root),
                "--video-dds-publisher-dir",
                os.fspath(video_dir),
                "--out",
                os.fspath(report_path),
            ],
            env=env,
        )

        assert result.returncode == 0, result.stderr
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["ok"] is True
        assert report["visual_events_codegen_ready"] is True
        assert report["probe_idls"] == [
            os.fspath(HEAD_STATE_IDL.resolve()),
            os.fspath(GAZE_TARGET_IDL.resolve()),
        ]
        assert set(report["generated_files"]) == {
            "head_state_v1.hpp",
            "head_state_v1.cpp",
            "gaze_target_v1.hpp",
            "gaze_target_v1.cpp",
        }
        assert report["expected_generated_file_presence"] == {
            "head_state_v1.hpp": True,
            "head_state_v1.cpp": True,
            "gaze_target_v1.hpp": True,
            "gaze_target_v1.cpp": True,
        }
        assert [probe["idl"] for probe in report["codegen_probes"]] == [
            os.fspath(HEAD_STATE_IDL.resolve()),
            os.fspath(GAZE_TARGET_IDL.resolve()),
        ]
    finally:
        shutil.rmtree(probe_output_dir, ignore_errors=True)


def test_build_tool_full_bridge_accepts_visual_events_idlc_env(tmp_path, repo_report_path):
    unitree_root = _make_minimal_unitree_sdk_root(tmp_path)
    video_dir = _make_minimal_video_dds_publisher_dir(tmp_path)
    fake_idlc = _make_fake_idlc(tmp_path, version="0.10.2", backends="c cxx")
    probe_idl = _make_probe_idl(tmp_path)
    probe_output_dir = _repo_build_probe_dir(tmp_path, "build-env-success")
    report_path = repo_report_path("full-bridge-env")
    env = os.environ.copy()
    env["PATH"] = ""
    env["VISUAL_EVENTS_IDLC"] = os.fspath(fake_idlc)
    try:
        result = _run_build_tool(
            [
                "--check",
                "--check-full-bridge",
                "--codegen-probe-idl",
                os.fspath(probe_idl),
                "--codegen-probe-output-dir",
                os.fspath(probe_output_dir),
                "--unitree-sdk-root",
                os.fspath(unitree_root),
                "--video-dds-publisher-dir",
                os.fspath(video_dir),
                "--out",
                os.fspath(report_path),
            ],
            env=env,
        )

        assert result.returncode == 0
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["visual_events_codegen_ready"] is True
        assert report["idl_generator"] == os.fspath(fake_idlc.resolve())
        assert report["oracle_ok"] is True
    finally:
        shutil.rmtree(probe_output_dir, ignore_errors=True)


def test_build_tool_full_bridge_default_fails_when_one_repo_idl_lacks_cpp(
    tmp_path,
    repo_report_path,
):
    unitree_root = _make_minimal_unitree_sdk_root(tmp_path)
    video_dir = _make_minimal_video_dds_publisher_dir(tmp_path)
    fake_idlc = _make_fake_idlc_hpp_only_for_stem(
        tmp_path,
        version="0.10.2",
        backends="c cxx",
        hpp_only_stem="gaze_target_v1",
    )
    probe_output_dir = _repo_build_probe_dir(tmp_path, "build-default-gaze-hpp-only")
    report_path = repo_report_path("full-bridge-default-gaze-hpp-only")
    env = os.environ.copy()
    env["PATH"] = ""

    try:
        result = _run_build_tool(
            [
                "--check",
                "--check-full-bridge",
                "--idlc",
                os.fspath(fake_idlc),
                "--codegen-probe-output-dir",
                os.fspath(probe_output_dir),
                "--unitree-sdk-root",
                os.fspath(unitree_root),
                "--video-dds-publisher-dir",
                os.fspath(video_dir),
                "--out",
                os.fspath(report_path),
            ],
            env=env,
        )

        assert result.returncode != 0
        assert "gaze_target_v1.cpp" in result.stderr
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["ok"] is False
        assert report["foundation_ready"] is True
        assert report["visual_events_codegen_ready"] is False
        assert "gaze_target_v1.cpp" in report["visual_events_codegen_error"]
        assert report["expected_generated_file_presence"] == {
            "head_state_v1.hpp": True,
            "head_state_v1.cpp": True,
            "gaze_target_v1.hpp": True,
            "gaze_target_v1.cpp": False,
        }
        assert report["codegen_probes"][1]["idl"] == os.fspath(GAZE_TARGET_IDL.resolve())
        assert report["codegen_probes"][1]["expected_generated_file_presence"] == {
            "gaze_target_v1.hpp": True,
            "gaze_target_v1.cpp": False,
        }
    finally:
        shutil.rmtree(probe_output_dir, ignore_errors=True)


def test_build_tool_full_bridge_rejects_fake_idlc_that_does_not_write_expected_cpp(
    tmp_path,
    repo_report_path,
):
    unitree_root = _make_minimal_unitree_sdk_root(tmp_path)
    video_dir = _make_minimal_video_dds_publisher_dir(tmp_path)
    fake_idlc = _make_fake_idlc(
        tmp_path,
        version="0.10.2",
        backends="c cxx",
        codegen="hpp_only",
    )
    probe_idl = _make_probe_idl(tmp_path)
    probe_output_dir = _repo_build_probe_dir(tmp_path, "build-hpp-only")
    report_path = repo_report_path("full-bridge-hpp-only")
    env = os.environ.copy()
    env["PATH"] = ""

    try:
        result = _run_build_tool(
            [
                "--check",
                "--check-full-bridge",
                "--idlc",
                os.fspath(fake_idlc),
                "--codegen-probe-idl",
                os.fspath(probe_idl),
                "--codegen-probe-output-dir",
                os.fspath(probe_output_dir),
                "--unitree-sdk-root",
                os.fspath(unitree_root),
                "--video-dds-publisher-dir",
                os.fspath(video_dir),
                "--out",
                os.fspath(report_path),
            ],
            env=env,
        )

        assert result.returncode != 0
        assert "missing expected generated files: CameraFrame_.cpp" in result.stderr
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["ok"] is False
        assert report["foundation_ready"] is True
        assert report["visual_events_codegen_ready"] is False
        assert "CameraFrame_.cpp" in report["visual_events_codegen_error"]
        assert report["oracle_ok"] is False
        assert report["expected_generated_files_present"] is False
    finally:
        shutil.rmtree(probe_output_dir, ignore_errors=True)


def test_build_tool_full_bridge_ignores_path_idlc_without_explicit_idlc(
    tmp_path,
    repo_report_path,
):
    unitree_root = _make_minimal_unitree_sdk_root(tmp_path)
    video_dir = _make_minimal_video_dds_publisher_dir(tmp_path)
    path_bin = tmp_path / "path-bin"
    path_bin.mkdir()
    fake_path_idlc = path_bin / "idlc"
    fake_path_idlc.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        "  --version) printf '%s\\n' 'CycloneDDS idlc 0.10.2' ;;\n"
        "  --help|-l) printf '%s\\n' 'available backends: c cxx' ;;\n"
        "  *) exit 64 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_path_idlc.chmod(0o755)
    report_path = repo_report_path("full-bridge-path")
    env = os.environ.copy()
    env["PATH"] = os.fspath(path_bin)
    env.pop("VISUAL_EVENTS_IDLC", None)

    result = _run_build_tool(
        [
            "--check",
            "--check-full-bridge",
            "--unitree-sdk-root",
            os.fspath(unitree_root),
            "--video-dds-publisher-dir",
            os.fspath(video_dir),
            "--out",
            os.fspath(report_path),
        ],
        env=env,
    )

    assert result.returncode != 0
    assert "explicit --idlc or VISUAL_EVENTS_IDLC is required" in result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["foundation_ready"] is True
    assert report["visual_events_codegen_ready"] is False
    assert "explicit --idlc or VISUAL_EVENTS_IDLC is required" in report["visual_events_codegen_error"]


def test_run_probe_validates_complete_status_frame_abi(tmp_path):
    import tools.build_dds_bridge as build_dds_bridge

    probe = tmp_path / "visual_events_dds_bridge_probe"
    probe.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' '{\"protocol_version\":1,\"type\":\"status\",\"code\":\"probe_ok\",\"message\":\"ok\",\"mode\":\"probe\"}'\n",
        encoding="utf-8",
    )
    probe.chmod(0o755)

    report = build_dds_bridge.run_probe(tmp_path)
    assert report["probe_status"]["code"] == "probe_ok"

    probe.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' '{\"protocol_version\":1,\"code\":\"probe_ok\",\"message\":\"ok\"}'\n",
        encoding="utf-8",
    )
    with pytest.raises(build_dds_bridge.CheckError, match="type=status"):
        build_dds_bridge.run_probe(tmp_path)


def test_native_bridge_build_and_probe_artifacts_are_ignored():
    gitignore = GITIGNORE.read_text(encoding="utf-8")
    assert "build/" in gitignore
    assert "artifacts/" in gitignore
    assert "native/dds_bridge" not in gitignore
