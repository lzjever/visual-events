#include "visual_events/dds_bridge/runtime_loop.hpp"

#include <cerrno>
#include <condition_variable>
#include <exception>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <poll.h>
#include <unistd.h>
#endif

namespace visual_events {
namespace dds_bridge {
namespace {

struct RuntimeLoopState {
    std::mutex mutex;
    std::condition_variable cv;
    bool stop_requested = false;
    bool emitter_stop_requested = false;
    bool camera_pending = false;
    bool head_pending = false;
    bool error_pending = false;
    int exit_code = 0;
    CameraJpegFrame camera;
    HeadStateFrame head;
    std::string error_code;
    std::string error_message;
};

void EnqueueCamera(RuntimeLoopState* state, const CameraJpegFrame& frame) {
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->stop_requested) {
            return;
        }
        state->camera = frame;
        state->camera_pending = true;
    }
    state->cv.notify_one();
}

void EnqueueHead(RuntimeLoopState* state, const HeadStateFrame& frame) {
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->stop_requested) {
            return;
        }
        state->head = frame;
        state->head_pending = true;
    }
    state->cv.notify_one();
}

void EnqueueFatal(
    RuntimeLoopState* state,
    std::string code,
    std::string message,
    std::ostream* diagnostics) {
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->error_code = std::move(code);
        state->error_message = std::move(message);
        state->error_pending = true;
        state->exit_code = 1;
        state->stop_requested = true;
        if (diagnostics != nullptr) {
            *diagnostics << state->error_code << ": " << state->error_message << '\n';
        }
    }
    state->cv.notify_all();
}

void RequestEmitterStop(RuntimeLoopState* state) {
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->stop_requested = true;
        state->emitter_stop_requested = true;
    }
    state->cv.notify_all();
}

void FinishInput(RuntimeLoopState* state, bool request_stop) {
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (request_stop) {
            state->stop_requested = true;
        }
    }
    state->cv.notify_all();
}

bool StopRequested(RuntimeLoopState* state) {
    std::lock_guard<std::mutex> lock(state->mutex);
    return state->stop_requested;
}

void WaitForStop(RuntimeLoopState* state) {
    std::unique_lock<std::mutex> lock(state->mutex);
    state->cv.wait(lock, [&] { return state->stop_requested; });
}

void EmitterLoop(RuntimeLoopState* state, std::ostream* output) {
    while (true) {
        std::vector<std::string> lines;
        {
            std::unique_lock<std::mutex> lock(state->mutex);
            state->cv.wait(lock, [&] {
                return state->emitter_stop_requested || state->camera_pending ||
                       state->head_pending || state->error_pending;
            });

            if (state->camera_pending) {
                lines.push_back(EncodeCameraJpegFrame(state->camera));
                state->camera_pending = false;
            }
            if (state->head_pending) {
                lines.push_back(EncodeHeadStateFrame(state->head));
                state->head_pending = false;
            }
            if (state->error_pending) {
                lines.push_back(EncodeErrorFrame(state->error_code, state->error_message, true));
                state->error_pending = false;
            }

            if (lines.empty() && state->emitter_stop_requested) {
                return;
            }
        }

        for (const std::string& line : lines) {
            *output << line << '\n';
            output->flush();
        }
    }
}

bool HandleGazeTargetLine(
    RuntimeBackend* backend,
    std::mutex* backend_mutex,
    RuntimeLoopState* state,
    const std::string& line,
    std::ostream* diagnostics) {
    const auto parsed = ParseGazeTargetLine(line);
    if (!parsed.ok) {
        EnqueueFatal(state, "invalid_gaze_target", parsed.error, diagnostics);
        return false;
    }

    std::string publish_error;
    bool published = false;
    try {
        std::lock_guard<std::mutex> lock(*backend_mutex);
        published = backend->PublishGaze(parsed.frame, &publish_error);
    } catch (const std::exception& exc) {
        publish_error = exc.what();
        published = false;
    } catch (...) {
        publish_error = "unknown gaze_target publish error";
        published = false;
    }
    if (!published) {
        if (publish_error.empty()) {
            publish_error = "gaze_target publish failed";
        }
        EnqueueFatal(state, "publish_gaze_failed", publish_error, diagnostics);
        return false;
    }
    return true;
}

void StdinStreamLoop(
    RuntimeBackend* backend,
    std::mutex* backend_mutex,
    RuntimeLoopState* state,
    std::istream* input,
    std::ostream* diagnostics) {
    std::string line;
    while (std::getline(*input, line)) {
        if (StopRequested(state)) {
            FinishInput(state, false);
            return;
        }
        if (!HandleGazeTargetLine(backend, backend_mutex, state, line, diagnostics)) {
            FinishInput(state, false);
            return;
        }
    }
    FinishInput(state, !StopRequested(state));
}

bool IsStandardInputStream(std::istream* input) {
    return input == &std::cin || input->rdbuf() == std::cin.rdbuf();
}

#if defined(__unix__) || defined(__APPLE__)
bool PopBufferedLine(std::string* buffer, std::string* line) {
    const std::string::size_type newline = buffer->find('\n');
    if (newline == std::string::npos) {
        return false;
    }
    *line = buffer->substr(0, newline);
    buffer->erase(0, newline + 1);
    return true;
}

void StdinFdLoop(
    RuntimeBackend* backend,
    std::mutex* backend_mutex,
    RuntimeLoopState* state,
    std::ostream* diagnostics) {
    constexpr int kPollTimeoutMs = 50;
    std::string buffer;
    std::string line;
    char chunk[4096];

    while (true) {
        while (PopBufferedLine(&buffer, &line)) {
            if (StopRequested(state)) {
                FinishInput(state, false);
                return;
            }
            if (!HandleGazeTargetLine(backend, backend_mutex, state, line, diagnostics)) {
                FinishInput(state, false);
                return;
            }
        }

        if (StopRequested(state)) {
            FinishInput(state, false);
            return;
        }

        pollfd input_fd{};
        input_fd.fd = STDIN_FILENO;
        input_fd.events = POLLIN | POLLHUP | POLLERR;
        const int ready = poll(&input_fd, 1, kPollTimeoutMs);
        if (ready < 0) {
            if (errno == EINTR) {
                continue;
            }
            EnqueueFatal(state, "stdin_read_failed", "failed to poll stdin", diagnostics);
            FinishInput(state, false);
            return;
        }
        if (ready == 0) {
            continue;
        }
        if ((input_fd.revents & POLLNVAL) != 0) {
            EnqueueFatal(state, "stdin_read_failed", "stdin is not readable", diagnostics);
            FinishInput(state, false);
            return;
        }

        const ssize_t bytes_read = read(STDIN_FILENO, chunk, sizeof(chunk));
        if (bytes_read > 0) {
            buffer.append(chunk, static_cast<size_t>(bytes_read));
            continue;
        }
        if (bytes_read == 0) {
            if (!buffer.empty()) {
                if (!HandleGazeTargetLine(backend, backend_mutex, state, buffer, diagnostics)) {
                    FinishInput(state, false);
                    return;
                }
            }
            FinishInput(state, !StopRequested(state));
            return;
        }
        if (errno == EINTR || errno == EAGAIN) {
            continue;
        }
        EnqueueFatal(state, "stdin_read_failed", "failed to read stdin", diagnostics);
        FinishInput(state, false);
        return;
    }
}
#endif

void StdinLoop(
    RuntimeBackend* backend,
    std::mutex* backend_mutex,
    RuntimeLoopState* state,
    std::istream* input,
    std::ostream* diagnostics) {
#if defined(__unix__) || defined(__APPLE__)
    if (IsStandardInputStream(input)) {
        StdinFdLoop(backend, backend_mutex, state, diagnostics);
        return;
    }
#endif
    StdinStreamLoop(backend, backend_mutex, state, input, diagnostics);
}

}  // namespace

int RunRuntimeLoop(
    RuntimeBackend& backend,
    std::istream& input,
    std::ostream& output,
    std::ostream& diagnostics) {
    RuntimeLoopState state;
    std::mutex backend_mutex;
    RuntimeBackendCallbacks callbacks;
    callbacks.camera = [&state](const CameraJpegFrame& frame) { EnqueueCamera(&state, frame); };
    callbacks.head = [&state](const HeadStateFrame& frame) { EnqueueHead(&state, frame); };
    callbacks.fatal = [&state, &diagnostics](std::string code, std::string message) {
        EnqueueFatal(&state, std::move(code), std::move(message), &diagnostics);
    };

    std::thread emitter([&state, &output] { EmitterLoop(&state, &output); });

    std::string startup_error;
    bool started = false;
    try {
        started = backend.Start(callbacks, &startup_error);
    } catch (const std::exception& exc) {
        startup_error = exc.what();
        started = false;
    } catch (...) {
        startup_error = "unknown runtime backend startup error";
        started = false;
    }

    if (!started) {
        if (startup_error.empty()) {
            startup_error = "runtime backend startup failed";
        }
        EnqueueFatal(&state, "dds_init_failed", startup_error, &diagnostics);
        backend.Close();
        RequestEmitterStop(&state);
        emitter.join();
        return 1;
    }

    std::thread stdin_reader(
        [&backend, &backend_mutex, &state, &input, &diagnostics] {
            StdinLoop(&backend, &backend_mutex, &state, &input, &diagnostics);
        });
    WaitForStop(&state);
    {
        std::lock_guard<std::mutex> lock(backend_mutex);
        backend.Close();
    }
    stdin_reader.join();
    RequestEmitterStop(&state);
    emitter.join();

    std::lock_guard<std::mutex> lock(state.mutex);
    return state.exit_code;
}

}  // namespace dds_bridge
}  // namespace visual_events
