#pragma once

#include <android/looper.h>
#include <array>
#include <mutex>
#include <string>
#include <vector>

namespace android_controls {
// JNI IDs shared with VkNRCActivity. Methods use VkNRCState::Method values.
enum Option { Left, Right, Nee, Fast, Roughness, Indirect, Bounces, Resolution, Accumulate, Window, Pause, Lock, Ema,
              Spp, MaxWindow, Ready, BenchmarkActive, Count };
enum Action { Retrain = 20, Restart, Benchmark, Capture, StopBenchmark, TrainOneFrame, CycleView };
struct Command { int id, value; };
using Snapshot = std::array<int, Count>;

// JNI and input callbacks only enqueue commands. Vulkan state belongs to the NativeActivity glue thread.
class Mailbox {
	std::mutex mutex_;
	ALooper *looper_{};
	std::vector<Command> commands_;
	Snapshot state_{};
	std::string status_;

public:
	void Bind(ALooper *looper, int resolution) {
		std::lock_guard guard(mutex_);
		looper_ = looper;
		commands_.clear();
		state_ = {1, 0, 1, 0, 500, 0, 64, resolution, 1, 0, 0, 0, 0, 0, 64, 0, 0};
		status_ = "Starting renderer";
	}
	void Close() {
		std::lock_guard guard(mutex_);
		state_[Ready] = 0;
		looper_ = nullptr;
		commands_.clear();
	}
	void Enqueue(Command command) {
		std::lock_guard guard(mutex_);
		if (!looper_) return;
		commands_.push_back(command);
		ALooper_wake(looper_);
	}
	std::vector<Command> Take() {
		std::lock_guard guard(mutex_);
		std::vector<Command> result;
		result.swap(commands_);
		return result;
	}
	bool Pending() { std::lock_guard guard(mutex_); return !commands_.empty(); }
	Snapshot Get() { std::lock_guard guard(mutex_); return state_; }
	void Publish(Snapshot state) { std::lock_guard guard(mutex_); state_ = state; }
	std::string Status() { std::lock_guard guard(mutex_); return status_; }
	void Status(std::string message) { std::lock_guard guard(mutex_); status_ = std::move(message); }
};
} // namespace android_controls
