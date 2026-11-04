#pragma once

#include <chrono>
#include <cstdint>
#include <optional>

class FrameBenchmark {
	using Clock = std::chrono::steady_clock;

	unsigned m_duration_seconds{};
	uint32_t m_warmup_remaining{};
	uint64_t m_frames{};
	Clock::time_point m_start{};

public:
	struct Result {
		uint64_t frames{};
		double seconds{};
		double MeanMilliseconds() const { return frames ? 1000.0 * seconds / double(frames) : 0.0; }
		double FramesPerSecond() const { return seconds > 0.0 ? double(frames) / seconds : 0.0; }
	};

	void Start(unsigned seconds) {
		Cancel();
		m_duration_seconds = seconds;
		m_warmup_remaining = seconds ? 32u : 0u;
	}
	void Cancel() { *this = FrameBenchmark{}; }
	bool IsActive() const { return m_duration_seconds != 0; }
	bool IsWarmingUp() const { return m_warmup_remaining != 0; }
	uint32_t WarmupRemaining() const { return m_warmup_remaining; }
	unsigned DurationSeconds() const { return m_duration_seconds; }
	double ElapsedSeconds() const {
		return IsActive() && !IsWarmingUp() ? std::chrono::duration<double>(Clock::now() - m_start).count() : 0.0;
	}

	// Call once after submitting a rendered frame. Drain the queue only at the
	// timing boundaries so in-flight warmup work and measured work stay separate.
	template <class Drain> std::optional<Result> FinishRenderedFrame(Drain &&drain) {
		if (!IsActive())
			return std::nullopt;
		if (IsWarmingUp()) {
			if (m_warmup_remaining == 1) {
				drain();
				m_start = Clock::now();
			}
			--m_warmup_remaining;
			return std::nullopt;
		}
		++m_frames;
		if (ElapsedSeconds() < m_duration_seconds)
			return std::nullopt;
		drain();
		const Result result{m_frames, ElapsedSeconds()};
		Cancel();
		return result;
	}
};
