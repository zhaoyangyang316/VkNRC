#pragma once

#include "RenderCapture.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <imgui.h>
#include <spdlog/fmt/fmt.h>
#include <vector>

namespace capture_detail {
inline std::string_view MethodName(VkNRCState::Method method) {
	switch (method) {
	case VkNRCState::kNone: return "PT";
	case VkNRCState::kNRC: return "NRC";
	case VkNRCState::kCache: return "Cache";
	}
	return {};
}

struct Line {
	std::vector<const ImFontGlyph *> glyphs;
	int width{};
};

inline Line FitLine(const ImFont &font, int limit, std::initializer_list<std::string_view> parts) {
	Line line;
	const auto append = [&](const ImFontGlyph *glyph) {
		const int advance = int(glyph->AdvanceX); // PixelSnapH gives integer advances.
		if (line.width + advance > limit)
			return false;
		line.glyphs.push_back(glyph);
		line.width += advance;
		return true;
	};
	bool fits = true;
	for (auto part : parts) {
		if (part.empty())
			continue;
		// The bundled fmt decoder handles all Unicode planes, including characters
		// beyond the vendored ImGui build's 16-bit glyph index.
		fmt::detail::for_each_codepoint(fmt::string_view(part.data(), part.size()), [&](uint32_t cp, fmt::string_view) {
			const auto *glyph = cp >= 32 && cp <= IM_UNICODE_CODEPOINT_MAX ? font.FindGlyphNoFallback(ImWchar(cp)) : nullptr;
			if (glyph)
				return fits = append(glyph);
			if (cp == fmt::detail::invalid_code_point)
				cp = 0xFFFD;
			for (char c : fmt::format("U+{:04X}", cp))
				if (!append(font.FindGlyph(ImWchar(c))))
					return fits = false;
			return true;
		});
		if (!fits)
			break;
	}
	if (!fits) {
		const auto *dot = font.FindGlyph('.');
		const int dots_width = 3 * int(dot->AdvanceX);
		while (!line.glyphs.empty() && line.width + dots_width > limit) {
			line.width -= int(line.glyphs.back()->AdvanceX);
			line.glyphs.pop_back();
		}
		for (int i = 0; i < 3; ++i)
			append(dot);
	}
	return line;
}

struct Caption {
	int x{}, y{}, width{}, height{};
	std::vector<unsigned char> ink;

	void CompositeRow(uint32_t scan_y, uint32_t stride, float *bgr) const {
		if (width == 0 || scan_y < uint32_t(y) || scan_y >= uint32_t(y + height))
			return;
		const auto *coverage = ink.data() + size_t(scan_y - y) * width;
		for (int i = 0; i < width; ++i) {
			// Opaque plate and antialiased ink, both in linear radiance.
			const float value = 0.025f + (1.0f - 0.025f) * (coverage[i] / 255.0f);
			for (size_t channel = 0; channel < 3; ++channel)
				bgr[channel * stride + x + i] = value;
		}
	}
};

inline bool MakeCaption(Caption &caption, VkExtent3D extent, const CaptureMetadata &metadata) {
	const auto &settings = metadata.settings;
	const auto left = MethodName(settings.left_method), right = MethodName(settings.right_method);
	if (left.empty() || right.empty())
		return false;

	ImFontAtlas atlas;
	ImFontConfig config;
	config.SizePixels = 16;
	config.OversampleH = config.OversampleV = 1;
	config.PixelSnapH = true;
	const auto *font = atlas.AddFontDefault(&config);
	if (!font || !atlas.Build())
		return false;

	const int w = int(extent.width), h = int(extent.height);
	const int margin = std::min({8, w / 12, h / 12}), pad = std::min({5, w / 12, h / 12});
	const int limit = std::min(720, w - 2 * (margin + pad));
	const std::array lines{
	    FitLine(*font, limit, {"Resolution: ", fmt::format("{}x{}", w, h), "  |  NEE: ", settings.nee ? "on" : "off"}),
	    FitLine(*font, limit, {fmt::format("Max bounces: {}  |  SPP: {}", settings.max_bounces, settings.spp)}),
	    FitLine(*font, limit, {"Left: ", left, "  |  Right: ", right}),
	    FitLine(*font, limit, {"Scene: ", settings.scene_name}),
	    FitLine(*font, limit, {settings.fast_mode ? fmt::format("Fast mode: on  |  Min roughness: {}", settings.min_roughness)
	                                              : "Fast mode: off"})};
	for (const auto &line : lines)
		caption.width = std::max(caption.width, line.width + 2 * pad);
	constexpr int line_pitch = 18;
	caption.height = std::min(h - 2 * margin, int(lines.size()) * line_pitch + 2 * pad);
	caption.x = w - margin - caption.width;
	caption.y = h - margin - caption.height;
	caption.ink.assign(size_t(caption.width) * caption.height, 0);

	int line_y = pad;
	for (const auto &line : lines) {
		int pen_x = pad;
		for (const auto *glyph : line.glyphs) {
			const int px = pen_x + int(std::lround(glyph->X0)), py = line_y + int(std::lround(glyph->Y0));
			const int u0 = int(std::lround(glyph->U0 * atlas.TexWidth)), u1 = int(std::lround(glyph->U1 * atlas.TexWidth));
			const int v0 = int(std::lround(glyph->V0 * atlas.TexHeight)), v1 = int(std::lround(glyph->V1 * atlas.TexHeight));
			// Oversampling is disabled, so each atlas texel maps to one output pixel.
			for (int dy = std::max(0, -py); dy < std::min(v1 - v0, caption.height - py); ++dy) {
				for (int dx = std::max(0, -px); dx < std::min(u1 - u0, caption.width - px); ++dx) {
					auto &pixel = caption.ink[size_t(py + dy) * caption.width + px + dx];
					pixel = std::max(pixel, atlas.TexPixelsAlpha8[(v0 + dy) * atlas.TexWidth + u0 + dx]);
				}
			}
			pen_x += int(glyph->AdvanceX);
		}
		line_y += line_pitch;
	}
	return true;
}
} // namespace capture_detail
