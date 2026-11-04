#include "../Scene.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {
// A static diagnostic image for emulator drivers without the NRC GPU features.
// Keep the Cornell geometry, emissive materials, camera and visibility test real.
constexpr int kWidth = 160, kHeight = 90;
struct Triangle {
	glm::vec3 a, e1, e2, normal;
	float area;
	const Scene::Material *material;
};

const Triangle *Intersect(const std::vector<Triangle> &scene, glm::vec3 origin, glm::vec3 dir, float &distance) {
	const Triangle *nearest = nullptr;
	for (const auto &t : scene) {
		auto p = glm::cross(dir, t.e2);
		float det = glm::dot(t.e1, p);
		if (std::abs(det) < 1e-7f) continue;
		auto delta = origin - t.a;
		float u = glm::dot(delta, p) / det;
		if (u < 0 || u > 1) continue;
		auto q = glm::cross(delta, t.e1);
		float v = glm::dot(dir, q) / det;
		float d = glm::dot(t.e2, q) / det;
		if (v >= 0 && u + v <= 1 && d > 1e-4f && d < distance) {
			distance = d;
			nearest = &t;
		}
	}
	return nearest;
}

std::vector<uint32_t> RenderPreview(const Scene &scene) {
	std::vector<Triangle> triangles;
	const auto &indices = scene.GetVertexIndices();
	const auto &vertices = scene.GetVertices();
	for (size_t i = 0; i + 2 < indices.size(); i += 3) {
		auto a = vertices[indices[i]], e1 = vertices[indices[i + 1]] - a, e2 = vertices[indices[i + 2]] - a;
		auto n = glm::cross(e1, e2);
		float twice_area = glm::length(n);
		if (twice_area > 0)
			triangles.push_back({a, e1, e2, n / twice_area, 0.5f * twice_area,
			                     &scene.GetMaterials().at(scene.GetMaterialIDs()[i / 3])});
	}
	std::vector<const Triangle *> emitters;
	for (const auto &t : triangles)
		if (glm::any(glm::greaterThan(t.material->emission, glm::vec3(0)))) emitters.push_back(&t);
	const glm::vec3 eye{0, -0.005007153f, 3.86123033f};
	const float tan_half = std::tan(glm::radians(39.30765f) * 0.5f);
	std::vector<uint32_t> pixels(kWidth * kHeight, 0xff000000u);
	for (int y = 0; y < kHeight; ++y) {
		for (int x = 0; x < kWidth; ++x) {
			auto dir = glm::normalize(glm::vec3{(2*x + 1 - kWidth) * tan_half / kHeight,
			                                    (kHeight - 2*y - 1) * tan_half / kHeight, -1});
			float distance = std::numeric_limits<float>::max();
			auto hit = Intersect(triangles, eye, dir, distance);
			if (!hit) continue;
			auto pos = eye + dir * distance;
			auto normal = glm::faceforward(hit->normal, dir, hit->normal);
			auto color = hit->material->emission;
			for (auto light : emitters) {
				if (light == hit) continue;
				// Centroid quadrature, one visibility ray per emitter triangle.
				auto offset = light->a + (light->e1 + light->e2) / 3.f - pos;
				float dist2 = glm::dot(offset, offset);
				auto wi = offset / std::sqrt(dist2);
				float cosine = std::max(glm::dot(normal, wi), 0.f);
				float limit = std::sqrt(dist2) - 2e-4f;
				if (cosine > 0 && !Intersect(triangles, pos, wi, limit))
					color += hit->material->diffuse * light->material->emission *
					         (cosine * std::abs(glm::dot(light->normal, wi)) * light->area / (3.14159265f * dist2));
			}
			auto channel = [](float c) {
				return uint32_t(std::lround(255.f * std::min(1.f, std::pow(std::max(c, 0.f), 1.f / 2.2f))));
			};
			pixels[y*kWidth + x] |= (channel(color.r) << 16) | (channel(color.g) << 8) | channel(color.b);
		}
	}
	return pixels;
}
} // namespace

#ifdef __ANDROID__
#include <jni.h>

extern "C" JNIEXPORT jintArray JNICALL Java_org_vknrc_demo_VkNRCActivity_renderCornellPreview(
    JNIEnv *env, jclass, jstring filename) {
	if (!filename) return nullptr;
	const char *path = env->GetStringUTFChars(filename, nullptr);
	if (!path) return nullptr;
	try {
		auto scene = Scene::LoadOBJSingleInstance(path);
		env->ReleaseStringUTFChars(filename, path);
		path = nullptr;
		if (scene.Empty()) return nullptr;
		auto pixels = RenderPreview(scene);
		auto result = env->NewIntArray(static_cast<jsize>(pixels.size()));
		if (result) env->SetIntArrayRegion(result, 0, static_cast<jsize>(pixels.size()), reinterpret_cast<const jint *>(pixels.data()));
		return result;
	} catch (...) {
		if (path) env->ReleaseStringUTFChars(filename, path);
		return nullptr;
	}
}
#endif
