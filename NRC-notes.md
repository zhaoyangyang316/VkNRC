# NRC research notes

## Denoising at one sample per pixel

Müller et al. used the pretrained screen-space denoiser from Hasselgren et al., *Neural Temporal Adaptive Sampling and Denoising* (2020), for the Living Room experiment in Figure 16. At 1 spp, adding NRC to denoised path tracing with ReSTIR reduced relative squared bias from 0.0268 to 0.0105. The paper identifies the denoiser for this experiment but does not say it used Hasselgren’s adaptive sampler. [NRC paper, Figure 16](https://arxiv.org/html/2106.12372v2#S7.F16)

Hasselgren’s full method trains two networks together. Before tracing, the sampler uses the previous denoised frame, reprojected with motion vectors, and the current frame’s geometry guides to allocate a ray budget across pixels. After tracing, the denoiser predicts per-pixel 5×5 filters at three spatial scales and blends the filtered result with reprojected history. The training loss measures both image error and error in frame-to-frame changes. [Authors’ preprint, §2 and Figures 2–3](https://research.nvidia.com/sites/default/files/pubs/2020-05_Neural-Temporal-Adaptive/NTASAD_preprint.pdf#page=2)

## Low roughness, training records, and cache lookups

**Training.** A mirror-like hit can supply a training record. The paper describes propagating radiance from the training path’s endpoint back to preceding vertices. It gives no roughness cutoff for storing those targets. In VkNRC, `TracePath` records the current hit while training is active and its eight-record buffer has space. The BRDF clamps roughness to 0.001, so a zero-roughness material in this implementation uses a narrow glossy lobe. The paper leaves the original tracer’s handling of a mathematically perfect mirror unspecified. [NRC paper, §3.2](https://arxiv.org/html/2106.12372v2#S3.SS2) · [VkNRC shader, `TracePath` and `GetHitBRDFArgs`](shader/src/path_tracer.comp)

**Rendering.** Normal NRC rendering queries the cache when accumulated path spread exceeds a threshold. Each segment’s spread depends on the sampling density at the *previous* surface. A sharp reflection sampled near its peak typically has high density and adds little spread. On a camera → mirror → wall path, the mirror can therefore produce a training record while the renderer traces the reflection and may continue past the wall. Conversely, a broad bounce before hitting a mirror can already provide enough spread to query the cache at the mirror. [NRC paper, §3.4, Eq. 3](https://arxiv.org/html/2106.12372v2#S3.SS4)

**Debug views.** The paper queries the *first non-specular hit* in Figure 12 to visualize the cache and compare it with the footprint heuristic. VkNRC’s **Cache** view instead queries the first eligible hit, including a glossy one. With **Indirect illumination only** enabled, that view traces the first bounce before querying. [NRC paper, Figure 12](https://arxiv.org/html/2106.12372v2#S6.F12) · [VkNRC shader, `METHOD_CACHE` branch](shader/src/path_tracer.comp)
