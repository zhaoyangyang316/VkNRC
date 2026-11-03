#ifndef CONSTANT_GLSL
#define CONSTANT_GLSL

#define M_PI 3.1415926535897932384626433832795
#define NRC_HIDDEN_LAYERS 5
#define NRC_TRAIN_BATCH_COUNT 4
#define NRC_TRAIN_BATCH_SIZE 16384
// Backpropagation and per-subgroup gradient sums use FP16. Reduce their range
// before accumulation; the optimizer divides by this scale again in FP32.
#define LOSS_SCALE (1.0 / 128.0)

// Optimizer Options
#define ADAM_BETA1 0.9
#define ADAM_BETA2 0.999
#define EMA_ALPHA 0.99

#endif
