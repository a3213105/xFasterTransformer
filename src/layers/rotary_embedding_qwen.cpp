// Copyright (c) 2023-2024 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#include "rotary_embedding_qwen.h"

#include "allocator.h"
#include "compile_util.h"

// unordered_map<base, tuple<emb_cos, emb_sin>>
static std::unordered_map<float, std::tuple<float *, float *>> embCosSin;
static float *cur_emb_cos = nullptr;
static float *cur_emb_sin = nullptr;
static float *logn = nullptr;
const int maxSupportedSeqLength = 32768;

bool QwenRotaryEmbedding::initialized = false;
bool QwenRotaryEmbedding::logn_initialized = false;
int QwenRotaryEmbedding::max_seq_len_cached = -1;
int QwenRotaryEmbedding::inv_freq_size = -1;

// dim: equals to head size
QwenRotaryEmbedding::QwenRotaryEmbedding(const int dim, const int max_position_embeddings, const float base) {
    this->dim = dim;
    this->base_initial = base;
    this->base = base;
    if (!initialized) {
        this->max_seq_len_cached = max_position_embeddings;
        this->inv_freq_size = (dim + 1) / 2;
        float *inv_freq = (float *)malloc(this->inv_freq_size * sizeof(float));
#if THREAD_TYPE == THREAD_TYPE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, this->inv_freq_size),
    [&](const tbb::blocked_range<int>& range) {
    for (int i = range.begin(); i < range.end(); ++i) {
#else
#   if THREAD_TYPE == THREAD_TYPE_OMP
#   pragma omp parallel for
#   endif
        for (int i = 0; i < this->inv_freq_size; i++) {
#endif
            inv_freq[i] = 1.0 / pow(base, float(i * 2) / dim);
        }
#if THREAD_TYPE == THREAD_TYPE_TBB
        });
#endif
        QwenCalEmb(inv_freq, base, embCosSin);
        free(inv_freq);

        auto &value = embCosSin[base];
        cur_emb_cos = std::get<0>(value);
        cur_emb_sin = std::get<1>(value);
        initialized = true;
    } else if (dim != this->inv_freq_size * 2) {
        printf("Incorrect dim=%d, inv_freq_size=%d\n", dim, this->inv_freq_size);
        exit(-1);
    }
};

QwenRotaryEmbedding::~QwenRotaryEmbedding() {}

void QwenRotaryEmbedding::init_logn(int max_seq_length, bool use_logn, bool use_ntk_scale) {
    this->use_ntk = use_ntk_scale;
    if (!logn_initialized) {
        logn_initialized = true;
        /*LOGN
        logn_list = [
            math.log(i, self.seq_length) if i > self.seq_length else 1
            for i in range(1, 32768)
        ]
        */
        if (use_logn) {
            REQUIRES(max_seq_length > 0,
                    "seq_length in config.ini is incorrect, please re-conv the model with the latest convert tools");
            if (max_seq_length > maxSupportedSeqLength) {
                printf("QWEN: max_seq_length > maxSupportedSeqLength, we will reduce max_seq_length to %d\n",
                        maxSupportedSeqLength);
                max_seq_length = maxSupportedSeqLength;
            }
        } else {
            max_seq_length = maxSupportedSeqLength;
        }
        logn = (float *)malloc(maxSupportedSeqLength * 2 * sizeof(float));
#if THREAD_TYPE == THREAD_TYPE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, max_seq_length),
    [&](const tbb::blocked_range<int>& range) {
    for (int i = range.begin(); i < range.end(); ++i) {
#else
#   if THREAD_TYPE == THREAD_TYPE_OMP
#   pragma omp parallel for
#   endif
        for (int i = 0; i < max_seq_length; i++) {
#endif
            logn[i] = 1.0;
        }
#if THREAD_TYPE == THREAD_TYPE_TBB
        });
#endif
        float log_base = log(max_seq_length);
#if THREAD_TYPE == THREAD_TYPE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(max_seq_length, maxSupportedSeqLength),
    [&](const tbb::blocked_range<int>& range) {
    for (int i = range.begin(); i < range.end(); ++i) {
#else
#   if THREAD_TYPE == THREAD_TYPE_OMP
#   pragma omp parallel for
#   endif
        for (int i = max_seq_length; i < maxSupportedSeqLength; i++) {
#endif
            logn[i] = log(i + 1) / log_base;
        }
#if THREAD_TYPE == THREAD_TYPE_TBB
        });
#endif
    }
}

float QwenRotaryEmbedding::getNewBaseValue(const int true_seq_len, const int max_seq_length) {
    if (max_seq_length <= 0 || !use_ntk) { return (float)1.0; }

    float context_value = log((float)true_seq_len / (float)max_seq_length) / log(2.0) + 1;
    float ntk_alpha = pow((float)2.0, ceil(context_value)) - 1;
    ntk_alpha = std::max(ntk_alpha, (float)1.0);
    float new_base = this->base_initial * pow(ntk_alpha, (float)this->dim / (this->dim - 2));
    return new_base;
}

void QwenRotaryEmbedding::QwenCalEmb(
        float *inv_freq, float base, std::unordered_map<float, std::tuple<float *, float *>> &embCosSin) {
    float *emb_cos = (float *)xft::alloc(this->max_seq_len_cached * (this->inv_freq_size * 2) * sizeof(float));
    float *emb_sin = (float *)xft::alloc(this->max_seq_len_cached * (this->inv_freq_size * 2) * sizeof(float));

    embCosSin[base] = std::make_tuple(emb_cos, emb_sin);

#if THREAD_TYPE == THREAD_TYPE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, this->max_seq_len_cached),
    [&](const tbb::blocked_range<int>& range) {
    for (int i = range.begin(); i < range.end(); ++i) {
#else
#   if THREAD_TYPE == THREAD_TYPE_OMP
#   pragma omp parallel for
#   endif
    for (int i = 0; i < this->max_seq_len_cached; i++) {
#endif
        float *pcos = emb_cos + i * this->inv_freq_size * 2;
        float *psin = emb_sin + i * this->inv_freq_size * 2;

        for (size_t j = 0; j < this->inv_freq_size; j++) {
            float tmp = i * inv_freq[j];
            float cos_tmp = std::cos(tmp);
            float sin_tmp = std::sin(tmp);

            pcos[j] = cos_tmp;
            pcos[j + this->inv_freq_size] = cos_tmp;
            psin[j] = sin_tmp;
            psin[j + this->inv_freq_size] = sin_tmp;
        }
    }
#if THREAD_TYPE == THREAD_TYPE_TBB
    });
#endif
}

// def rotate_half(x):
//     """Rotates half the hidden dims of the input."""
//     x1 = x[..., : x.shape[-1] // 2]
//     x2 = x[..., x.shape[-1] // 2 :]
//     return torch.cat((-x2, x1), dim=-1)
// def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
//     # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
//     cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
//     sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
//     cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
//     sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
//     q_embed = (q * cos) + (rotate_half(q) * sin)
//     k_embed = (k * cos) + (rotate_half(k) * sin)
//     return q_embed, k_embed
//
// qk_shape: 4 values of [batch_size, seq_len, head_num, head_size]
// position_ids: an array in the size of seq_len
// query and key is the matrix like below:
//
// |<------------------------------ head_size * head_num --------------------------------->|
// |_head_size|___________________________________________________________________________________
// |          |          |          |          |          |          |          |          |    ^
// |          |          |          |          |          |          |          |          |    |
// |          |          |          |          |          |          |          |          | bs*seq_len
// |          |          |          |          |          |          |          |          |    |
// |          |          |          |          |          |          |          |          |    |
// |__________|__________|__________|__________|__________|__________|__________|__________|____v__
void QwenRotaryEmbedding::forward(
        float *query, float *key, int qStride, int kStride, const int *qkShape, const int *positionIds) {
    int dim = this->inv_freq_size * 2;
    REQUIRES(dim == qkShape[3], "Incorrect shape, this dimention is not the head size.");

    const int batchSize = qkShape[0];
    const int seqLen = qkShape[1];
    const int qHeads = qkShape[2];
    const int kHeads = qkShape[4];
    const int maxSeqLength = qkShape[5];
    const int pastKeyLength = qkShape[6];
    const int heads = std::max(qHeads, kHeads);
    const int half = this->inv_freq_size;
    int kv_len = seqLen + pastKeyLength;
    REQUIRES(kv_len < maxSupportedSeqLength, "process seq length must less than 32768.");

    /*** In QWEN torch version, they acturally used seq_len+past_key_len as kv_len to calculate new base
        kv_seq_len = hidden_states.size()[1]
        if past_key_values[0] is not None:
            # past key values[0][0] shape: bs * seq_len * head_num * dim
            if self.use_cache_quantization:
                kv_seq_len += past_key_values[0][0][0].shape[2]
            else:
                kv_seq_len += past_key_values[0][0].shape[1]

    ***/
    float new_base = getNewBaseValue(kv_len, maxSeqLength);
    if (std::abs(new_base - this->base) > 1e-5) {
        this->base = new_base;
        auto it = embCosSin.find(new_base);
        if (it == embCosSin.end()) {
            float *inv_freq = (float *)malloc(this->inv_freq_size * sizeof(float));
#if THREAD_TYPE == THREAD_TYPE_TBB
            tbb::parallel_for(tbb::blocked_range<int>(0, this->inv_freq_size),
            [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i < range.end(); ++i) {
#else
#   if THREAD_TYPE == THREAD_TYPE_OMP
#   pragma omp parallel for
#   endif
            for (int i = 0; i < this->inv_freq_size; i++) {
#endif
                inv_freq[i] = 1.0 / pow(new_base, float(i * 2) / dim);
            }
#if THREAD_TYPE == THREAD_TYPE_TBB
            });
#endif
            QwenCalEmb(inv_freq, new_base, embCosSin);
            free(inv_freq);
        }

        auto &value = embCosSin[new_base];
        cur_emb_cos = std::get<0>(value);
        cur_emb_sin = std::get<1>(value);
    }

    /*** LOGN in Torch
        if key_size > self.seq_length and self.use_logn_attn and not self.training:
            if self.use_cache_quantization:
                seq_start = key[0].size(2) - query.size(1)
                seq_end = key[0].size(2)
            else:
                seq_start = key.size(1) - query.size(1)
                seq_end = key.size(1)
            logn_tensor = self.logn_tensor[:, seq_start:seq_end, :, :].type_as(query)
            query = query * logn_tensor.expand_as(query)
    ***/
    float *q_scale = logn + pastKeyLength;

#if THREAD_TYPE == THREAD_TYPE_TBB
    tbb::parallel_for(tbb::blocked_range3d<int, int, int>(0, heads, 0, batchSize, 0, seqLen),
    [&](const tbb::blocked_range3d<int, int, int>& range) {
    for (int head = range.pages().begin(); head < range.pages().end(); ++head) {
        for (int bs = range.rows().begin(); bs < range.rows().end(); ++bs) {
            for (int seq = range.cols().begin(); seq < range.cols().end(); ++seq) {
#else
#   if THREAD_TYPE == THREAD_TYPE_OMP
#   pragma omp parallel for collapse(3)
#   endif
    for (int head = 0; head < heads; ++head) {
        for (int bs = 0; bs < batchSize; ++bs) {
            for (int seq = 0; seq < seqLen; ++seq) {
#endif
                int pos = positionIds[seq];
                float *pcos = cur_emb_cos + pos * dim;
                float *psin = cur_emb_sin + pos * dim;

                float *q = query + bs * seqLen * qStride + seq * qStride + head * dim;
                float *k = key + bs * seqLen * kStride + seq * kStride + head * dim;
#if THREAD_TYPE == THREAD_TYPE_OMP
#   pragma omp simd
#endif
                for (int i = 0; i < half; ++i) {
                    if (head < qHeads) {
                        auto q1 = q[i];
                        q[i] = (q[i] * pcos[i] - q[i + half] * psin[i]) * q_scale[seq];
                        q[i + half] = (q[i + half] * pcos[i + half] + q1 * psin[i + half]) * q_scale[seq];
                    }
                    if (head < kHeads) {
                        auto k1 = k[i];
                        k[i] = k[i] * pcos[i] - k[i + half] * psin[i];
                        k[i + half] = k[i + half] * pcos[i + half] + k1 * psin[i + half];
                    }
                }
            }
        }
    }
#if THREAD_TYPE == THREAD_TYPE_TBB
    });
#endif
}

void QwenRotaryEmbedding::forward(
        bfloat16_t *query, bfloat16_t *key, int qStride, int kStride, const int *qkShape, const int *positionIds) {
    int dim = this->inv_freq_size * 2;
    REQUIRES(dim == qkShape[3], "Incorrect shape, this dimention is not the head size.");

    const int batchSize = qkShape[0];
    const int seqLen = qkShape[1];
    const int qHeads = qkShape[2];
    const int kHeads = qkShape[4];
    const int maxSeqLength = qkShape[5];
    const int pastKeyLength = qkShape[6];
    const int heads = std::max(qHeads, kHeads);
    const int half = this->inv_freq_size;
    int kv_len = seqLen + pastKeyLength;
    REQUIRES(kv_len < maxSupportedSeqLength, "process seq length must less than 32768.");

    float new_base = getNewBaseValue(kv_len, maxSeqLength);
    if (std::abs(new_base - this->base) > 1e-5) {
        this->base = new_base;

        auto it = embCosSin.find(new_base);
        if (it == embCosSin.end()) {
            float *inv_freq = (float *)malloc(this->inv_freq_size * sizeof(float));
#if THREAD_TYPE == THREAD_TYPE_TBB
            tbb::parallel_for(tbb::blocked_range<int>(0, this->inv_freq_size),
            [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i < range.end(); ++i) {
#else
#   if THREAD_TYPE == THREAD_TYPE_OMP
#   pragma omp parallel for
#   endif
            for (int i = 0; i < this->inv_freq_size; i++) {
#endif
                inv_freq[i] = 1.0 / pow(base, float(i * 2) / dim);
            }
#if THREAD_TYPE == THREAD_TYPE_TBB
            });
#endif
            QwenCalEmb(inv_freq, new_base, embCosSin);
            free(inv_freq);
        }

        auto &value = embCosSin[new_base];
        cur_emb_cos = std::get<0>(value);
        cur_emb_sin = std::get<1>(value);
    }

    xft::qwenApplyRotaryPosEmbeding(query, key, qStride, kStride, cur_emb_cos, cur_emb_sin, inv_freq_size, logn,
            maxSupportedSeqLength, qkShape, positionIds);
}

void QwenRotaryEmbedding::forward(
        float16_t *query, float16_t *key, int qStride, int kStride, const int *qkShape, const int *positionIds) {
    int dim = this->inv_freq_size * 2;
    REQUIRES(dim == qkShape[3], "Incorrect shape, this dimention is not the head size.");

    const int seqLen = qkShape[1];
    const int maxSeqLength = qkShape[5];
    const int pastKeyLength = qkShape[6];
    int kv_len = seqLen + pastKeyLength;
    REQUIRES(kv_len < maxSupportedSeqLength, "process seq length must less than 32768.");

    float new_base = getNewBaseValue(kv_len, maxSeqLength);
    if (std::abs(new_base - this->base) > 1e-5) {
        this->base = new_base;

        auto it = embCosSin.find(new_base);
        if (it == embCosSin.end()) {
            float *inv_freq = (float *)malloc(this->inv_freq_size * sizeof(float));
#if THREAD_TYPE == THREAD_TYPE_TBB
            tbb::parallel_for(tbb::blocked_range<int>(0, this->inv_freq_size),
            [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i < range.end(); ++i) {
#else
#   if THREAD_TYPE == THREAD_TYPE_OMP
#   pragma omp parallel for
#   endif
            for (int i = 0; i < this->inv_freq_size; i++) {
#endif
                inv_freq[i] = 1.0 / pow(base, float(i * 2) / dim);
            }
#if THREAD_TYPE == THREAD_TYPE_TBB
            });
#endif
            QwenCalEmb(inv_freq, new_base, embCosSin);
            free(inv_freq);
        }

        auto &value = embCosSin[new_base];
        cur_emb_cos = std::get<0>(value);
        cur_emb_sin = std::get<1>(value);
    }

    xft::qwenApplyRotaryPosEmbeding(query, key, qStride, kStride, cur_emb_cos, cur_emb_sin, inv_freq_size, logn,
            maxSupportedSeqLength, qkShape, positionIds);
}

// For continuous batching
void QwenRotaryEmbedding::forward(
        float *query, float *key, int totSeqLen, int qStride, int kStride, int qHeads, int kHeads, int *positionIds) {
    xft::qwenApplyRotaryPosEmbed(query, key, cur_emb_cos, cur_emb_sin, qStride, kStride, this->dim, logn,
            maxSupportedSeqLength, totSeqLen, qHeads, kHeads, positionIds);
}

void QwenRotaryEmbedding::forward(bfloat16_t *query, bfloat16_t *key, int totSeqLen, int qStride, int kStride,
        int qHeads, int kHeads, int *positionIds) {
    xft::qwenApplyRotaryPosEmbed(query, key, cur_emb_cos, cur_emb_sin, qStride, kStride, this->dim, logn,
            maxSupportedSeqLength, totSeqLen, qHeads, kHeads, positionIds);
}

void QwenRotaryEmbedding::forward(float16_t *query, float16_t *key, int totSeqLen, int qStride, int kStride, int qHeads,
        int kHeads, int *positionIds) {
    xft::qwenApplyRotaryPosEmbed(query, key, cur_emb_cos, cur_emb_sin, qStride, kStride, this->dim, logn,
            maxSupportedSeqLength, totSeqLen, qHeads, kHeads, positionIds);
}
