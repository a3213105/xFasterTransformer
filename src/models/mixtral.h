// Copyright (c) 2024-2025 Intel Corporation
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
#pragma once

#include "common_decoder.h"
#include "mlp_mixtral.h"
#include "rms_norm.h"
#include "rotary_embedding.h"
#include "token_embedding.h"
#include "type_selector.h"

template <typename WeiT, typename KVCacheT>
class MixtralLLM
    : public CommonDecoder<Attention<WeiT, LlamaRotaryEmbedding, RmsNorm, typename TypeSelector<WeiT>::InType,
                                   typename TypeSelector<WeiT>::ImType, typename TypeSelector<WeiT>::OutType, true>,
              MixtralMLP<WeiT, typename TypeSelector<WeiT>::InType, typename TypeSelector<WeiT>::ImType,
                      typename TypeSelector<WeiT>::OutType>,
              KVCacheT> {
public:
    MixtralLLM(const std::string &modelPath);
    ~MixtralLLM();

    void prepareAttnMask(int *ids, int step);

    void embeddingForward(int *ids, float *output, int tokenSize);
    void embeddingForward(int *ids, bfloat16_t *output, int tokenSize);
    void embeddingForward(int *ids, float16_t *output, int tokenSize);

    void lastLayerNormForward(float *input, float *output, int rows);
    void lastLayerNormForward(bfloat16_t *input, bfloat16_t *output, int rows);
    void lastLayerNormForward(float16_t *input, float16_t *output, int rows);

private:
    void setEmbeddingWeights(const std::string &modelPath);
    void setFinalLnWeight(const std::string &modelPath);

private:
    TokenEmbedding<float16_t> *embedding;
    RmsNorm finalLN;
};

REGISTER_MODEL(MixtralLLM, mixtral)
