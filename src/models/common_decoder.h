// Copyright (c) 2023 Intel Corporation
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

#include <string>
#include <tuple>
#include <vector>

#include "INIReader.h"
#include "abstract_decoder.h"
#include "attention.h"
#include "datatypes.h"
#include "debugger.h"
#include "decoder_block.h"
#include "decoder_layer.h"
#include "dist_linear.h"
#include "dtype.h"
#include "kvcache_manager.h"
#include "messenger.h"
#include "mlp_chatglm2.h"
#include "mlp_standard.h"
#include "model_factory.h"
#include "sequence.h"
#include "timeline.h"
#include "transformer_ctx.h"
#include "transpose_util.h"
#include "weight_util.h"

using namespace xft;

struct QKPO_Dummy {
    QKPO_Dummy(int dim, int maxPos) {}
    void forward(float *query, float *key, int qStride, int kStride, const int *qk_shape, const int *position_ids) {}
    void forward(float *query, float *key, int totSeqLen, int qStride, int kStride, int qHeads, int kHeads,
            int *positionIds) {};
};

// To get data types in MLP class
template <typename T>
struct MlpTypeExtractor;
template <template <typename...> class MLP_CLS, typename WeiT, typename InT, typename ImT, typename OutT>
struct MlpTypeExtractor<MLP_CLS<WeiT, InT, ImT, OutT>> {
    using Tin = InT;
    using Tim = ImT;
    using Tout = OutT;
};
template <typename WeiT, typename InT, typename ImT, typename OutT>
struct MlpTypeExtractor<MLP<WeiT, InT, ImT, OutT, true>> {
    using Tin = InT;
    using Tim = ImT;
    using Tout = OutT;
};
template <typename WeiT, typename InT, typename ImT, typename OutT, typename NORM_CLS>
struct MlpTypeExtractor<ChatGLM2MLP<WeiT, InT, ImT, OutT, NORM_CLS, true>> {
    using Tin = InT;
    using Tim = ImT;
    using Tout = OutT;
};

/*
Pipeline parallel and tensor parallel introduction:

  1) MPI_Instances = 16,XFT_PIPELINE_STAGE = 4  =>  ctx->ppSize = 4, ctx->tpSize = 4
  2) TP sync by oneCCL(row_comm) or shared_memory
  3) PP sync by MPI MPI_COMM_WORLD

  World Rank:      => Row Rank:       => Rank:  tp0 tp1 tp2 tp3
  [ 0,  1,  2,  3,    [ 0, 1, 2, 3];      pp0 [  0,  1,  2,  3];
    4,  5,  6,  7,    [ 0, 1, 2, 3];      pp1 [  0,  1,  2,  3];
    8,  9, 10, 11,    [ 0, 1, 2, 3];      pp2 [  0,  1,  2,  3];
   12, 13, 14, 15];   [ 0, 1, 2, 3];      pp3 [  0,  1,  2,  3];

                                      Prompts
                                         │
            ┌──────────────────┬─────────┴────────┬──────────────────┐
            │                  │                  │                  │
            ▼                  ▼                  ▼                  ▼
       Embedding(PP0)     Embedding(PP0)     Embedding(PP0)     Embedding(PP0)
            │                  │                  │                  │
  PP0       │                  │                  │                  │
  ┌─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────┐
  │ TP0     │          TP1     │          TP2     │          TP3     │    layer0-7  │
  │ ┌───────▼────────┐ ┌───────▼────────┐ ┌───────▼────────┐ ┌───────▼────────┐     │
  │ │ OMP            │ │ OMP            │ │ OMP            │ │ OMP            │     │
  │ │ │ │ │ │ │ │    │ │ │ │ │ │ │ │    │ │ │ │ │ │ │ │    │ │ │ │ │ │ │ │    │     │
  │ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│     │
  │ └───────┬────────┘ └───────┬────────┘ └───────┬────────┘ └───────┬────────┘     │
  │ ┌───────┼──────────────────┼─────AllReduce────┼──────────────────┼────────┐     │
  │ └───────┼──────────────────┼──────────────────┼──────────────────┼────────┘     │
  └─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────┘
  PP1       │ MPI Send/Recv    │                  │                  │
  ┌─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────┐
  │ TP0     │          TP1     │           TP2    │            TP3   │   layer8-15  │
  │ ┌───────▼────────┐ ┌───────▼────────┐ ┌───────▼────────┐ ┌───────▼────────┐     │
  │ │ OMP            │ │ OMP            │ │ OMP            │ │ OMP            │     │
  │ │ │ │ │ │ │ │    │ │ │ │ │ │ │ │    │ │ │ │ │ │ │ │    │ │ │ │ │ │ │ │    │     │
  │ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│     │
  │ └───────┬────────┘ └───────┬────────┘ └───────┬────────┘ └───────┬────────┘     │
  │ ┌───────┼──────────────────┼─────AllReduce────┼──────────────────┼────────┐     │
  │ └───────┼──────────────────┼──────────────────┼──────────────────┼────────┘     │
  └─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────┘
  PP2       │ MPI Send/Recv    │                  │                  │
  ┌─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────┐
  │ TP0     │          TP1     │           TP2    │            TP3   │  layer16-23  │
  │ ┌───────▼────────┐ ┌───────▼────────┐ ┌───────▼────────┐ ┌───────▼────────┐     │
  │ │ OMP            │ │ OMP            │ │ OMP            │ │ OMP            │     │
  │ │ │ │ │ │ │ │    │ │ │ │ │ │ │ │    │ │ │ │ │ │ │ │    │ │ │ │ │ │ │ │    │     │
  │ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│     │
  │ └───────┬────────┘ └───────┬────────┘ └───────┬────────┘ └───────┬────────┘     │
  │ ┌───────┼──────────────────┼─────AllReduce────┼──────────────────┼────────┐     │
  │ └───────┼──────────────────┼──────────────────┼──────────────────┼────────┘     │
  └─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────┘
  PP3       │ MPI Send/Recv    │                  │                  │
  ┌─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────┐
  │ TP0     │          TP1     │           TP2    │            TP3   │  layer24-31  │
  │ ┌───────▼────────┐ ┌───────▼────────┐ ┌───────▼────────┐ ┌───────▼────────┐     │
  │ │ OMP            │ │ OMP            │ │ OMP            │ │ OMP            │     │
  │ │ │ │ │ │ │ │    │ │ │ │ │ │ │ │    │ │ │ │ │ │ │ │    │ │ │ │ │ │ │ │    │     │
  │ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│ │ ▼ ▼ ▼ ▼ ▼ ▼ ...│     │
  │ └───────┬────────┘ └───────┬────────┘ └───────┬────────┘ └───────┬────────┘     │
  │ ┌───────┼──────────────────┼─────AllReduce────┼──────────────────┼────────┐     │
  │ └───────┼──────────────────┼──────────────────┼──────────────────┼────────┘     │
  └─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────┘
            │                  │                  │                  │
            ▼                  ▼                  ▼                  ▼
       Predictor(PP3)     Predictor(PP3)     Predictor(PP3)     Predictor(PP3)
            │ MPI Send/Recv    │                  │                  │
            ▼                  ▼                  ▼                  ▼
       Searchers(PP0)     Searchers(PP0)     Searchers(PP0)     Searchers(PP0)
            │
            ▼
         Output
*/

// Template parameters:
// ATTN_CLS - class for attention impl.
// MLP_CLS - MLP implementation
// KVCacheT - data type of the cached keys/values
// ATTN_MLP_PARALLEL - true means attention and MLP are in parallel, using the same initial input
template <typename ATTN_CLS, typename MLP_CLS, typename KVCacheT = float16_t, bool ATTN_MLP_PARALLEL = false>
class CommonDecoder : public AbstractDecoder {
public:
    CommonDecoder(const std::string &modelPath, const std::string &modelType)
        : messenger(Messenger::getInstance())
#ifdef XFT_DEBUG
        , dbg("model_decoder.csv")
#endif
    {
        // Make sure Attention output can be feed to MLP
        static_assert(std::is_same_v<AttnOutT, MlpInT>, "Error: Attention Output and MLP Input are not the same type.");

        // Make sure MLP output can be feed to Attention
        static_assert(std::is_same_v<MlpOutT, AttnInT>, "Error: MLP Output and Attention Input are not the same type.");

        std::string configPath = modelPath + "/config.ini";
        INIReader reader = INIReader(configPath);
        const int attHeadNum = reader.GetInteger(modelType, "head_num");
        // Use the same head number for the default multi-head attention
        const int kvHeadNum = reader.GetInteger(modelType, "kv_head_num", attHeadNum);
        const int headSize = reader.GetInteger(modelType, "size_per_head");
        const int imSize = reader.GetInteger(modelType, "inter_size");
        const int layers = reader.GetInteger(modelType, "num_layer");
        const int vocabSize = reader.GetInteger(modelType, "vocab_size");
        // Max Position Embedding for position embedding functions, with a default value set to 0
        const int maxPosEmbed = reader.GetInteger(modelType, "max_pos_seq_len", 0);
        // Max num of tokens that LLM can process. Also for allocating buffers. Default maxPosEmbed
        const int maxPositions = reader.GetInteger(modelType, "model_max_length", maxPosEmbed);
        // Seq length in Qwen model, if none, please ignore
        const int maxSeqLength = reader.GetInteger(modelType, "seq_length", -1);
        const bool useLogN = reader.GetInteger(modelType, "use_logn_attn", true);
        const bool useNTK = reader.GetInteger(modelType, "use_dynamic_ntk", true);
        const int hiddenSize = reader.GetInteger(modelType, "hidden_size", attHeadNum * headSize);
        const int embeddingSize = hiddenSize;
        const int multi_query_group_num = reader.GetInteger(modelType, "multi_query_group_num", attHeadNum);
        const float epsilon = reader.GetFloat(modelType, "layernorm_eps", 1e-6);
        const std::string ropeType = reader.Get(modelType, "rope_scaling_type", "");
        const float ropeFactor = reader.GetFloat(modelType, "rope_scaling_factor", 1.0);
        const int ropeOrgMaxPosEmbed
                = reader.GetInteger(modelType, "rope_scaling_original_max_position_embeddings", 2048);
        const float ropeTheta = reader.GetFloat(modelType, "rope_theta", 10000.0);
        const float vextraPolFactor = 1;
        const float vattnFactor = 1;
        const float vbetaFast = reader.GetInteger(modelType, "rope_scaling_beta_fast", 32);
        const float vbetaSlow = reader.GetInteger(modelType, "rope_scaling_beta_slow", 1);
        const float vmscale = reader.GetFloat(modelType, "rope_scaling_mscale", 1.0);
        const float vmscaleAllDim = reader.GetFloat(modelType, "rope_scaling_mscale_all_dim", 1.0);
        RopeParams *ropeParamsPtr = new RopeParams(ropeTheta, ropeType, ropeFactor, ropeOrgMaxPosEmbed,
            vextraPolFactor, vattnFactor, vbetaFast, vbetaSlow, vmscale, vmscaleAllDim);

        std::string act = reader.Get(modelType, "activation_type");
        std::transform(act.begin(), act.end(), act.begin(), ::tolower);

        this->startId = reader.GetInteger(modelType, "start_id", 0);
        this->endId = reader.GetInteger(modelType, "end_id", startId);

        this->initSeqLen = 0;
        this->accSeqLen = 0;

        this->prefixSeqLen = 0;
        this->prefixSharing = false;

        // Quantization config
        const std::string quantQweightDataType = reader.Get(modelType, "quant_qweight_data_type", "");
        const std::string quantScalesDataType = reader.Get(modelType, "quant_scales_data_type", "");
        const std::string quantZerosDataType = reader.Get(modelType, "quant_zeros_data_type", "");
        const int quantGroupsize = reader.GetInteger(modelType, "quant_groupsize", -1);

        DataType srcWeightType = getWeightType(configPath, modelType);
        DataType attnWeightType = ATTN_CLS::getWeightDataType();

        DataType dt = DataType::fp32;
        if (attnWeightType == srcWeightType && 
            (attnWeightType == DataType::bf16 || attnWeightType == DataType::fp16 || attnWeightType == DataType::fp8_e4m3)) {
            dt = srcWeightType;
        }

        if (quantQweightDataType == "int8" || quantQweightDataType == "uint4") {
            dt = quantQweightDataType == "int8" ? DataType::int8 : DataType::int4;
            REQUIRES(quantScalesDataType == "fp32", "scales should be fp32 data type.");
            REQUIRES(quantZerosDataType == "fp32", "zeros should be fp32 data type.");
            REQUIRES(quantGroupsize == -1, "Quantization with groupsize is not supported.");
        }

        // Buffer related (not initialized)
        this->inputTokens = nullptr;
        this->maskSize = 0;
        this->attnMask = nullptr;
        actBuffers.reset(new xft::Matrix<float>());

        // Context
        DecoderContext *ctx = getDecoderContext(layers, hiddenSize, headSize, attHeadNum, kvHeadNum, imSize, act,
                epsilon, vocabSize, embeddingSize, maxPositions, maxPosEmbed, maxSeqLength, useLogN, useNTK,
                ropeParamsPtr);

        ctx->ResetConfigReader(configPath);
        // For MoE
        ctx->sparseExperts = reader.GetInteger(modelType, "sparse_experts", 8);
        ctx->denseExperts = reader.GetInteger(modelType, "dense_experts", 0);

        // For MLA
        ctx->qLoraRank = reader.GetInteger(modelType, "q_lora_rank", 0);
        ctx->kvLoraRank = reader.GetInteger(modelType, "kv_lora_rank", 0);
        ctx->nopeDim = reader.GetInteger(modelType, "qk_nope_head_dim", 0);
        ctx->ropeDim = reader.GetInteger(modelType, "qk_rope_head_dim", 0);

        // For DeepSeek MoE
        ctx->normTopKProb = reader.GetBoolean(modelType, "norm_topk_prob", false);
        ctx->firstKDenseReplace = reader.GetInteger(modelType, "first_k_dense_replace", 0);
        ctx->numExpertsPerTok = reader.GetInteger(modelType, "num_experts_per_tok", 0);
        ctx->topkGroup = reader.GetInteger(modelType, "topk_group", 0);
        ctx->nGroup = reader.GetInteger(modelType, "n_group", 0);
        ctx->moeIntermediateSize = reader.GetInteger(modelType, "moe_intermediate_size", 0);
        ctx->topkMethod = reader.Get(modelType, "topk_method", "");
        ctx->scoringFunc = reader.Get(modelType, "scoring_func", "");
        ctx->routedScalingFac = reader.GetFloat(modelType, "routed_scaling_factor", 1.0);

        // For Qwen3
        ctx->doQKNorm = reader.GetBoolean(modelType, "do_qk_norm", false);

        if (ctx->nopeDim && ctx->ropeDim) { // scale in MLA is different
            float mscale = 0.1 * std::log(40) + 1.0;
            ctx->attFactor = 1 / sqrtf(ctx->nopeDim + ctx->ropeDim) * mscale * mscale;
        } else if (attHeadNum != 0) {
            ctx->attFactor = 1 / sqrtf(ctx->attHeadSize);
        }

        // Decoder
        if (layers % ctx->ppSize != 0) {
            std::cerr << "Warning: layers cannot be evenly divided by pipeline parallel stage size(ppSize)."
                      << std::endl;
            std::exit(-1);
        }

        decoderBlock = new DecoderBlock<ATTN_CLS, MLP_CLS, KVCacheT, ATTN_MLP_PARALLEL>(ctx, modelPath, layers, dt);
        auto maxSeqLen = maxSeqLength > 0 ? maxSeqLength : maxPositions;
        if (ctx->kvLoraRank != 0) {
            // For MLA, cached key dimension is ropeDim, cached value dimension is kvLoraRank
            KVCacheMgr::instance().configure(
                    maxSeqLen, 1, ctx->ropeDim, 1, ctx->kvLoraRank, layers, getDataType<KVCacheT>());
        } else {
            KVCacheMgr::instance().configure(
                    maxSeqLen, kvHeadNum, headSize, kvHeadNum, headSize, layers, getDataType<KVCacheT>());
        }

        // Predictor
        int workers = messenger.getSize();
        int rank = messenger.getRank();
        this->predictor = new DistLinear<LinearWeiT>(hiddenSize, vocabSize, rank, workers);
        this->setPredictorWeight(ctx, modelPath);

        // KVCache Manager
        this->kvCacheMgr.reset(new KVCacheManager<KVCacheT>(layers));
    }

    virtual ~CommonDecoder() {
        if (this->inputTokens) free(this->inputTokens);
        if (this->attnMask) xft::dealloc(this->attnMask);

        delete this->decoderBlock;
        delete this->predictor;
    }

    std::tuple<float *, int, int> forward(int *ids, int64_t *dims, int step, bool logitsAll = false) {
        // Assume input has been synced with master in higher level.
        // Assume the 1st step input's shape is [userSideBS][1][seqLen].
        TimeLine t("Decoder.forward");
        TimeLine t1("Decoder.embedding");

        int userSideBS = dims[0];
        int beamSize = dims[1];
        int batchSize = (step == 0 ? userSideBS : userSideBS * beamSize); // as sequence are duplicated at step 0
        int seqLen = dims[2];
        int pastSeqLen = step == 0 ? 0 : this->accSeqLen;
        int inputSeqLen = seqLen;

        // Prepare context
        DecoderContext *ctx = this->getContext();
        ctx->resize(batchSize, seqLen, pastSeqLen);
        int hiddenSize = ctx->hiddenSize;

        if (step == 0) {
            // Reset initial and accumulated sequence length at the first step
            this->initSeqLen = seqLen;
            this->accSeqLen = 0;
            if (this->prefixSharing) {
                pastSeqLen = this->prefixSeqLen;
                inputSeqLen = seqLen - pastSeqLen;

                int *prefixIDs = (int *)malloc(userSideBS * pastSeqLen * sizeof(int));
                int *newIDs = (int *)malloc(userSideBS * inputSeqLen * sizeof(int));
                for (int bs = 0; bs < userSideBS; bs++) {
                    memcpy(prefixIDs + pastSeqLen * bs, ids + seqLen * bs, pastSeqLen * sizeof(int));
                    memcpy(newIDs + inputSeqLen * bs, ids + seqLen * bs + pastSeqLen, inputSeqLen * sizeof(int));
                }

                this->getPositionIds(prefixIDs, batchSize, pastSeqLen, 0);

                free(prefixIDs);
                ids = newIDs;
                ctx->resize(batchSize, inputSeqLen, pastSeqLen);
            }

            // Enlarge buffer if needed
            prepareBuffers(ctx, userSideBS, beamSize, logitsAll);
        }

        AttnInT *embBuf = (AttnInT *)actBuffers->Data();
        MlpOutT *outBuf = (MlpOutT *)(embBuf + batchSize * inputSeqLen * hiddenSize);

        // Embedding
        this->embeddingForward(ids, embBuf, batchSize * inputSeqLen);
        this->accSeqLen += seqLen;

#ifdef XFT_DEBUG
        dbg.debugPrint("---- embedding.forward ----\n");
        dbg.debugPrint("ids:\n");
        dbg.dumpMatrix(ids, batchSize, inputSeqLen, inputSeqLen);
        dbg.debugPrint("embBuf(rows: %d, cols: %d, stride: %d):\n", batchSize * inputSeqLen, hiddenSize, hiddenSize);
        dbg.dumpMatrix(embBuf, batchSize * inputSeqLen, hiddenSize, hiddenSize);
#endif

        // Prepare attention mask
        this->prepareAttnMask(ids, step + this->prefixSharing);
        // prepareAttnMeta

        // Token position ids, note: different models may have different impl.
        int *positionIds = this->getPositionIds(ids, batchSize, inputSeqLen, step + this->prefixSharing);
        t1.release();

#ifdef PIPELINE_PARALLEL
        // if current pipeline parallel stage rank isn't the first stage, should receive previous stage data
        if (ctx->ppSize > 1 && ctx->ppRank > 0) {
            int curr_world_rank = ctx->ppRank * ctx->tpSize + ctx->tpRank;
            int prev_world_rank = (ctx->ppRank - 1) * ctx->tpSize + ctx->tpRank;
            int count = batchSize * inputSeqLen * hiddenSize;
            int32_t sequenceID;
            MPI_Recv(&sequenceID, 1, MPI_INT32_T, prev_world_rank, curr_world_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            TimeLine t("Decoder.Seq" + std::to_string(sequenceID) + ".MPI_Recv");
            MPI_Recv(embBuf, count, MPI_FLOAT, prev_world_rank, curr_world_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // TODO: Error: different scope when dynamic loading so file
            // this->messenger.worldRecvFP32(embBuf, count, prev_world_rank, curr_world_rank);
            if (!SequencePool::getInstance().has(sequenceID)) {
                auto *groupMeta = SequencePool::getInstance().newGroupMeta(sequenceID, seqLen);
                groupMeta->get(0)->setPastSeqLen(pastSeqLen);
                groupMeta->get(0)->allocBuffer<AttnInT>(hiddenSize, embBuf);
                SequencePool::getInstance().add(groupMeta);
            }
            TaskWaitingQueue::getInstance().push(SequencePool::getInstance().get(sequenceID));
        }

        if (!InputQueue::getInstance().empty()) {
            if (!TaskWaitingQueue::getInstance().isFull()) {
                auto *groupMeta = InputQueue::getInstance().pop();
                groupMeta->get(0)->setPastSeqLen(pastSeqLen);
                groupMeta->get(0)->allocBuffer<AttnInT>(hiddenSize, embBuf);
                SequencePool::getInstance().add(groupMeta);
                TaskWaitingQueue::getInstance().push(
                        SequencePool::getInstance().get(groupMeta->get(0)->getSequenceID()));
            }
        }

        while (TaskWaitingQueue::getInstance().empty());

        SequenceGroupMeta *runningTask = nullptr;
        int32_t sequenceID = -1;
        if (!TaskWaitingQueue::getInstance().empty()) {
            runningTask = TaskWaitingQueue::getInstance().pop();
            sequenceID = runningTask->get(0)->getSequenceID();
            TimeLine t("Decoder.Seq" + std::to_string(sequenceID) + ".Step");
#endif

#ifdef XFT_GPU
        TimeLine tmcpyc2g("Decoder.memcopyCPU2GPU");
        size_t embBufSize = batchSize * inputSeqLen * hiddenSize * sizeof(AttnInT);
        AttnInT *embBufGPU = (AttnInT *)ctx->getBuffer<AttnInT>("embBufGPU", embBufSize, ctx->device);
        AttnInT *outBufGPU = (AttnInT *)ctx->getBuffer<AttnInT>(
                "outBufGPU", actBuffers->Rows() * actBuffers->Cols() * sizeof(float) - embBufSize, ctx->device);
        xft::memcopy(embBufGPU, embBuf, embBufSize, ctx->device);
        embBuf = embBufGPU;
        outBuf = outBufGPU;
        tmcpyc2g.release();
#endif

        // Decoder: forward
        int layers_per_pp_stage = decoderBlock->size();
        for (int i = 0; i < layers_per_pp_stage; ++i) {
            int workers = this->messenger.getSize();
            if (step == 0 && this->prefixSharing) {
                // Expand the prefix KV cache for each batch
                this->kvCacheMgr->expandPrefixCache(i, userSideBS, this->prefixSeqLen);
            }
            KVCacheTensor<KVCacheT> &presentKey = this->kvCacheMgr->getKey(i);
            KVCacheTensor<KVCacheT> &presentValue = this->kvCacheMgr->getValue(i);

            // Pls be noted: in attention, 'outBuf' is used as imtermediate buffer, 'tmpBuf' is used as output
            AttnOutT *attnOut = (AttnOutT *)(this->getContext()->tmpBuf.Data());
            // attnMeta (inputSeqLens, pastSeqLens, seqStartLoc, is_prompt(useSelfAttn), causal, attnMask)
            decoderBlock->get(i)->forwardAttention(getContext(), embBuf, outBuf, attnOut, attnMask,
                    presentKey, // presentKey,
                    presentValue, // presentValue,
                    inputSeqLen, // inputSeqLen,
                    pastSeqLen, // pastSeqLen
                    step == 0, // useSelfAttn,
                    true, // doLnBefore,
                    positionIds);

            // Expand the KV cache as it only has values for beam 0
            if (step == 0 && beamSize > 1) { this->kvCacheMgr->expandCache(i, userSideBS, beamSize, seqLen); }

            // Merge the result of attention
            // When attention and FFN/MLP are in parallel, do not need to reduce after attention
            if constexpr (!ATTN_MLP_PARALLEL) {
                if (this->messenger.getSize() > 1) {
                    this->messenger.reduceAdd(attnOut, attnOut, batchSize * inputSeqLen * hiddenSize);
                }
            }

            // When attention and FFN/MLP are in parallel, use the initial embedding as input
            if constexpr (ATTN_MLP_PARALLEL) {
                if (this->messenger.getSize() > 1) {
                    decoderBlock->get(i)->forwardFFN(getContext(), embBuf, outBuf, hiddenSize, hiddenSize, true);
                    this->messenger.reduceAdd(outBuf, embBuf, batchSize * inputSeqLen * hiddenSize);
                } else {
                    decoderBlock->get(i)->forwardFFN(getContext(), embBuf, embBuf, hiddenSize, hiddenSize, true);
                }
            } else {
                // FFN (for multiple workers, output into outBuf and then reduce add to embBuf)
                if (this->messenger.getSize() > 1) {
                    decoderBlock->get(i)->forwardFFN(getContext(), attnOut, outBuf, hiddenSize, hiddenSize, true);
                    this->messenger.reduceAdd(outBuf, embBuf, batchSize * inputSeqLen * hiddenSize);
                } else {
                    decoderBlock->get(i)->forwardFFN(getContext(), attnOut, embBuf, hiddenSize, hiddenSize, true);
                }
            }
        }

#ifdef PIPELINE_PARALLEL
        }

        // If current pipeline stage isn't the end of stage, should send data to next stage and return nullptr
        if (ctx->ppSize > 1 && ctx->ppRank < ctx->ppSize - 1) {
            TimeLine t("Decoder.Seq" + std::to_string(sequenceID) + ".MPI_Send");
            int next_world_rank = (ctx->ppRank + 1) * ctx->tpSize + ctx->tpRank;
            int count = batchSize * inputSeqLen * hiddenSize;
            MPI_Send(&sequenceID, 1, MPI_INT32_T, next_world_rank, next_world_rank, MPI_COMM_WORLD);
            MPI_Send(embBuf, count, MPI_FLOAT, next_world_rank, next_world_rank, MPI_COMM_WORLD);
            // TODO: Error: different scope when dynamic loading so file
            // this->messenger.worldSendFP32(embBuf, count, next_world_rank, next_world_rank);
            return std::tuple<float *, int, int>(nullptr, 0, 0);
        }
#endif

        // Prepare input for final Layer Norm (only care about the last row of the result)
        // Shape of embBuf: (bs, seqLen, hiddenSize)
        MlpOutT *lnIn = embBuf;
        if (inputSeqLen > 1 && !logitsAll) { // copy is not needed when seqLen = 1 or logitsAll is true
            lnIn = outBuf;
#pragma omp parallel for
            for (int b = 0; b < batchSize; ++b) {
                xft::memcopy(lnIn + b * hiddenSize, embBuf + ((b + 1) * inputSeqLen - 1) * hiddenSize,
                        hiddenSize * sizeof(MlpOutT), ctx->device ? ctx->device : nullptr);
            }
        }

#ifdef XFT_DEBUG
        dbg.debugPrint(">>> DecoderLayer Output[%d, %d] (%d):\n", batchSize * inputSeqLen, hiddenSize, hiddenSize);
        dbg.dumpMatrix(embBuf, batchSize * inputSeqLen, hiddenSize, hiddenSize, false, ctx->device);
        dbg.debugPrint("LayerNorm In:\n");

        if (!logitsAll) {
            dbg.dumpMatrix(lnIn, batchSize, hiddenSize, hiddenSize, false, ctx->device);
        } else {
            dbg.dumpMatrix(lnIn, batchSize * inputSeqLen, hiddenSize, hiddenSize, false, ctx->device);
        }
#endif

        // LN, as it supports inplace computing, input and output can be the same
        MlpOutT *lnOut = embBuf;
        if (!logitsAll)
            lastLayerNormForward(lnIn, lnOut, batchSize);
        else
            lastLayerNormForward(lnIn, lnOut, batchSize * seqLen);

#ifdef XFT_DEBUG
        dbg.debugPrint("LayerNorm Out:\n");
        if (!logitsAll) {
            dbg.dumpMatrix(lnOut, batchSize, hiddenSize, hiddenSize, false, ctx->device);
        } else {
            dbg.dumpMatrix(lnOut, batchSize * inputSeqLen, hiddenSize, hiddenSize, false, ctx->device);
        }
#endif

        // Predictor
        const int splitSize = this->predictor->getSplitSize();
        float *finalOut = (float *)outBuf;
        if (!logitsAll)
            this->predictor->forward(ctx, lnOut, finalOut, batchSize);
        else
            this->predictor->forward(ctx, lnOut, finalOut, batchSize * seqLen);

#ifdef XFT_DEBUG
        dbg.debugPrint("finalOut:\n");
        if (!logitsAll) {
            dbg.dumpMatrix(finalOut, batchSize, splitSize, splitSize, false, ctx->device);
        } else {
            dbg.dumpMatrix(finalOut, batchSize * inputSeqLen, splitSize, splitSize, false, ctx->device);
        }
#endif

#ifdef XFT_GPU
        TimeLine tmcpyg2c("Decoder.memcopyGPU2CPU");
        embBuf = (AttnInT *)actBuffers->Data();
        float *finalOutCPU = (float *)(embBuf + batchSize * inputSeqLen * hiddenSize);
        xft::memcopy(finalOutCPU, finalOut, batchSize * splitSize * sizeof(float), ctx->device);
        finalOut = finalOutCPU;
        tmcpyg2c.release();
#endif

        // Expand the result to make it cover multiple beams
        if (step == 0 && beamSize > 1) {
            for (int b = userSideBS - 1; b >= 0; --b) {
                float *src = finalOut + b * splitSize;
#pragma omp parallel for
                for (int idx = b * beamSize; idx < (b + 1) * beamSize; ++idx) {
                    if (idx == b) { continue; }
                    float *dst = finalOut + idx * splitSize;
                    memcpy(dst, src, splitSize * sizeof(float));
                }
            }
        }

        // free temporary new ids for prefix sharing
        if (step == 0 && this->prefixSharing) { free(ids); }

        return std::tuple<float *, int, int>(
                finalOut, this->predictor->getSplitOffset(), this->predictor->getSplitSize());
    }

    std::tuple<float *, int, int> forward(std::vector<xft::SequenceMeta *> &seqs, bool logitsAll = false) {
        // Assume all sequences are all prompts(step==0) or all decodes(step>0)
        // Assume input has been synced with master in higher level.
        TimeLine t("Decoder.forward");
        TimeLine t1("Decoder.embedding");

        if (unlikely(seqs.empty())) { return std::tuple<float *, int, int>(nullptr, 0, 0); }

        DecoderContext *ctx = this->getContext();
        int batchSize = seqs.size();
        int hiddenSize = ctx->hiddenSize;

        // Prepare input
        int totInputSeqLen = 0;
        int totPastSeqLen = 0;
        std::vector<int> allInputIds;
        for (auto seq : seqs) {
            totInputSeqLen += seq->getInputSeqLen();
            totPastSeqLen += seq->getPastSeqLen();
            auto ids = seq->getInputTokens();
            allInputIds.insert(allInputIds.end(), ids.begin(), ids.end());
        }

        // Prepare context
        ctx->resize(totInputSeqLen, totInputSeqLen + totPastSeqLen);

        // Prepare buffers
        int logitRows = (!logitsAll && seqs[0]->getStep() == 0) ? seqs.size() : totInputSeqLen;
        prepareBuffer(ctx, totInputSeqLen, logitRows);

        AttnInT *embBuf = (AttnInT *)actBuffers->Data();
        MlpOutT *outBuf = (MlpOutT *)(embBuf + totInputSeqLen * hiddenSize);

        // Embedding
        this->embeddingForward(allInputIds.data(), embBuf, totInputSeqLen);
        t1.release();
        
#ifdef XFT_GPU
        TimeLine tmcpyc2g("Decoder.memcopyCPU2GPU");
        size_t embBufSize = totInputSeqLen * hiddenSize * sizeof(AttnInT);
        AttnInT *embBufGPU = (AttnInT *)ctx->getBuffer<AttnInT>("embBufGPU", embBufSize, ctx->device);
        AttnInT *outBufGPU = (AttnInT *)ctx->getBuffer<AttnInT>(
                "outBufGPU", actBuffers->Rows() * actBuffers->Cols() * sizeof(float) - embBufSize, ctx->device);
        xft::memcopy(embBufGPU, embBuf, embBufSize, ctx->device);
        embBuf = embBufGPU;
        outBuf = outBufGPU;
        tmcpyc2g.release();
#endif

        // Decoder block (all layers)
        decoderBlock->forward(ctx, seqs, embBuf, embBuf);

        // Prepare input for final Layer Norm (only care about the last row of the result)
        // Shape of embBuf: (total_input_seqlen, hiddenSize)
        MlpOutT *lnIn = embBuf;
        if (logitRows != totInputSeqLen) {
            int offset = -1;
            for (int b = 0; b < batchSize; ++b) {
                offset += seqs[b]->getInputSeqLen();
                xft::memcopy(
                        lnIn + b * hiddenSize, embBuf + offset * hiddenSize, hiddenSize * sizeof(MlpOutT), ctx->device);
            }
        }

#ifdef XFT_DEBUG
        dbg.debugPrint(">>> DecoderLayer Output[%d, %d] (%d):\n", logitRows, hiddenSize, hiddenSize);
        dbg.dumpMatrix(embBuf, logitRows, hiddenSize, hiddenSize, false, ctx->device);
        dbg.debugPrint("LayerNorm In:\n");

        dbg.dumpMatrix(lnIn, logitRows, hiddenSize, hiddenSize, false, ctx->device);
#endif

        // Last normalization layer
        MlpOutT *lnOut = embBuf;
        lastLayerNormForward(lnIn, lnOut, logitRows);

#ifdef XFT_DEBUG
        dbg.debugPrint("LayerNorm Out:\n");
        dbg.dumpMatrix(lnOut, logitRows, hiddenSize, hiddenSize, false, ctx->device);
#endif

        // Predictor
        float *finalOut = (float *)outBuf;
        auto splitSize = this->predictor->getSplitSize();
        this->predictor->forward(ctx, lnOut, finalOut, logitRows);

#ifdef XFT_DEBUG
        dbg.debugPrint("finalOut:\n");
        dbg.dumpMatrix(finalOut, logitRows, splitSize, splitSize, false, ctx->device);
#endif

#ifdef XFT_GPU
        TimeLine tmcpyg2c("Decoder.memcopyGPU2CPU");
        embBuf = (AttnInT *)actBuffers->Data();
        float *finalOutCPU = (float *)(embBuf + totInputSeqLen * hiddenSize);
        xft::memcopy(finalOutCPU, finalOut, logitRows * splitSize * sizeof(float), ctx->device);
        finalOut = finalOutCPU;
        tmcpyg2c.release();
#endif

        return std::tuple<float *, int, int>(
                finalOut, this->predictor->getSplitOffset(), this->predictor->getSplitSize());
    }

    void setPrefix(int *ids, int seqLen) {
        this->prefixSharing = true;
        this->prefixSeqLen = seqLen;
        prefixForward(ids, seqLen);
    }

    void unsetPrefix() { this->prefixSharing = false; }

    void prefixForward(int *ids, int seqLen) {
        // Assume input has been synced with master in higher level.
        // Assume the prefix token's shape is [1][1][seqLen].
        TimeLine t("Decoder.prefixForward");
        TimeLine t1("Decoder.prefixEmbedding");

        // Prepare context
        DecoderContext *ctx = this->getContext();
        ctx->resize(1, seqLen, 0);

        prepareBuffers(ctx, 1, 1, false, true);

        AttnInT *embBuf = (AttnInT *)actBuffers->Data();
        MlpOutT *outBuf = (MlpOutT *)(embBuf + 1 * seqLen * ctx->hiddenSize);

        // Embedding
        this->embeddingForward(ids, embBuf, 1 * seqLen);

        // Prepare attention mask
        this->prepareAttnMask(ids, 0);

        // Token position ids, note: different models may have different impl.
        int *positionIds = this->getPositionIds(ids, 1, seqLen, 0);
        t1.release();

        // Decoder: forward
        // TODO: Add PIPELINE_PARALLEL feature
        int hiddenSize = ctx->hiddenSize;
        for (int i = 0; i < this->decoderBlock->size(); ++i) {
            int workers = this->messenger.getSize();
            KVCacheTensor<KVCacheT> &presentKey = this->kvCacheMgr->getPrefixKey(i);
            KVCacheTensor<KVCacheT> &presentValue = this->kvCacheMgr->getPrefixValue(i);

            // Pls be noted: in attention, 'outBuf' is used as imtermediate buffer, 'tmpBuf' is used as output
            AttnOutT *attnOut = (AttnOutT *)(this->getContext()->tmpBuf.Data());
            decoderBlock->get(i)->forwardAttention(getContext(), embBuf, outBuf, attnOut, attnMask,
                    presentKey, // presentKey,
                    presentValue, // presentValue,
                    seqLen, // inputSeqLen,
                    0, // pastSeqLen
                    true, // useSelfAttn,
                    true, // doLnBefore,
                    positionIds);

            // Merge the result of attention
            // When attention and FFN/MLP are in parallel, do not need to reduce after attention
            if constexpr (!ATTN_MLP_PARALLEL) {
                if (this->messenger.getSize() > 1) { this->messenger.reduceAdd(attnOut, attnOut, seqLen * hiddenSize); }
            }

            // When attention and FFN/MLP are in parallel, use the initial embedding as input
            if constexpr (ATTN_MLP_PARALLEL) {
                if (this->messenger.getSize() > 1) {
                    decoderBlock->get(i)->forwardFFN(getContext(), embBuf, outBuf, hiddenSize, hiddenSize, true);
                    this->messenger.reduceAdd(outBuf, embBuf, seqLen * hiddenSize);
                } else {
                    decoderBlock->get(i)->forwardFFN(getContext(), embBuf, embBuf, hiddenSize, hiddenSize, true);
                }
            } else {
                // FFN (for multiple workers, output into outBuf and then reduce add to embBuf)
                if (this->messenger.getSize() > 1) {
                    decoderBlock->get(i)->forwardFFN(getContext(), attnOut, outBuf, hiddenSize, hiddenSize, true);
                    this->messenger.reduceAdd(outBuf, embBuf, seqLen * hiddenSize);
                } else {
                    decoderBlock->get(i)->forwardFFN(getContext(), attnOut, embBuf, hiddenSize, hiddenSize, true);
                }
            }
        }
    }

    // Reorder cached keys and values, size=batchSize*beamSize
    void reorderCache(int *idx, int size) { kvCacheMgr->reorderCache(idx, size, initSeqLen, accSeqLen); }

    // Get decoder context
    DecoderContext *getContext() { return context.get(); }

    // How many layers on Duty
    int getLayers() { return decoderBlock->size(); }

    Messenger &getMessenger() { return messenger; }

    bool isMaster() { return messenger.isMaster(); }

    int getRank() { return messenger.getRank(); }

    int getEndId() { return endId; }

    int getInitSeqLen() { return initSeqLen; }

    std::tuple<std::shared_ptr<DecoderContext>, std::shared_ptr<KVCacheManager<KVCacheT>>,
            std::shared_ptr<xft::Matrix<float>>>
    getSharedResources() {
        return std::make_tuple(context, kvCacheMgr, actBuffers);
    }

    void setSharedResources(const std::tuple<std::shared_ptr<DecoderContext>, std::shared_ptr<KVCacheManager<KVCacheT>>,
            std::shared_ptr<xft::Matrix<float>>> &r) {
        this->context = std::get<0>(r);
        this->kvCacheMgr = std::get<1>(r);
        this->actBuffers = std::get<2>(r);
    }

    // When first step is skipped, call this function to make everything aligned
    void skipFirstStep(int initSeqLen) {
        // Reset initial and accumulated sequence length at the first step
        this->initSeqLen = initSeqLen;
        this->accSeqLen = initSeqLen;
    }

protected:
    using DECODER = Decoder<ATTN_CLS, MLP_CLS>;

    DecoderContext *getDecoderContext(int layers, const int hiddenSize, const int headSize, const int attHeadNum,
            const int kvHeadNum, const int imSize, const std::string &act, const float epsilon, int vocabSize,
            int embeddingSize, int maxPositions, int maxPosEmbed, int maxSeqLength, bool useLogN, bool useNTK,
            RopeParams *ropeParamsPtr) {
        Env &env = Env::getInstance();
        int tpSize = messenger.getSize();
        int tpRank = messenger.getRank();
        int ppSize = env.getPipelineStage();
        int ppRank = messenger.getColor();
        // printf("ppSize: %d, ppRank: %d, tpSize: %d, tpRank: %d\n", ppSize, ppRank, tpSize, tpRank);

        if (context != nullptr) {
            if (context->hiddenSize == hiddenSize && context->attHeadNum == attHeadNum
                    && context->kvHeadNum == kvHeadNum && context->intermediateSize == imSize
                    && context->tpRank == tpRank) {
                return context.get();
            } else {
                printf("Different context size not unsupported!\n");
                exit(-1);
            }
        } else {
            int engineIdx = 0;
            if (env.getEngineKind() == xft::DeviceKind::iGPU && env.getEngineIndex() < 0) // Sequential assignment
                engineIdx = ppRank * tpSize + tpRank;
            else // assignment through the user
                engineIdx = env.getEngineIndex();

            this->mmHelper.reset(new MMHelper(env.getEngineKind(), engineIdx));
#ifdef XFT_GPU
            if (env.getEngineKind() == xft::DeviceKind::iGPU) {
                auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
                this->device.reset(new sycl::queue(devices[this->mmHelper->getEngineCount() + engineIdx]));
            }
#endif
            this->context.reset(new DecoderContext(layers, hiddenSize, headSize, attHeadNum, kvHeadNum, imSize, act,
                    epsilon, vocabSize, embeddingSize, maxPositions, maxPosEmbed, maxSeqLength, tpRank, tpSize,
                    this->mmHelper.get(), this->device.get(), ppSize, ppRank, ropeParamsPtr, useLogN, useNTK));
        }

        return this->context.get();
    }

    void setPredictorWeight(DecoderContext *ctx, const std::string &modelPath) {
        int inputSize = predictor->getInputSize();
        int outputSize = predictor->getOutputSize();

        float *weight = (float *)malloc(inputSize * outputSize * sizeof(float));
        float *bias = nullptr;

        xft::DataType wType = xft::DataType::unknown;
        if (getWeightType(modelPath + "/config.ini") == xft::DataType::fp8_e4m3){
            wType = xft::DataType::bf16;
        }

        loadWeight(modelPath + "/model.lm_head.weight.bin", weight, inputSize * outputSize, wType);

        predictor->setWeight(ctx, weight, bias);

        free(weight);
    }

    virtual void prepareBuffers(
            DecoderContext *ctx, int userSideBS, int beamSize, bool logitsAll = false, bool prefix = false) {
        int batchSize = ctx->batchSize;
        int hiddenSize = ctx->hiddenSize;
        int seqLen = ctx->inputSeqLen;
        int vocabSize = ctx->vocabSize;
        int maxPositions = ctx->maxPositions;
        int layers = this->decoderBlock->size();
        int workers = this->messenger.getSize();
        int rank = this->messenger.getRank();

        // Prepare buffers
        int logitsLen = logitsAll ? batchSize * seqLen : userSideBS * beamSize;
        int actRows = batchSize * seqLen; // rows for activation

        // Convert final output buffer size into rows in the units of hiddenSize
        int outRows = actRows;
        if (logitsLen * vocabSize > outRows * hiddenSize) { outRows = logitsLen * vocabSize / hiddenSize + 1; }

        this->actBuffers->Resize(actRows + outRows, hiddenSize);

        // Attention mask
        int sizeRequired = batchSize * seqLen * seqLen;
        getAttnMask(sizeRequired);

        // Cached keys/values
        // The maximum sequence length is to be the same as maxPositions, at most
        // And the cache always needs to account for beam size
        auto ranges = SplitUtil::getHeadRange(ctx->attHeadNum, ctx->kvHeadNum, workers, rank);
        auto kvRange = ranges.second;
        int headsPerSplit = kvRange.second - kvRange.first;

        this->kvCacheMgr->resize(prefix ? this->prefixSeqLen : maxPositions, userSideBS * beamSize, headsPerSplit,
                ctx->attHeadSize, prefix);
    }

    void prepareBuffer(DecoderContext *ctx, int totInputSeqLen, int logitRows) {
        int hiddenSize = ctx->hiddenSize;
        int vocabSize = ctx->vocabSize;

        // Convert final output buffer size into units of hiddenSize
        int outRows = std::ceil(1.0f * logitRows * vocabSize / hiddenSize);

        this->actBuffers->Resize(totInputSeqLen + outRows, hiddenSize);
    }

    float *getAttnMask(int sizeRequired) {
        if (this->maskSize < sizeRequired) {
            if (this->attnMask) free(this->attnMask);
            this->attnMask = (float *)xft::alloc(sizeRequired * sizeof(float));
            this->maskSize = sizeRequired;
        }
        return this->attnMask;
    }

    int getStartId() { return startId; }

    virtual void embeddingForward(int *ids, float *output, int tokenSize) {
        printf("embeddingForward(float) must be implemented.\n");
        exit(-1);
    }
    virtual void embeddingForward(int *ids, bfloat16_t *output, int tokenSize) {
        printf("embeddingForward(bfloat16_t) must be implemented.\n");
        exit(-1);
    }
    virtual void embeddingForward(int *ids, float16_t *output, int tokenSize) {
        printf("embeddingForward(float16_t) must be implemented.\n");
        exit(-1);
    }

    virtual void lastLayerNormForward(float *input, float *output, int rows) {
        printf("lastLayerNormForward(float) must be implemented.\n");
        exit(-1);
    }
    virtual void lastLayerNormForward(bfloat16_t *input, bfloat16_t *output, int rows) {
        printf("lastLayerNormForward(bfloat16_t) must be implemented.\n");
        exit(-1);
    }
    virtual void lastLayerNormForward(float16_t *input, float16_t *output, int rows) {
        printf("lastLayerNormForward(float16_t) must be implemented.\n");
        exit(-1);
    }

    virtual void prepareAttnMask(int *ids, int step) = 0;

public:
    virtual int *getPositionIds(int *ids, int batchSize, int seqLen, int step) { return nullptr; }

protected:
    // For communication
    Messenger &messenger;

    // Execution context
    std::shared_ptr<DecoderContext> context;
    std::shared_ptr<MMHelper> mmHelper;
    std::shared_ptr<void> device;

    // The initial input sequence length, which is the prompt token size
    int initSeqLen;
    // Accumulated sequence length, = past_seq_len + current_seq_len
    int accSeqLen;
    // The prefix input  sequence length
    int prefixSeqLen;

    bool prefixSharing;

    // If not the master, need to receive token IDs from the master
    int *inputTokens;

    std::shared_ptr<KVCacheManager<KVCacheT>> kvCacheMgr;

    // Embedding output data type = input data type of Attention
    using AttnInT = typename AttnTypeExtractor<ATTN_CLS>::Tin;
    using AttnOutT = typename AttnTypeExtractor<ATTN_CLS>::Tout;
    using MlpInT = typename MlpTypeExtractor<MLP_CLS>::Tin;
    using MlpOutT = typename MlpTypeExtractor<MLP_CLS>::Tout;

    // Activation buffers (declared as float, but the actual data type may be different)
    std::shared_ptr<xft::Matrix<float>> actBuffers;

protected:
    // Decoder block (all decoder layers)
    DecoderBlock<ATTN_CLS, MLP_CLS, KVCacheT, ATTN_MLP_PARALLEL> *decoderBlock;

    using LinearWeiT = typename std::conditional<std::is_same_v<MlpOutT, bfloat16_t>, bfloat16_t, float16_t>::type;
    DistLinear<LinearWeiT> *predictor;

private:
    int maskSize; // size of allocated attnMask
    float *attnMask; // attention mask, set as private as may need to enlarge

    int startId;
    int endId;

#ifdef XFT_DEBUG
    Debugger dbg;
#endif
};
