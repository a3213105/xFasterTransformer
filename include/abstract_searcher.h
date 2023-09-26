#pragma once
#include <cstdint>
#include <tuple>
#include <vector>

class AbstractSearcher {
public:
    // First call to get NextToken, return {batchSize, numBeams}. For
    // greadySearch, numBeams = 1.
    virtual std::vector<int> getNextToken(int *ids, int batchSize, int seqLen) = 0;

    // Subsequent calls to get next Token
    virtual std::vector<int> getNextToken() = 0;

    virtual bool isDone() = 0;

    virtual std::vector<int32_t> finalize() = 0;
};

struct SearcherConfig {
    bool doEarlyStopping = false;
    int maxLen = -1;
    int numBeams = 1;
    int numBeamHypsToKeep = 1;
    int eosTokenId = -1;
    int padTokenId = -1;
    float lenPenalty = 1.0;

    SearcherConfig(int maxLen_ = -1, int numBeams_ = 1, int numBeamHypsToKeep_ = 1, float lenPenalty_ = 1.0,
            bool doEarlyStopping_ = false, int eosTokenId_ = -1, int padTokenId_ = -1)
        : maxLen(maxLen_)
        , numBeams(numBeams_)
        , numBeamHypsToKeep(numBeamHypsToKeep_)
        , lenPenalty(lenPenalty_)
        , doEarlyStopping(doEarlyStopping_)
        , eosTokenId(eosTokenId_)
        , padTokenId(padTokenId_) {}
};