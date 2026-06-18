// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared

/// Dual-RoPE parameters for models that precompute cos/sin in the runner (Gemma4
/// large-context). The graph takes precomputed `rope_cos`/`rope_sin` rows instead
/// of `position_ids`, so the runner builds the combined sliding+global table rows
/// per step from these. nil for models that gather RoPE in-graph from `position_ids`.
public struct RoPEConfig: Codable, Sendable, Equatable {
    public let slidingHeadDim: Int
    public let globalHeadDim: Int
    public let slidingRopeTheta: Double
    public let globalRopeTheta: Double
    public let partialRotaryFactor: Double

    public init(
        slidingHeadDim: Int,
        globalHeadDim: Int,
        slidingRopeTheta: Double,
        globalRopeTheta: Double,
        partialRotaryFactor: Double
    ) {
        self.slidingHeadDim = slidingHeadDim
        self.globalHeadDim = globalHeadDim
        self.slidingRopeTheta = slidingRopeTheta
        self.globalRopeTheta = globalRopeTheta
        self.partialRotaryFactor = partialRotaryFactor
    }

    enum CodingKeys: String, CodingKey {
        case slidingHeadDim = "sliding_head_dim"
        case globalHeadDim = "global_head_dim"
        case slidingRopeTheta = "sliding_rope_theta"
        case globalRopeTheta = "global_rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
    }
}

/// `language` block of `metadata.json` schema 0.2 — LLM-specific config.
public struct LanguageConfig: Codable, Sendable, Equatable {
    public let tokenizer: String
    public let vocabSize: Int
    public let maxContextLength: Int

    /// `true` if the bundle ships its own tokenizer directory; `false` to
    /// load via HuggingFace at runtime. Defaults to `true` when omitted.
    public let embeddedTokenizer: Bool

    /// Optional override for graph-function role → physical names. When
    /// absent, the runtime probes via `AIModelAsset.summary()` and applies
    /// known role conventions (`main`, `extend_<N>`, `load_embeddings`, ...).
    public let functionMap: FunctionMap?

    /// End-of-generation token ids beyond the tokenizer's single `eos_token`.
    /// Chat models (e.g. Gemma's `<end_of_turn>`) stop on additional ids that
    /// the tokenizer config alone doesn't expose. Empty/nil when unspecified.
    public let eosTokenIds: [Int]?

    /// Sliding-window size for models with a sliding KV cache (Gemma4). The
    /// runner uses it to build the windowed `sliding_causal_mask`. nil when the
    /// model has no sliding-window attention.
    public let slidingWindow: Int?

    /// Dual-RoPE parameters (Gemma4 large-context). When present, the graph takes
    /// precomputed `rope_cos`/`rope_sin` and the runner builds the rows from these;
    /// nil for models that gather RoPE in-graph from `position_ids`.
    public let rope: RoPEConfig?

    public init(
        tokenizer: String,
        vocabSize: Int,
        maxContextLength: Int,
        embeddedTokenizer: Bool = true,
        functionMap: FunctionMap? = nil,
        eosTokenIds: [Int]? = nil,
        slidingWindow: Int? = nil,
        rope: RoPEConfig? = nil
    ) {
        self.tokenizer = tokenizer
        self.vocabSize = vocabSize
        self.maxContextLength = maxContextLength
        self.embeddedTokenizer = embeddedTokenizer
        self.functionMap = functionMap
        self.eosTokenIds = eosTokenIds
        self.slidingWindow = slidingWindow
        self.rope = rope
    }

    enum CodingKeys: String, CodingKey {
        case tokenizer
        case vocabSize = "vocab_size"
        case maxContextLength = "max_context_length"
        case embeddedTokenizer = "embedded_tokenizer"
        case functionMap = "function_map"
        case eosTokenIds = "eos_token_ids"
        case slidingWindow = "sliding_window"
        case rope
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.tokenizer = try c.decode(String.self, forKey: .tokenizer)
        self.vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        self.maxContextLength = try c.decode(Int.self, forKey: .maxContextLength)
        self.embeddedTokenizer = try c.decodeIfPresent(Bool.self, forKey: .embeddedTokenizer) ?? true
        self.functionMap = try c.decodeIfPresent(FunctionMap.self, forKey: .functionMap)
        self.eosTokenIds = try c.decodeIfPresent([Int].self, forKey: .eosTokenIds)
        self.slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow)
        self.rope = try c.decodeIfPresent(RoPEConfig.self, forKey: .rope)
    }
}
