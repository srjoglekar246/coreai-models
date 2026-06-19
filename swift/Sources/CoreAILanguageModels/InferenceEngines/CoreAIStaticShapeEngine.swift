// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

/// Static-shape inference engine using Core AI models.
public final class StaticShapeEngine: InferenceEngine, @unchecked Sendable {
    public typealias ConfigType = ModelConfig

    public var supportsLogits: Bool { true }

    // MARK: I/O name contracts — models must use these exact names

    private static let logitsOutputName = "out_logits"
    private static let keyCacheName = "key_cache"
    private static let valueCacheName = "value_cache"
    private static let slidingKeyCacheName = "sliding_key_cache"
    private static let slidingValueCacheName = "sliding_value_cache"
    private static let slidingCausalMaskName = "sliding_causal_mask"
    private static let slidingInStepName = "sliding_in_step"
    private static let causalMaskName = "causal_mask"

    public var vocabSize: Int { config.vocabSize }

    public let config: ModelConfig
    private let model: AIModel

    // MARK: Properties

    // Lazily loaded inference functions, keyed by name.
    private var functions: [String: InferenceFunction]

    // Available function names by category.
    // Extend functions are sorted by query length (ascending) for graph selection.
    private let extendFunctionNames: [String]
    private let gatherFunctionNames: Set<String>

    // Embedding table loaded once at init.
    private let embeddingTable: NDArray

    // Externalized Per-Layer Embeddings (Gemma4), loaded once at init when the
    // model graph declares a `ple_embeddings` input. nil for models without PLE.
    private let perLayerEmbeddings: PerLayerEmbeddings?
    private static let pleInputName = "ple_embeddings"

    // Largest query length across all extend functions — used as prefill threshold.
    private let maxQueryLength: Int

    // Fixed-size state caches (IOSurface). Every model has key/value; Gemma4
    // also has a sliding-window ring. Kept as separate stored properties (not a
    // dict) so each view borrows distinct, instance-lifetime storage. When the
    // model has no sliding cache, the sliding properties alias key/value and are
    // never bound (gated by `hasSlidingCache`).
    private var keyCache: NDArray
    private var valueCache: NDArray
    private var slidingKeyCache: NDArray
    private var slidingValueCache: NDArray
    private let hasSlidingCache: Bool
    // Sliding-cache ring depth S (last dim of the sliding cache); 0 when absent.
    private let slidingRingDepth: Int

    // Right-sized global KV cache (memory + mid-context decode locality). The global
    // key/value caches are allocated at the SESSION's current ctx bucket, not the model
    // max: each bucket's graph is compiled with its own per-ctx seq strides, so the buffer
    // must be laid out for exactly the running bucket. `currentGlobalCtx` is the ctx the
    // global buffers are currently laid out for; `ensureGlobalCtx` grows + re-lays-out the
    // written prefix when decode crosses into a larger bucket. The sliding
    // ring is fixed-size across buckets, so it is allocated once and never re-laid-out.
    private var currentGlobalCtx: Int
    // ctx bucket → an extend function name with that ctx (any q_len), for fetching the
    // per-ctx global-cache state descriptors when (re)allocating.
    private let extendNameByCtx: [Int: String]

    // Combined per-dim RoPE theta (sliding ‖ global), precomputed once at init from
    // the model's dual-RoPE config. Gemma4 large-context graphs take precomputed
    // `rope_cos`/`rope_sin` rows (the runner builds them per step from this) instead
    // of `position_ids`. Empty for models that gather RoPE in-graph.
    private let ropeTheta: [Double]

    // Number of tokens already processed in the current sequence.
    private var processedTokenCount: Int = 0

    // MARK: - Initialization

    public init(
        configuration: ModelConfig,
        preparedModel: PreparedModel,
        perLayerEmbeddingsURL: URL? = nil
    ) async throws {
        self.config = configuration
        self.model = preparedModel.model
        self.functions = [:]

        let allNames = model.functionNames
        CLILogger.log("Model loaded: \(allNames.count) functions: \(allNames.sorted())")

        // Categorize functions
        self.extendFunctionNames =
            allNames
            .filter { $0.hasPrefix("extend") || $0.hasPrefix("prompt") }
            .sorted()
        self.gatherFunctionNames = Set(allNames.filter { $0.hasPrefix("gather_embeddings") })

        CLILogger.log(
            "Parsed \(extendFunctionNames.count) decoder functions, \(gatherFunctionNames.count) gather functions")

        // Compute max query length from function names for prefill threshold
        self.maxQueryLength =
            extendFunctionNames.compactMap { name -> Int? in
                let parts = name.split(separator: "_")
                return parts.last.flatMap { Int($0) }
            }.max() ?? 64

        // Grab largest context length extend function to use the descriptors for allocating largest context length
        // key/value caches.
        var largestContextExtend: (name: String, descriptor: InferenceFunctionDescriptor)?
        for name in extendFunctionNames {
            let desc = try Self.requireDescriptor(model: model, functionName: name)
            if Self.contextLength(descriptor: desc, config: configuration) == configuration.maxContextLength {
                largestContextExtend = (name, desc)
                break
            }
        }
        guard let (largestExtendName, largestExtendDescriptor) = largestContextExtend else {
            throw InferenceRuntimeError.invalidState(
                "Failed to find an extend function with the max context length of \(configuration.maxContextLength)")
        }

        // Validate output/state contract against the max-context function
        try Self.validateIOContract(descriptor: largestExtendDescriptor, functionName: largestExtendName)

        // Diagnostic: dump each bucket's KV-cache physical layout (shape / preferred
        // strides / channel interleave / byte count). Drives the right-sized-allocation
        // work — confirms whether per-ctx buckets differ only in the seq-stride and how
        // the channel interleave folds into the buffer. Gated; off by default.
        if ProcessInfo.processInfo.environment["COREAI_CACHE_LAYOUT_DEBUG"] != nil {
            for name in extendFunctionNames {
                guard let d = model.functionDescriptor(for: name) else { continue }
                for cacheName in [
                    Self.keyCacheName, Self.valueCacheName,
                    Self.slidingKeyCacheName, Self.slidingValueCacheName,
                ] {
                    guard case .ndArray(let cd) = d.stateDescriptor(of: cacheName) else { continue }
                    let il =
                        cd.interleaveLayout.map { "(dim \($0.dimension), factor \($0.factor))" } ?? "nil"
                    CLILogger.log(
                        "CACHE_LAYOUT \(name) \(cacheName): shape=\(cd.shape) "
                            + "strides=\(cd.preferredStrides) interleave=\(il) bytes=\(cd.minimumByteCount)")
                }
            }
        }

        // Load embeddings
        self.embeddingTable = try await Self.loadEmbeddingTable(from: model)

        // Load externalized Per-Layer Embeddings (Gemma4) if the graph wants them.
        let wantsPLE = largestExtendDescriptor.inputNames.contains(Self.pleInputName)
        if wantsPLE {
            guard let pleURL = perLayerEmbeddingsURL else {
                throw InferenceRuntimeError.invalidState(
                    "Model declares '\(Self.pleInputName)' input but no PLE artifact was provided")
            }
            let ple = try PerLayerEmbeddings(contentsOf: pleURL)
            CLILogger.log(
                "Loaded PLE table: vocab=\(ple.vocabSize), rowWidth=\(ple.rowWidth) from \(pleURL.lastPathComponent)")
            self.perLayerEmbeddings = ple
        } else {
            self.perLayerEmbeddings = nil
        }

        // Allocate one IOSurface NDArray per state. The sliding ring is fixed-size
        // across buckets, so it is sized from the max-context descriptor once. The
        // global key/value caches are RIGHT-SIZED: allocated at the smallest ctx
        // bucket and grown (with a written-prefix re-layout) as decode crosses into
        // larger buckets — see `ensureGlobalCtx`.
        func allocateState(_ name: String, _ descriptor: InferenceFunctionDescriptor) -> NDArray? {
            guard case .ndArray(let d) = descriptor.stateDescriptor(of: name) else {
                return nil
            }
            CLILogger.log("Cache '\(name)' allocated: \(d.minimumByteCount) bytes (IOSurface)")
            var arr = NDArray(descriptor: d)
            // Zero the full backing buffer. The chunked-flash global attention reads
            // ALL key positions every step (including the unwritten tail) with an
            // ADDITIVE mask; any garbage in unwritten positions feeds `q @ k` and can
            // overflow fp16 to Inf, which `+ (-40000)` cannot suppress → NaN logits.
            Self.zeroCache(&arr)
            return arr
        }

        // Map each ctx bucket → one of its extend function names (any q_len), so the
        // global cache can be (re)allocated from the right per-ctx state descriptor.
        var nameByCtx: [Int: String] = [:]
        for name in extendFunctionNames where name.hasPrefix("extend") {
            let ctx = try Self.contextLength(model: model, functionName: name, config: configuration)
            if nameByCtx[ctx] == nil { nameByCtx[ctx] = name }
        }
        self.extendNameByCtx = nameByCtx
        guard let smallestCtx = nameByCtx.keys.min(),
            let smallestName = nameByCtx[smallestCtx],
            let smallestDesc = model.functionDescriptor(for: smallestName)
        else {
            throw InferenceRuntimeError.invalidState("No extend function to size the global KV cache from")
        }

        // Global caches: start at the smallest ctx bucket (grown on demand).
        guard let key = allocateState(Self.keyCacheName, smallestDesc),
            let value = allocateState(Self.valueCacheName, smallestDesc)
        else {
            throw InferenceRuntimeError.invalidState(
                "No KV cache state descriptors found — cannot allocate cache buffers")
        }
        self.keyCache = key
        self.valueCache = value
        self.currentGlobalCtx = smallestCtx
        // Sliding ring (Gemma4). Absent on other models — alias key/value as an
        // unused placeholder so the properties stay non-optional and bindable.
        let slidingKey = allocateState(Self.slidingKeyCacheName, largestExtendDescriptor)
        let slidingValue = allocateState(Self.slidingValueCacheName, largestExtendDescriptor)
        self.hasSlidingCache = slidingKey != nil && slidingValue != nil
        self.slidingKeyCache = slidingKey ?? key
        self.slidingValueCache = slidingValue ?? value
        // Ring depth S = the sliding cache's sequence (last) dimension.
        if case .ndArray(let sd) = largestExtendDescriptor.stateDescriptor(of: Self.slidingKeyCacheName) {
            self.slidingRingDepth = sd.shape.last ?? 0
        } else {
            self.slidingRingDepth = 0
        }

        // Precompute the combined RoPE theta vector once (position-independent), so
        // each step only does `width` sincos per token.
        if let rope = configuration.rope {
            self.ropeTheta = Self.buildRopeTheta(rope)
            CLILogger.log(
                "RoPE precompute: width=\(self.ropeTheta.count) "
                    + "(sliding \(rope.slidingHeadDim) + global \(rope.globalHeadDim))")
        } else {
            self.ropeTheta = []
        }

        CLILogger.log("Engine initialized")
    }

    public convenience init(configuration: ModelConfig, modelURL: URL) async throws {
        let preparedModel = try await PreparedModel.prepare(at: modelURL)
        let pleURL = Self.resolvePerLayerEmbeddingsURL(near: modelURL)
        try await self.init(
            configuration: configuration,
            preparedModel: preparedModel,
            perLayerEmbeddingsURL: pleURL
        )
    }

    /// Looks for a sibling `*_ple.safetensors` artifact in the bundle directory
    /// (the parent of the `.aimodel`). Returns nil when none is present.
    static func resolvePerLayerEmbeddingsURL(near modelURL: URL) -> URL? {
        let bundleDir = modelURL.deletingLastPathComponent()
        guard
            let entries = try? FileManager.default.contentsOfDirectory(
                at: bundleDir, includingPropertiesForKeys: nil)
        else { return nil }
        return entries.first { $0.lastPathComponent.hasSuffix("_ple.safetensors") }
    }

    // MARK: - Initialization Helpers

    /// Zeroes an NDArray's backing buffer, so a freshly-allocated KV cache reads as
    /// 0 in every unwritten position. Mirrors `CoreAISequentialEngine.zeroFill`.
    private static func zeroCache(_ array: inout NDArray) {
        let count = array.shape.reduce(1, *)
        var view = array.mutableView(as: LogitsScalarType.self)
        view.withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<count { ptr[i] = 0 }
        }
    }

    // MARK: - Right-sized global KV cache (grow + re-layout)

    /// Ensures the global key/value caches are laid out for context bucket `ctx`,
    /// reallocating + re-laying-out the written prefix when it differs from the
    /// currently-allocated `currentGlobalCtx`.
    ///
    /// Each bucket's graph is compiled with its own per-ctx seq strides (channel
    /// stride = `ctx · interleave`), so the bound buffer must match the running
    /// bucket exactly. Within a session `forwardGraph` picks the smallest ctx >
    /// position, which is monotonic non-decreasing → this only ever grows mid-session;
    /// after `reset()` (position 0) the first bucket shrinks it back with nothing to copy.
    private func ensureGlobalCtx(_ ctx: Int) throws {
        guard ctx != currentGlobalCtx else { return }
        guard let name = extendNameByCtx[ctx],
            let desc = model.functionDescriptor(for: name),
            case .ndArray(let keyDesc) = desc.stateDescriptor(of: Self.keyCacheName),
            case .ndArray(let valueDesc) = desc.stateDescriptor(of: Self.valueCacheName)
        else {
            throw InferenceRuntimeError.invalidState(
                "Cannot resolve global KV cache descriptors for ctx \(ctx)")
        }

        var newKey = NDArray(descriptor: keyDesc)
        var newValue = NDArray(descriptor: valueDesc)
        Self.zeroCache(&newKey)
        Self.zeroCache(&newValue)

        // Copy the already-written seq prefix (positions [0, processedTokenCount)) into
        // the new layout. Bounded by the old capacity (always ≥ written on a grow).
        let copyLen = min(processedTokenCount, currentGlobalCtx)
        if copyLen > 0 {
            Self.copyGlobalPrefix(from: keyCache, to: &newKey, copyLen: copyLen)
            Self.copyGlobalPrefix(from: valueCache, to: &newValue, copyLen: copyLen)
        }

        keyCache = newKey
        valueCache = newValue
        let bytes = keyDesc.minimumByteCount + valueDesc.minimumByteCount
        CLILogger.log(
            "Global KV cache re-laid-out: ctx \(currentGlobalCtx) → \(ctx) "
                + "(copied \(copyLen) positions, \(bytes) bytes total k+v)")
        currentGlobalCtx = ctx
    }

    /// Copies the first `copyLen` sequence positions of a flat global KV cache from
    /// `src` to `dst`, re-laying-out for `dst`'s (larger) ctx.
    ///
    /// The cache is `[n, 1, C, 1, ctx]` with a channel interleave (dim 2, factor F):
    /// physically `[n, C/F, ctx, F]` row-major (F innermost, seq next), with no
    /// padding (`minimumByteCount == product · 2`). So for each of the
    /// `groupCount = n · C / F` `(n, channel-group)` groups, positions [0, copyLen)
    /// across the F interleaved channels form ONE contiguous run of `copyLen · F`
    /// elements at group base `g · (ctx · F)`. src and dst share interleave + group
    /// order, differing only in `ctx` (the seq stride scale) — so a per-group run copy
    /// is correct regardless of the interleave details.
    private static func copyGlobalPrefix(from src: NDArray, to dst: inout NDArray, copyLen: Int) {
        let srcShape = src.shape
        let dstShape = dst.shape
        let seqDim = srcShape.count - 1
        let srcSeq = srcShape[seqDim]
        let dstSeq = dstShape[seqDim]
        let factor = src.interleaveLayout?.factor ?? 1
        precondition(
            src.interleaveLayout?.dimension ?? 0 < seqDim,
            "copyGlobalPrefix expects the interleaved dim to be inside the seq dim")
        precondition(copyLen <= srcSeq && copyLen <= dstSeq, "copyGlobalPrefix overflow")

        let groupCount = srcShape.reduce(1, *) / srcSeq / factor
        let srcGroupStride = srcSeq * factor
        let dstGroupStride = dstSeq * factor
        let runElems = copyLen * factor

        let srcView = src.view(as: LogitsScalarType.self)
        srcView.withUnsafePointer { sptr, _, _ in
            var dstView = dst.mutableView(as: LogitsScalarType.self)
            dstView.withUnsafeMutablePointer { dptr, _, _ in
                for g in 0..<groupCount {
                    dptr.advanced(by: g * dstGroupStride)
                        .update(from: sptr.advanced(by: g * srcGroupStride), count: runElems)
                }
            }
        }
    }

    private static func requireDescriptor(
        model: AIModel, functionName: String
    ) throws -> InferenceFunctionDescriptor {
        guard let desc = model.functionDescriptor(for: functionName) else {
            throw InferenceRuntimeError.invalidState("Cannot find descriptor for '\(functionName)'")
        }
        return desc
    }

    private static func requireFunction(
        model: AIModel, functionName: String
    ) throws -> InferenceFunction {
        guard let fn = try model.loadFunction(named: functionName) else {
            throw InferenceRuntimeError.invalidState("Cannot load function '\(functionName)'")
        }
        return fn
    }

    private static func validateIOContract(
        descriptor: InferenceFunctionDescriptor, functionName: String
    ) throws {
        guard descriptor.outputNames.contains(logitsOutputName) else {
            throw InferenceRuntimeError.invalidState(
                "Function '\(functionName)' missing required output '\(logitsOutputName)'. "
                    + "Available outputs: \(descriptor.outputNames)")
        }
        if descriptor.stateNames.count == 1 {
            throw InferenceRuntimeError.invalidState(
                "Function '\(functionName)' has exactly 1 state (\(descriptor.stateNames)) "
                    + "— expected 0 (internal to model) or 2 (\(keyCacheName), \(valueCacheName))")
        }
        if descriptor.stateNames.count >= 2 {
            guard descriptor.stateNames.contains(keyCacheName),
                descriptor.stateNames.contains(valueCacheName)
            else {
                throw InferenceRuntimeError.invalidState(
                    "Function '\(functionName)' has states \(descriptor.stateNames) "
                        + "but missing required '\(keyCacheName)' and/or '\(valueCacheName)'")
            }
        }
    }

    private static func loadEmbeddingTable(from model: AIModel) async throws -> NDArray {
        CLILogger.log("Loading embeddings...")
        guard let embeddingFunction = try model.loadFunction(named: "load_embeddings") else {
            throw InferenceRuntimeError.invalidState("Cannot load 'load_embeddings'")
        }

        guard case .ndArray(let embeddingDesc) = embeddingFunction.descriptor.outputDescriptor(of: "embedding_table")
        else {
            throw InferenceRuntimeError.invalidState(
                "load_embeddings has no 'embedding_table' ndArray output descriptor")
        }
        var embeddingArray = NDArray(descriptor: embeddingDesc)

        var outputViews = InferenceFunction.MutableViews()
        outputViews.insert(&embeddingArray, for: "embedding_table")

        _ = try await embeddingFunction.run(
            inputs: [:],
            outputViews: consume outputViews
        )

        CLILogger.log("Embeddings loaded: shape=\(embeddingArray.shape)")
        return embeddingArray
    }

    // MARK: - Function Loading (lazy)

    private func loadFunction(named name: String) throws -> InferenceFunction {
        if let fn = functions[name] { return fn }
        guard let fn = try model.loadFunction(named: name) else {
            throw InferenceRuntimeError.invalidState("Cannot load function '\(name)'")
        }
        functions[name] = fn
        return fn
    }

    private func functionDescriptor(for name: String) throws -> InferenceFunctionDescriptor {
        if let fn = functions[name] { return fn.descriptor }
        guard let desc = model.functionDescriptor(for: name) else {
            throw InferenceRuntimeError.invalidState("Cannot find descriptor for '\(name)'")
        }
        return desc
    }

    /// Returns the query length for a given function by reading the
    /// `transformer_input` descriptor's sequence dimension.
    private func queryLength(of functionName: String) throws -> Int {
        let desc = try functionDescriptor(for: functionName)
        if let txName = desc.inputNames.first(where: { $0.contains("transformer_input") }),
            case .ndArray(let nd) = desc.inputDescriptor(of: txName), nd.shape.count >= 2
        {
            return nd.shape[1]
        }
        // Fallback: parse from function name (extend_<ctx>_<seq>)
        let parts = functionName.split(separator: "_")
        if let last = parts.last, let seq = Int(last) { return seq }
        return 1
    }

    /// Returns the context length for a given function by reading the
    /// key_cache state descriptor.
    private func contextLength(of functionName: String) throws -> Int {
        let desc = try functionDescriptor(for: functionName)
        return Self.contextLength(descriptor: desc, config: config)
    }

    private static func contextLength(
        model: AIModel, functionName: String, config: ModelConfig
    ) throws -> Int {
        guard let desc = model.functionDescriptor(for: functionName) else {
            return config.maxContextLength
        }
        return contextLength(descriptor: desc, config: config)
    }

    private static func contextLength(
        descriptor: InferenceFunctionDescriptor, config: ModelConfig
    ) -> Int {
        if case .ndArray(let keyDesc) = descriptor.stateDescriptor(of: keyCacheName) {
            // A dynamic (-1) last dim can't be a concrete context length; fall
            // back to the configured max so function selection still matches.
            if keyDesc.shape.contains(-1) {
                return config.maxContextLength
            }
            // KV cache shape is [n_layers, batch, channels, 1, seq_len]; the
            // context length is the last dimension. (Not max(): models with a
            // merged dual-head-dim cache have more channels than context.)
            return keyDesc.shape.last ?? config.maxContextLength
        }
        return config.maxContextLength
    }

    // MARK: - Graph Selection

    private func forwardGraph(numInputTokens: Int, currentPosition: Int, isPrefill: Bool) throws -> String {
        var pairs: [(contextLength: Int, queryLength: Int)] = []
        for name in extendFunctionNames {
            let parts = Array(name.split(separator: "_").suffix(2))
            guard parts.count == 2, let maxCtx = Int(parts[0]), let seqLen = Int(parts[1]) else { continue }
            pairs.append((maxCtx, seqLen))
        }

        let sorted = pairs.sorted { $0.queryLength < $1.queryLength }
        guard let maxPair = sorted.last else {
            throw InferenceRuntimeError.invalidState(
                "No extend functions found in static-shape engine")
        }
        let selectedSeq =
            sorted.first(where: { $0.queryLength >= numInputTokens })?.queryLength
            ?? maxPair.queryLength
        let candidates = pairs.filter { $0.queryLength == selectedSeq }

        guard
            let selected =
                candidates
                .sorted(by: { $0.contextLength < $1.contextLength })
                .first(where: { $0.contextLength > currentPosition })
        else {
            throw InferenceRuntimeError.invalidState(
                "No graph with cache_len > \(currentPosition) and seq_len = \(selectedSeq)")
        }
        return isPrefill
            ? "prompt_opt_\(selected.contextLength)_\(selected.queryLength)"
            : "extend_\(selected.contextLength)_\(selected.queryLength)"
    }

    // MARK: - Causal Mask

    private static func fillCausalMask(
        _ view: inout NDArray.MutableView<LogitsScalarType>,
        tokensInBatch: Int,
        alignedStep: Int
    ) {
        view.withUnsafeMutablePointer { ptr, shape, strides in
            // Stride-aware indexing for non-contiguous strides
            for context in 0..<shape[1] {
                for query in 0..<shape[3] {
                    let offset = context &* strides[1] &+ query &* strides[3]
                    ptr[offset] = LogitsScalarType(-40000.0)
                }
            }

            // Unmask positions where attention is allowed
            for query in 0..<tokensInBatch {
                let queryPos = alignedStep + query
                let upperBound = min(queryPos, shape[1] &- 1)
                for context in 0...upperBound {
                    let offset = context &* strides[1] &+ query &* strides[3]
                    ptr[offset] = 0
                }
            }
        }
    }

    /// Builds the sliding-window mask `(1, S, 1, q_len)` for the ring cache.
    ///
    /// The sliding cache is a ring of depth `S` (= `shape[1]`): the key/value for
    /// absolute position `p` lives at slot `p % S`. For each query at position
    /// `p = alignedStep + query` we unmask exactly the in-window causal keys —
    /// positions `[max(0, p - window + 1), p]` — at their ring slots. Because
    /// `S >= window + q_len - 1`, those `window` positions map to distinct slots
    /// (no collisions) and keys written by later queries in the same chunk stay
    /// masked. Everything else is -40000 (fp16-safe -inf).
    private static func fillSlidingMask(
        _ view: inout NDArray.MutableView<LogitsScalarType>,
        tokensInBatch: Int,
        alignedStep: Int,
        window: Int
    ) {
        view.withUnsafeMutablePointer { ptr, shape, strides in
            let ringDepth = shape[1]
            for context in 0..<shape[1] {
                for query in 0..<shape[3] {
                    let offset = context &* strides[1] &+ query &* strides[3]
                    ptr[offset] = LogitsScalarType(-40000.0)
                }
            }

            for query in 0..<tokensInBatch {
                let queryPos = alignedStep + query
                let lowerPos = max(0, queryPos &- window &+ 1)
                for pos in lowerPos...queryPos {
                    let slot = pos % ringDepth
                    let offset = slot &* strides[1] &+ query &* strides[3]
                    ptr[offset] = 0
                }
            }
        }
    }

    // MARK: - RoPE (precomputed cos/sin)

    /// Builds the combined per-dim RoPE theta vector (sliding ‖ global), the exact
    /// torch reference `Gemma4CombinedRoPE._compute_sin_and_cos`. Width =
    /// `sliding_head_dim + global_head_dim`. For position `p`, `rope_cos[d] =
    /// cos(p · theta[d])` and `rope_sin[d] = sin(p · theta[d])`.
    ///
    /// - Sliding sub-range `[0, sliding_hd)`: standard (full-rotary) RoPE — the
    ///   `sliding_hd/2` inverse frequencies repeated twice (GPT-NeoX layout).
    /// - Global sub-range `[sliding_hd, sliding_hd + global_hd)`: partial rotary —
    ///   only the first `rope_angles = floor(partial_rotary_factor · global_hd / 2)`
    ///   of the `global_hd/2` frequencies rotate; the rest are 0 (NoPE), repeated twice.
    static func buildRopeTheta(_ rope: RoPEConfig) -> [Double] {
        let slidingHd = rope.slidingHeadDim
        let globalHd = rope.globalHeadDim
        var theta = [Double](repeating: 0, count: slidingHd + globalHd)

        let sHalf = slidingHd / 2
        for j in 0..<slidingHd {
            let k = j % sHalf
            theta[j] = pow(rope.slidingRopeTheta, -(Double(2 * k) / Double(slidingHd)))
        }

        let gHalf = globalHd / 2
        let ropeAngles = Int((rope.partialRotaryFactor * Double(globalHd)) / 2.0)
        for j in 0..<globalHd {
            let m = j % gHalf
            theta[slidingHd + j] =
                m < ropeAngles ? pow(rope.globalRopeTheta, -(Double(2 * m) / Double(globalHd))) : 0
        }
        return theta
    }

    /// Fills the `rope_cos` / `rope_sin` graph inputs for this batch's positions.
    /// Token `i` is at absolute position `alignedStep + i`; for each of the `width`
    /// dims we write `cos`/`sin` of `pos · theta[d]` as float16. `q_len ≤ 64` and
    /// `width = 768`, so this is a few × 10⁴ sincos per step — negligible vs the
    /// graph forward (confirmed by the `RopeBuild` profile span).
    private func fillRope(
        into inputs: inout [String: NDArray],
        desc: InferenceFunctionDescriptor,
        batchSize: Int,
        alignedStep: Int
    ) throws {
        guard !ropeTheta.isEmpty else {
            throw InferenceRuntimeError.invalidState(
                "Graph wants 'rope_cos'/'rope_sin' but the model config has no rope parameters")
        }
        for (name, isCos) in [("rope_cos", true), ("rope_sin", false)] {
            guard case .ndArray(let nd) = desc.inputDescriptor(of: name) else {
                throw InferenceRuntimeError.invalidState("Graph '\(name)' has no ndArray descriptor")
            }
            var arr = NDArray(descriptor: nd)
            var view = arr.mutableView(as: LogitsScalarType.self)
            view.withUnsafeMutablePointer { ptr, shape, strides in
                // shape: (1, q_len, width)
                let width = shape[2]
                for i in 0..<batchSize {
                    let pos = Double(alignedStep + i)
                    let rowBase = i &* strides[1]
                    for d in 0..<width {
                        let angle = pos * ropeTheta[d]
                        let value = isCos ? cos(angle) : sin(angle)
                        ptr[rowBase &+ d &* strides[2]] = LogitsScalarType(value)
                    }
                }
            }
            inputs[name] = arr
        }
    }

    // MARK: - Generate (primary API)

    public func generate(
        with input: [TokenId],
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) throws -> some AsyncSequence<InferenceOutput, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let forced = inferenceOptions.forcedContinuation
                    let maxTokens: Int
                    if let forced {
                        maxTokens = forced.count
                    } else {
                        maxTokens = min(
                            inferenceOptions.maxTokens ?? Int.max,
                            max(0, self.config.maxContextLength - input.count)
                        )
                    }
                    let returnsLogits = inferenceOptions.includeLogits
                    var inputTokens = input

                    for i in 0..<maxTokens {
                        try Task.checkCancellation()
                        // When forced, we still need the forward pass (for logits + KV cache update)
                        // but skip the sampler — the next token is predetermined.
                        let (logits, sampledToken) = try await self.inference(
                            inputTokens: inputTokens,
                            samplingConfig: samplingConfiguration,
                            returnsLogits: returnsLogits || forced != nil
                        )

                        let nextToken = forced?[i] ?? sampledToken

                        continuation.yield(
                            InferenceOutput(
                                tokenId: nextToken,
                                logits: returnsLogits ? logits : nil
                            ))
                        inputTokens.append(nextToken)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Inference

    public func inference(
        inputTokens: [Int32], samplingConfig: SamplingConfiguration, returnsLogits: Bool
    ) async throws -> (logits: [LogitsScalarType]?, token: Int32) {
        CLILogger.log("Inference: \(inputTokens.count) tokens, processed: \(processedTokenCount)")

        let totalTokenCount = inputTokens.count
        guard processedTokenCount < totalTokenCount else {
            throw InferenceRuntimeError.invalidState("No new tokens to process")
        }

        var logitBuffer = [LogitsScalarType](repeating: 0, count: config.vocabSize)
        var currentPosition = processedTokenCount

        while currentPosition < totalTokenCount {
            let remaining = totalTokenCount - currentPosition
            let usePrefill = remaining > maxQueryLength
            let graphName = try forwardGraph(
                numInputTokens: remaining, currentPosition: currentPosition, isPrefill: usePrefill)

            // Right-size / grow the global KV cache to this bucket's ctx before binding
            // (the sliding ring is fixed-size and never re-laid-out). processedTokenCount
            // is the written prefix copied into the new layout on a grow.
            try ensureGlobalCtx(try contextLength(of: graphName))

            let batchSize = try queryLength(of: graphName)
            let batchStartToken = (currentPosition / batchSize) * batchSize
            let batchEndToken = min(batchStartToken + batchSize - 1, totalTokenCount - 1)
            let tokensInBatch = batchEndToken - batchStartToken + 1

            CLILogger.log("Graph: \(graphName), batch=\(batchSize), step=\(batchStartToken), tokens=\(tokensInBatch)")

            let prepareSpan = InstrumentsProfiler.beginPrepareStep(
                operation: "buildInputs", engine: "StaticShape")
            let inputs = try await buildInputs(
                graphName: graphName,
                batchTokens: inputTokens[batchStartToken...batchEndToken],
                batchSize: batchSize,
                alignedStep: batchStartToken,
                tokensInBatch: tokensInBatch
            )
            prepareSpan.end()

            let logitsSpan = InstrumentsProfiler.beginLogitsInference(
                step: batchStartToken, tokens: tokensInBatch, engine: "StaticShape")

            let fn = try loadFunction(named: graphName)
            let desc = try functionDescriptor(for: graphName)

            // Bind every state this function declares from its persistent cache.
            // All extend functions share cache shape/strides/interleave per state,
            // so no copy is needed on graph switch — we slice to this function's
            // descriptor shape (e.g. the global cache's smaller cache_len window).
            // Bind every state this function declares from its persistent cache.
            // Views alias the instance-lifetime IOSurface backing (no copy on
            // graph switch), sliced to this function's descriptor shape (e.g.
            // the global cache's smaller cache_len window).
            guard case .ndArray(let keyCacheDescriptor) = desc.stateDescriptor(of: Self.keyCacheName),
                case .ndArray(let valueCacheDescriptor) = desc.stateDescriptor(of: Self.valueCacheName)
            else {
                throw InferenceRuntimeError.invalidState("Missing KV cache state descriptors for '\(graphName)'")
            }

            // Build state views and run. Each branch keeps the view borrows and
            // the run() call in one straight-line scope (a lifetime-dependent
            // view can't be inserted into an outer-scope `states` from inside a
            // nested block), so the sliding case is its own branch.
            var outputs: InferenceFunction.Outputs
            if hasSlidingCache,
                case .ndArray(let slidingKeyDescriptor) = desc.stateDescriptor(of: Self.slidingKeyCacheName),
                case .ndArray(let slidingValueDescriptor) = desc.stateDescriptor(of: Self.slidingValueCacheName)
            {
                var states = InferenceFunction.MutableViews()
                states.insert(
                    keyCache.mutableRawView().slice(at: keyCacheDescriptor.shape.map { 0..<$0 }),
                    for: Self.keyCacheName)
                states.insert(
                    valueCache.mutableRawView().slice(at: valueCacheDescriptor.shape.map { 0..<$0 }),
                    for: Self.valueCacheName)
                states.insert(
                    slidingKeyCache.mutableRawView().slice(at: slidingKeyDescriptor.shape.map { 0..<$0 }),
                    for: Self.slidingKeyCacheName)
                states.insert(
                    slidingValueCache.mutableRawView().slice(at: slidingValueDescriptor.shape.map { 0..<$0 }),
                    for: Self.slidingValueCacheName)
                outputs = try await fn.run(
                    inputs: inputs, states: consume states, outputViews: InferenceFunction.MutableViews())
            } else {
                var states = InferenceFunction.MutableViews()
                states.insert(
                    keyCache.mutableRawView().slice(at: keyCacheDescriptor.shape.map { 0..<$0 }),
                    for: Self.keyCacheName)
                states.insert(
                    valueCache.mutableRawView().slice(at: valueCacheDescriptor.shape.map { 0..<$0 }),
                    for: Self.valueCacheName)
                outputs = try await fn.run(
                    inputs: inputs, states: consume states, outputViews: InferenceFunction.MutableViews())
            }

            let logitsArray = outputs.remove(Self.logitsOutputName)?.ndArray
            logitsSpan.end()

            // TEMPORARY diagnostic (COREAI_LOGITS_DEBUG): scan EVERY step's output
            // (prefill included) for NaN/Inf, to pinpoint the step/graph where the
            // multi-block path first blows up.
            if ProcessInfo.processInfo.environment["COREAI_LOGITS_DEBUG"] != nil,
                let logitsArray
            {
                let dbgView = logitsArray.view(as: LogitsScalarType.self)
                if let els = dbgView.contiguousElements {
                    var bad = 0
                    var idx = 0
                    let n = els.count
                    while idx < n {
                        let x = Float(els[idx])
                        if x.isNaN || x.isInfinite { bad += 1 }
                        idx += 1
                    }
                    CLILogger.log(
                        "LOGITS_DEBUG graph=\(graphName) step=\(batchStartToken) "
                            + "prefill=\(usePrefill) nan/inf=\(bad)/\(n)")
                }
            }

            // Extract logits from the last token position.
            if !usePrefill, let logitsArray {
                let copySpan = InstrumentsProfiler.beginLogitsCopy()
                let logitsView = logitsArray.view(as: LogitsScalarType.self)
                guard let logits = logitsView.contiguousElements else {
                    throw InferenceRuntimeError.invalidState(
                        "Logits array has non-contiguous (interleaved) layout — cannot extract values safely")
                }
                let offset = (tokensInBatch - 1) * config.vocabSize
                for i in 0..<config.vocabSize {
                    logitBuffer[i] = logits[offset + i]
                }
                copySpan.end()
            }

            currentPosition = batchEndToken + 1
            processedTokenCount = currentPosition
        }

        let actualLogits = returnsLogits ? logitBuffer : nil
        let sampleSpan = InstrumentsProfiler.beginSample(strategy: "cpu-fallback")
        let nextToken = samplingConfig.fallbackSampler(from: &logitBuffer)
        sampleSpan.end()
        CLILogger.log("Token: \(nextToken), processed: \(processedTokenCount)")
        return (logits: actualLogits, token: nextToken)
    }

    // MARK: - Inference Helpers

    private func buildInputs<Tokens: Collection<Int32>>(
        graphName: String,
        batchTokens: Tokens,
        batchSize: Int,
        alignedStep: Int,
        tokensInBatch: Int
    ) async throws -> [String: NDArray] {
        let desc = try functionDescriptor(for: graphName)
        var inputs = [String: NDArray]()

        if desc.inputNames.contains("embedding_table") {
            inputs["embedding_table"] = embeddingTable
        }

        // Gather embeddings for this batch's tokens
        if let txName = desc.inputNames.first(where: { $0.contains("transformer_input") }) {
            let gatherName = "gather_embeddings_\(batchSize)"
            guard gatherFunctionNames.contains(gatherName) else {
                throw InferenceRuntimeError.invalidState(
                    "No gather function '\(gatherName)' for batch size \(batchSize)")
            }
            let gatherSpan = InstrumentsProfiler.beginGatherEmbeddings()
            let gatheredOpt = try await runGather(tokenIDs: Array(batchTokens), batchSize: batchSize)
            gatherSpan.end()
            guard let gathered = gatheredOpt else {
                throw InferenceRuntimeError.invalidState("Gather '\(gatherName)' returned no output")
            }
            inputs[txName] = gathered
        }
        // RoPE inputs. Gemma4 large-context graphs take precomputed `rope_cos` /
        // `rope_sin` (the runner builds the combined sliding+global table rows per
        // step); all other models take `position_ids` and gather RoPE in-graph.
        if desc.inputNames.contains("rope_cos") {
            let ropeSpan = InstrumentsProfiler.beginRopeBuild()
            try fillRope(
                into: &inputs, desc: desc, batchSize: batchSize, alignedStep: alignedStep)
            ropeSpan.end()
        } else if let posName = desc.inputNames.first(where: { $0.contains("pos") }) {
            if case .ndArray(let nd) = desc.inputDescriptor(of: posName) {
                var pos = NDArray(descriptor: nd)
                var posView = pos.mutableView(as: UInt16.self)
                guard var posSpan = posView.contiguousElements else {
                    throw InferenceRuntimeError.invalidState("pos array has non-contiguous layout")
                }
                for i in 0..<batchSize {
                    posSpan[i] = UInt16(alignedStep + i)
                }
                inputs[posName] = pos
            }
        } else {
            throw InferenceRuntimeError.invalidState(
                "Graph '\(graphName)' has no 'rope_cos'/'rope_sin' or 'position_ids' input")
        }

        // Causal mask (flat (1, ctx, 1, q_len) — same for the flat single-block SDPA
        // and the chunked-flash BlockedSDPA, which walks it in block_size chunks).
        let maskSpan = InstrumentsProfiler.beginMaskBuild()
        if case .ndArray(let nd) = desc.inputDescriptor(of: Self.causalMaskName) {
            var mask = NDArray(descriptor: nd)
            var maskView = mask.mutableView(as: LogitsScalarType.self)
            Self.fillCausalMask(&maskView, tokensInBatch: tokensInBatch, alignedStep: alignedStep)
            inputs[Self.causalMaskName] = mask
        }

        // Sliding-window mask (Gemma4): like the causal mask but limited to the
        // last `window` keys and indexed into the ring by absolute position % S.
        if case .ndArray(let nd) = desc.inputDescriptor(of: Self.slidingCausalMaskName) {
            guard let window = config.slidingWindow else {
                throw InferenceRuntimeError.invalidState(
                    "Graph '\(graphName)' wants '\(Self.slidingCausalMaskName)' "
                        + "but the model config has no sliding_window")
            }
            var mask = NDArray(descriptor: nd)
            var maskView = mask.mutableView(as: LogitsScalarType.self)
            Self.fillSlidingMask(
                &maskView, tokensInBatch: tokensInBatch, alignedStep: alignedStep, window: window)
            inputs[Self.slidingCausalMaskName] = mask
        }
        maskSpan.end()

        // Step(s). Models have `in_step` (the absolute flat write offset into the
        // global cache). Gemma4 also has `sliding_in_step` = alignedStep % S, the
        // sliding ring write offset (computed here so the graph needs no in-graph
        // remainder op). Both inputs match `*step*`, so set each
        // by name.
        for stepName in desc.inputNames where stepName.contains("step") && !stepName.contains("pos") {
            guard case .ndArray(let nd) = desc.inputDescriptor(of: stepName) else { continue }
            var step = NDArray(descriptor: nd)
            var stepView = step.mutableView(as: Int32.self)
            guard var stepSpan = stepView.contiguousElements else {
                throw InferenceRuntimeError.invalidState("step array has non-contiguous layout")
            }
            if stepName == Self.slidingInStepName {
                stepSpan[0] = Int32(slidingRingDepth > 0 ? alignedStep % slidingRingDepth : alignedStep)
            } else {
                stepSpan[0] = Int32(alignedStep)
            }
            inputs[stepName] = step
        }

        // Per-Layer Embeddings (Gemma4): gather INT8 rows for this batch's tokens.
        if desc.inputNames.contains(Self.pleInputName),
            case .ndArray(let nd) = desc.inputDescriptor(of: Self.pleInputName)
        {
            guard let ple = perLayerEmbeddings else {
                throw InferenceRuntimeError.invalidState(
                    "Graph '\(graphName)' wants '\(Self.pleInputName)' but no PLE table is loaded")
            }
            let elementCount = nd.shape.reduce(1, *)
            let rowWidth = nd.shape.last ?? ple.rowWidth
            guard rowWidth == ple.rowWidth else {
                throw InferenceRuntimeError.invalidState(
                    "PLE row width mismatch: graph expects \(rowWidth), table has \(ple.rowWidth)")
            }
            var pleArray = NDArray(descriptor: nd)
            var pleView = pleArray.mutableView(as: Int8.self)
            // The flat row-major gather below assumes a contiguous buffer; the
            // ple_embeddings input is exported without interleave so it is, but
            // verify rather than silently write to wrong offsets.
            guard pleView.contiguousElements != nil else {
                throw InferenceRuntimeError.invalidState(
                    "ple_embeddings array has non-contiguous layout")
            }
            let pleSpan = InstrumentsProfiler.beginPLEGather()
            pleView.withUnsafeMutablePointer { ptr, _, _ in
                ptr.update(repeating: 0, count: elementCount)
                let buf = UnsafeMutableBufferPointer(start: ptr, count: elementCount)
                ple.gather(tokenIDs: Array(batchTokens), batchSize: batchSize, into: buf)
            }
            pleSpan.end()
            inputs[Self.pleInputName] = pleArray
        }

        return inputs
    }

    // MARK: - Gather Embeddings

    private func runGather(tokenIDs: [Int32], batchSize: Int) async throws -> NDArray? {
        let name = "gather_embeddings_\(batchSize)"
        let fn = try loadFunction(named: name)
        let desc = try functionDescriptor(for: name)

        // Token IDs input
        let tokenInputName = "in_new_token_ids"
        guard let tokenDesc = desc.inputDescriptor(of: tokenInputName),
            case .ndArray(let tokenNDDesc) = tokenDesc
        else {
            throw InferenceRuntimeError.invalidState("No descriptor for '\(tokenInputName)'")
        }

        var tokenArray = NDArray(descriptor: tokenNDDesc)
        var tokenView = tokenArray.mutableView(as: Int32.self)
        guard var tokenSpan = tokenView.contiguousElements else {
            throw InferenceRuntimeError.invalidState("tokenArray has non-contiguous layout")
        }
        // Zero unused (padding) query slots first: a partial final batch leaves
        // slots [tokensInBatch..<batchSize] otherwise uninitialized, so they'd gather
        // a garbage token id → garbage query embedding. Padding with token 0 keeps the
        // discarded columns finite (garbage could feed NaN into shared reductions).
        for i in 0..<tokenSpan.count { tokenSpan[i] = 0 }
        if tokenNDDesc.shape.count == 2 {
            for i in 0..<min(batchSize, tokenIDs.count) {
                tokenSpan[i] = tokenIDs[i]
            }
        } else {
            tokenSpan[0] = tokenIDs[0]
        }

        var inputs: [String: NDArray] = [tokenInputName: tokenArray]
        inputs["embedding_table"] = embeddingTable

        var outputs = try await fn.run(
            inputs: inputs,
            outputViews: InferenceFunction.MutableViews()
        )

        let expectedOutput = "out_transformer_input"
        return outputs.remove(expectedOutput)?.ndArray
            ?? outputs.remove(desc.outputNames.first ?? "")?.ndArray
    }

    // MARK: - Lifecycle

    public func reset() {
        let resetSpan = InstrumentsProfiler.beginReset(engine: "StaticShape")
        processedTokenCount = 0
        resetSpan.end()
    }

    public func warmup(queryLength: Int, sampling: SamplingConfiguration?) async throws {
        for fnName in extendFunctionNames {
            self.functions[fnName] = try Self.requireFunction(model: model, functionName: fnName)
        }
        reset()
    }
}
