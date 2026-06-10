// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Loads an externalized INT8 Per-Layer Embeddings (PLE) table and gathers
/// per-token rows to feed the `ple_embeddings` graph input.
///
/// Gemma4 externalizes its per-layer embedding table (one INT8 row of
/// `numLayers * perLayerDim` values per vocabulary token) into a separate
/// `*_ple.safetensors` artifact instead of baking the multi-GB table into the
/// model graph. At inference we mmap the file and copy the rows for the current
/// batch's tokens into the `ple_embeddings` input; the graph dequantizes them
/// in-place with the scale/zero-point baked in at export time.
///
/// ## Safetensors layout
/// `[8-byte little-endian header length][JSON header][raw tensor bytes]`. The
/// `embed_tokens_per_layer` entry is a 2-D INT8 tensor of shape
/// `[vocabSize, rowWidth]`.
final class PerLayerEmbeddings: @unchecked Sendable {
    /// The mmapped file contents (header + raw INT8 rows).
    private let data: Data
    /// Byte offset where the INT8 tensor data begins.
    private let dataStart: Int
    /// Number of vocabulary rows.
    let vocabSize: Int
    /// INT8 elements per token row (`numLayers * perLayerDim`).
    let rowWidth: Int

    private static let tensorKey = "embed_tokens_per_layer"

    enum PLEError: Error, CustomStringConvertible {
        case tooSmall
        case badHeader(String)
        case missingTensor

        var description: String {
            switch self {
            case .tooSmall: return "PLE file is too small to contain a safetensors header"
            case .badHeader(let m): return "PLE safetensors header invalid: \(m)"
            case .missingTensor: return "PLE file missing '\(PerLayerEmbeddings.tensorKey)' tensor"
            }
        }
    }

    private struct TensorInfo: Decodable {
        let dtype: String
        let shape: [Int]
        let dataOffsets: [Int]
        enum CodingKeys: String, CodingKey {
            case dtype
            case shape
            case dataOffsets = "data_offsets"
        }
    }

    init(contentsOf url: URL) throws {
        let mapped = try Data(contentsOf: url, options: .mappedIfSafe)
        guard mapped.count >= 8 else { throw PLEError.tooSmall }

        // First 8 bytes: little-endian uint64 JSON header length.
        var len: UInt64 = 0
        for i in 0..<8 {
            len |= UInt64(mapped[mapped.startIndex + i]) << (8 * i)
        }
        let headerLength = Int(len)
        guard mapped.count >= 8 + headerLength else { throw PLEError.tooSmall }

        let headerData = mapped.subdata(in: (mapped.startIndex + 8)..<(mapped.startIndex + 8 + headerLength))
        guard
            let json = try JSONSerialization.jsonObject(with: headerData) as? [String: Any],
            let tensorDict = json[Self.tensorKey] as? [String: Any]
        else {
            throw PLEError.missingTensor
        }
        guard
            let shape = tensorDict["shape"] as? [Int], shape.count == 2,
            let offsets = tensorDict["data_offsets"] as? [Int], offsets.count == 2
        else {
            throw PLEError.badHeader("missing/invalid shape or data_offsets for \(Self.tensorKey)")
        }

        self.data = mapped
        self.vocabSize = shape[0]
        self.rowWidth = shape[1]
        self.dataStart = 8 + headerLength + offsets[0]

        // The tensor data must actually fit in the mapped file — otherwise a
        // valid token id could index past the mmap (SIGBUS) during gather.
        let expectedBytes = vocabSize * rowWidth  // INT8 => 1 byte/element
        guard offsets[1] - offsets[0] == expectedBytes,
            dataStart + expectedBytes <= mapped.count
        else {
            throw PLEError.badHeader(
                "PLE tensor data out of bounds: shape \(shape), offsets \(offsets), "
                    + "file \(mapped.count) bytes")
        }
    }

    /// Copies the PLE rows for `tokenIDs` into `dest`, a buffer holding
    /// `batchSize * rowWidth` INT8 values laid out row-major (token-major).
    ///
    /// Tokens beyond `tokenIDs.count` (padding up to `batchSize`) are left as
    /// whatever `dest` already contains (callers pass a zeroed buffer).
    func gather(tokenIDs: [Int32], batchSize: Int, into dest: UnsafeMutableBufferPointer<Int8>) {
        precondition(dest.count >= batchSize * rowWidth, "PLE destination buffer too small")
        let count = min(batchSize, tokenIDs.count)
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            guard let base = raw.baseAddress else { return }
            let src = base.advanced(by: dataStart).assumingMemoryBound(to: Int8.self)
            for i in 0..<count {
                let token = Int(tokenIDs[i])
                guard token >= 0, token < vocabSize else { continue }
                let srcRow = src.advanced(by: token * rowWidth)
                let dstRow = dest.baseAddress!.advanced(by: i * rowWidth)
                dstRow.update(from: srcRow, count: rowWidth)
            }
        }
    }
}
