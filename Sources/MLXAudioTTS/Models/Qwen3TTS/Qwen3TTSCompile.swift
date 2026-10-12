// Compiled code-predictor frame (mx.compile over the 15 sub-steps). All behaviour is behind `qwen3TTSCompileCP` (default 0 = off).
// Weights are captured as constants of the compiled graph, so the compiled closures are cached per model instance.
import Foundation
@preconcurrency import MLX
@preconcurrency import MLXLMCommon
import MLXNN
import MLXRandom

/// 0 = off (stock path), 1 = compile each code-predictor sub-step (incl. sampling), 2 = compile the whole 15-sub-step frame.
nonisolated(unsafe) public var qwen3TTSCompileCP = 0

// MARK: functional (cache-as-arrays) forward, mirrors the stock modules op-for-op

extension CodePredictorAttention {
    func fwdFunctional(
        _ x: MLXArray, positionEmbeddings: (MLXArray, MLXArray), mask: MLXArray?,
        kPrev: MLXArray?, vPrev: MLXArray?
    ) -> (MLXArray, MLXArray, MLXArray) {
        let (batch, seqLen, _) = (x.dim(0), x.dim(1), x.dim(2))
        var q = qProj(x).reshaped(batch, seqLen, numHeads, headDim)
        var k = kProj(x).reshaped(batch, seqLen, numKvHeads, headDim)
        var v = vProj(x).reshaped(batch, seqLen, numKvHeads, headDim)
        q = qNorm(q)
        k = kNorm(k)
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)
        let (cosVal, sinVal) = positionEmbeddings
        let cosE = expandedDimensions(cosVal, axis: 1)
        let sinE = expandedDimensions(sinVal, axis: 1)
        q = q * cosE + cpRotateHalf(q) * sinE
        k = k * cosE + cpRotateHalf(k) * sinE
        if let kPrev, let vPrev {
            k = concatenated([kPrev, k], axis: 2)
            v = concatenated([vPrev, v], axis: 2)
        }
        let output = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
        return (oProj(output.transposed(0, 2, 1, 3).reshaped(batch, seqLen, -1)), k, v)
    }
}

extension Qwen3TTSCodePredictor {
    /// One code-predictor forward with explicit cache arrays (nil = empty). Returns logits and the grown caches.
    func stepFunctional(
        _ inputsEmbeds: MLXArray, kc: [MLXArray?], vc: [MLXArray?], generationStep: Int
    ) -> (MLXArray, [MLXArray], [MLXArray]) {
        var embeds = inputsEmbeds
        if let proj = projection { embeds = proj(embeds) }
        let (batch, seqLen) = (embeds.dim(0), embeds.dim(1))
        let offset = kc[0]?.dim(2) ?? 0
        let pos = MLXArray(Int32(offset) ..< Int32(offset + seqLen)).reshaped(1, seqLen)
        let posIds = broadcast(pos, to: [batch, seqLen])
        let posEmb = model.rotaryEmb(embeds, positionIds: posIds)
        var causalMask: MLXArray? = nil
        if seqLen > 1 {
            causalMask = MultiHeadAttention.createAdditiveCausalMask(seqLen).asType(embeds.dtype)
        }
        var x = embeds
        var nk = [MLXArray](), nv = [MLXArray]()
        for (i, layer) in model.layers.enumerated() {
            let h = layer.inputLayernorm(x)
            let (a, k, v) = layer.selfAttn.fwdFunctional(h, positionEmbeddings: posEmb, mask: causalMask, kPrev: kc[i], vPrev: vc[i])
            var out = x + a
            out = out + layer.mlp(layer.postAttentionLayernorm(out))
            x = out
            nk.append(k); nv.append(v)
        }
        let logits = lmHead[generationStep](model.norm(x))
        return (logits, nk, nv)
    }
}

// MARK: compiled closures, cached per (model, sampling params)

final class Qwen3CPCompileStore: @unchecked Sendable {
    nonisolated(unsafe) static let shared = Qwen3CPCompileStore()
    var key = ""
    var frame: (@Sendable ([MLXArray]) -> [MLXArray])?
    var sub: [Int: @Sendable ([MLXArray]) -> [MLXArray]] = [:]
}

extension Qwen3TTSModel {
    private func cpStoreKey(_ mode: Int, _ t: Float, _ p: Float, _ k: Int, _ m: Float) -> String {
        return "\(ObjectIdentifier(self).hashValue)|\(mode)|\(t)|\(p)|\(k)|\(m)"
    }

    /// Mode 2. Returns the 15 sampled code tokens ([1,1] each) of one frame.
    func cpCompiledFrame(
        codeHidden: MLXArray, nextToken: MLXArray, temperature: Float, topP: Float, topK: Int, minP: Float
    ) -> [MLXArray] {
        let store = Qwen3CPCompileStore.shared
        let key = cpStoreKey(2, temperature, topP, topK, minP)
        if store.key != key { store.key = key; store.frame = nil; store.sub = [:] }
        let nGroups = talker.codePredictor.numCodeGroups
        if store.frame == nil {
            let cp = talker.codePredictor
            let nLayers = cp.model.layers.count
            store.frame = compile(inputs: [MLXRandom.globalState], outputs: [MLXRandom.globalState]) {
                (args: [MLXArray]) -> [MLXArray] in
                let hidden = args[0], tok0 = args[1]
                var kc = [MLXArray?](repeating: nil, count: nLayers)
                var vc = [MLXArray?](repeating: nil, count: nLayers)
                var codes = [MLXArray]()
                var last = tok0
                for codeIdx in 0 ..< nGroups - 1 {
                    let input: MLXArray
                    if codeIdx == 0 {
                        input = concatenated([hidden, self.talker.getInputEmbeddings()(tok0)], axis: 1)
                    } else {
                        input = cp.codecEmbedding[codeIdx - 1](last)
                    }
                    let (lg, nk, nv) = cp.stepFunctional(input, kc: kc, vc: vc, generationStep: codeIdx)
                    kc = nk; vc = nv
                    let c = self.sampleToken(lg, temperature: temperature, topP: topP, topK: topK, minP: minP)
                    codes.append(c); last = c
                }
                return codes
            }
        }
        let out = store.frame!([codeHidden, nextToken])
        return Array(out.prefix(nGroups - 1))
    }

    /// Mode 1. One sub-step (embedding lookup + 5 layers + lm_head + sampling) as one compiled function; caches are explicit arrays.
    func cpCompiledSubstep(
        codeIdx: Int, codeHidden: MLXArray, nextToken: MLXArray, prevToken: MLXArray,
        kc: inout [MLXArray?], vc: inout [MLXArray?],
        temperature: Float, topP: Float, topK: Int, minP: Float
    ) -> MLXArray {
        let store = Qwen3CPCompileStore.shared
        let key = cpStoreKey(1, temperature, topP, topK, minP)
        if store.key != key { store.key = key; store.frame = nil; store.sub = [:] }
        let cp = talker.codePredictor
        let nLayers = cp.model.layers.count
        if store.sub[codeIdx] == nil {
            store.sub[codeIdx] = compile(inputs: [MLXRandom.globalState], outputs: [MLXRandom.globalState]) {
                (args: [MLXArray]) -> [MLXArray] in
                let input: MLXArray
                var kIn = [MLXArray?](repeating: nil, count: nLayers)
                var vIn = [MLXArray?](repeating: nil, count: nLayers)
                if codeIdx == 0 {
                    input = concatenated([args[0], self.talker.getInputEmbeddings()(args[1])], axis: 1)
                } else {
                    input = cp.codecEmbedding[codeIdx - 1](args[0])
                    for i in 0 ..< nLayers { kIn[i] = args[1 + i]; vIn[i] = args[1 + nLayers + i] }
                }
                let (lg, nk, nv) = cp.stepFunctional(input, kc: kIn, vc: vIn, generationStep: codeIdx)
                let c = self.sampleToken(lg, temperature: temperature, topP: topP, topK: topK, minP: minP)
                return [c] + nk + nv
            }
        }
        var args: [MLXArray]
        if codeIdx == 0 {
            args = [codeHidden, nextToken]
        } else {
            args = [prevToken] + kc.map { $0! } + vc.map { $0! }
        }
        let out = store.sub[codeIdx]!(args)
        kc = (0 ..< nLayers).map { out[1 + $0] }
        vc = (0 ..< nLayers).map { out[1 + nLayers + $0] }
        return out[0]
    }
}
