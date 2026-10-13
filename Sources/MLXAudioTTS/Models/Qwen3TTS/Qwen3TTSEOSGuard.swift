import Foundation

// EOS guard for short-text Qwen3-TTS generation (yuitalk2 issue #237).
//
// On-device V4 0.6B often fails to sample EOS after a short text has been read out and keeps
// emitting acoustic tokens that decode to noise. The mitigation is (a) a small positive bias on
// the EOS logit once a text-dependent minimum number of frames has been produced, and (b) a
// text-dependent frame cap that replaces the legacy `max(75, tokens * 6)` floor.
//
// Everything is OFF by default (`qwen3TTSEOSGuard == .off`). "Off" means the extra MLX ops are
// never built and `effectiveMaxTokens` is the legacy expression, so the generation loop is
// byte-for-byte what it was before.
//
// This file is deliberately pure Swift (no MLX state) so the formulas are unit-testable
// without a model or a Metal device.
//
// Known differences from the Mac/Python experiment that produced the numbers (do not treat the
// Mac failure rates as predictions for this loop):
//   1. `sampleToken` re-inserts the EOS logit after top-k/top-p/min-p, so EOS is never cut by
//      top-k here; the Python replica's `apply_top_k` could cut it.
//   2. Repetition penalty here covers ALL generated main-codebook tokens; the replica used a
//      64-frame window.
//   3. The replica capped at a fixed 150 frames; this loop's legacy cap is `max(75, tokens * 6)`.
//   4. Rounding ties are resolved half-to-even to match Python's `round()`.

/// Process-wide configuration. Snapshotted once at the start of every generation.
public struct Qwen3TTSEOSGuardConfig: Equatable, Sendable {
    public var eosBias: Float = 0            // 0 = bias off
    public var minFramesFloor: Int = 6
    public var minFramesPerMora: Double = 1.5
    public var capEnabled: Bool = false
    public var capFrameRateHz: Double = 12.5
    public var capSlope: Double = 0.25       // seconds per mora
    public var capIntercept: Double = 0.8    // seconds
    public static let off = Qwen3TTSEOSGuardConfig()

    public init() {}
}

nonisolated(unsafe) public var qwen3TTSEOSGuard = Qwen3TTSEOSGuardConfig.off

/// Read-only report of the most recent generation (see `Qwen3TTSModel.lastGenerationReport`).
public struct Qwen3TTSGenerationReport: Equatable, Sendable {
    /// Non-EOS frames generated (the step that sampled EOS is not counted).
    public var frames: Int
    /// `capFrames` when the cap is enabled, otherwise nil.
    public var cap: Int?
    /// The loop ran to its limit without ever sampling EOS (whether the limit came from the new
    /// cap or the legacy expression).
    public var hitCap: Bool
    /// The EOS bias was actually added on at least one sampling step.
    public var eosBiased: Bool

    public init(frames: Int, cap: Int?, hitCap: Bool, eosBiased: Bool) {
        self.frames = frames
        self.cap = cap
        self.hitCap = hitCap
        self.eosBiased = eosBiased
    }
}

public enum Qwen3TTSEOSGuard {
    // MARK: Mora count

    /// Per-character mora count, matching the Mac script `common_eg.py::mora()` except that the
    /// middle dot `・` (U+30FB) counts 0 here.
    ///
    ///   kana U+3041...U+30FA and the long-vowel mark `ー` (U+30FC): 1
    ///   small ya/yu/yo and small vowels (ゃゅょ ャュョ ぁぃぅぇぉ ァィゥェォ): 0
    ///   kanji U+4E00...U+9FFF and digits 0-9 / 0-9 full-width: 2 each (per digit)
    ///   everything else: 0
    public static func moraCount(_ text: String) -> Int {
        var total = 0
        for scalar in text.unicodeScalars {
            let v = scalar.value
            switch v {
            case 0x3041, 0x3043, 0x3045, 0x3047, 0x3049,   // ぁぃぅぇぉ
                 0x3083, 0x3085, 0x3087,                   // ゃゅょ
                 0x30A1, 0x30A3, 0x30A5, 0x30A7, 0x30A9,   // ァィゥェォ
                 0x30E3, 0x30E5, 0x30E7:                   // ャュョ
                continue
            case 0x3041 ... 0x30FA, 0x30FC:
                total += 1
            case 0x4E00 ... 0x9FFF, 0x0030 ... 0x0039, 0xFF10 ... 0xFF19:
                total += 2
            default:
                continue
            }
        }
        return total
    }

    // MARK: Formulas

    /// Quantize to 1e-9 first so binary error in e.g. `0.8` cannot flip a tie, then round
    /// half-to-even (Python `round()`).
    static func roundHalfEven(_ x: Double) -> Int {
        let quantized = (x * 1e9).rounded() / 1e9
        return Int(quantized.rounded(.toNearestOrEven))
    }

    /// `max(floor, round(perMora * mora))`.
    public static func minFrames(mora: Int, config: Qwen3TTSEOSGuardConfig) -> Int {
        max(config.minFramesFloor, roundHalfEven(config.minFramesPerMora * Double(mora)))
    }

    /// `round(rateHz * (slope * mora + intercept))`.
    public static func capFrames(mora: Int, config: Qwen3TTSEOSGuardConfig) -> Int {
        roundHalfEven(config.capFrameRateHz * (config.capSlope * Double(mora) + config.capIntercept))
    }

    /// The pre-guard limit. Kept as a function so the "off" path is provably this expression.
    public static func legacyMaxTokens(maxTokens: Int, targetTokenCount: Int) -> Int {
        min(maxTokens, max(75, targetTokenCount * 6))
    }

    /// With the cap enabled the new cap REPLACES the legacy expression (no min with it).
    public static func effectiveMaxTokens(
        maxTokens: Int,
        targetTokenCount: Int,
        mora: Int,
        config: Qwen3TTSEOSGuardConfig
    ) -> Int {
        guard config.capEnabled else {
            return legacyMaxTokens(maxTokens: maxTokens, targetTokenCount: targetTokenCount)
        }
        return min(maxTokens, capFrames(mora: mora, config: config))
    }
}

/// Thread-safe holder for the model's last report (written by the generation task, read by the
/// host after the stream finished).
final class Qwen3TTSReportBox: @unchecked Sendable {
    private let lock = NSLock()
    private var report: Qwen3TTSGenerationReport?

    var value: Qwen3TTSGenerationReport? {
        get { lock.withLock { report } }
        set { lock.withLock { report = newValue } }
    }
}
