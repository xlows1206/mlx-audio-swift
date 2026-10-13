import Foundation
@testable import MLXAudioTTS
import Testing

// Pure-function tests for the EOS guard (yuitalk2 #237). No model files, no Metal.
//   swift test --filter Qwen3TTSEOSGuardTests
@Suite("Qwen3TTS EOS guard")
struct Qwen3TTSEOSGuardTests {
    @Test(arguments: [
        ("こんばん", 4),
        ("りんごを食べる", 8),
        ("これは123円です", 13),
        ("むっつ", 3),
        ("きみ", 2),
        ("れいぞうこ", 5),
        ("はじめ", 3),
        ("こんにちは", 5),
        ("いらっしゃいませ", 7),
        ("コーヒーをお願いします", 12),
        ("東京へ行く", 8),
        ("今晩のパーティーは何時からですか", 19),
        ("", 0),
        ("。！", 0),
        ("ABC", 0),
        ("１２", 4),
    ])
    func moraCount(text: String, expected: Int) {
        #expect(Qwen3TTSEOSGuard.moraCount(text) == expected)
    }

    @Test func middleDotAndSpacesCountZero() {
        #expect(Qwen3TTSEOSGuard.moraCount("あ・い う") == 3)
    }

    // (mora, minFrames, capFrames) at default parameters, including every half-even tie.
    @Test(arguments: [
        (0, 6, 10),
        (2, 6, 16),
        (3, 6, 19),
        (4, 6, 22),
        (5, 8, 26),
        (7, 10, 32),
        (8, 12, 35),
        (12, 18, 48),
        (13, 20, 51),
        (19, 28, 69),
    ])
    func minFramesAndCap(mora: Int, minFrames: Int, cap: Int) {
        let config = Qwen3TTSEOSGuardConfig()
        #expect(Qwen3TTSEOSGuard.minFrames(mora: mora, config: config) == minFrames)
        #expect(Qwen3TTSEOSGuard.capFrames(mora: mora, config: config) == cap)
    }

    @Test func capIsAlwaysAboveMinFrames() {
        let config = Qwen3TTSEOSGuardConfig()
        for mora in 0 ... 200 {
            #expect(Qwen3TTSEOSGuard.capFrames(mora: mora, config: config)
                > Qwen3TTSEOSGuard.minFrames(mora: mora, config: config))
        }
    }

    @Test func defaultsAreOff() {
        let config = Qwen3TTSEOSGuardConfig()
        #expect(config == Qwen3TTSEOSGuardConfig.off)
        #expect(config.eosBias == 0)
        #expect(config.capEnabled == false)
        #expect(config.minFramesFloor == 6)
        #expect(config.minFramesPerMora == 1.5)
        #expect(config.capFrameRateHz == 12.5)
        #expect(config.capSlope == 0.25)
        #expect(config.capIntercept == 0.8)
    }

    @Test func capDisabledUsesLegacyLimit() {
        let off = Qwen3TTSEOSGuardConfig.off
        // 5 tokens: 5 * 6 = 30 < 75, so the legacy floor of 75 wins.
        #expect(Qwen3TTSEOSGuard.legacyMaxTokens(maxTokens: 4096, targetTokenCount: 5) == 75)
        #expect(Qwen3TTSEOSGuard.legacyMaxTokens(maxTokens: 4096, targetTokenCount: 20) == 120)
        #expect(Qwen3TTSEOSGuard.legacyMaxTokens(maxTokens: 100, targetTokenCount: 20) == 100)
        #expect(Qwen3TTSEOSGuard.effectiveMaxTokens(
            maxTokens: 4096, targetTokenCount: 5, mora: 5, config: off) == 75)
    }

    @Test func capEnabledReplacesLegacyLimit() {
        var on = Qwen3TTSEOSGuardConfig()
        on.capEnabled = true
        // Not min'd with the legacy 75: the new cap (26 at mora 5) replaces it.
        #expect(Qwen3TTSEOSGuard.effectiveMaxTokens(
            maxTokens: 4096, targetTokenCount: 5, mora: 5, config: on) == 26)
        // maxTokens still bounds it.
        #expect(Qwen3TTSEOSGuard.effectiveMaxTokens(
            maxTokens: 10, targetTokenCount: 5, mora: 5, config: on) == 10)
    }

    @Test func globalDefaultIsOff() {
        #expect(qwen3TTSEOSGuard == Qwen3TTSEOSGuardConfig.off)
    }
}
