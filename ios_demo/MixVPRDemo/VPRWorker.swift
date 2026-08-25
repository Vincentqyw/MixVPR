import Accelerate
import CoreVideo
import Foundation
import os
import QuartzCore
import UIKit

struct Place: Identifiable {
    let id = UUID()
    let name: String
    let descriptor: [Float]   // L2-normalised
    let thumbnail: UIImage
}

struct Match {
    let id: UUID
    let score: Float
}

struct FrameStats {
    var inferMs = 0.0
    var preMs = 0.0
    var fps = 0.0
    var scores: [UUID: Float] = [:]
    var best: Match?
}

/// Owns the engine, the registered places and every per-frame computation on one serial queue.
final class VPRWorker {
    private let queue = DispatchQueue(label: "mixvpr.work", qos: .userInteractive)
    private let inFlight = OSAllocatedUnfairLock(initialState: false)

    // Everything below is only touched on `queue`.
    private var engine: MixVPREngine?
    private var variant: ModelVariant = .fp16
    private var compute: ComputeChoice = .all
    private var paused = false
    private var latest: [Float]?
    private var lastFrame: CVPixelBuffer?
    private var places: [Place] = []
    private var frames = 0
    private var windowStart = CACurrentMediaTime()
    private var fps = 0.0
    private var emaInfer = 0.0
    private var emaPre = 0.0
    // `--log`: print per-second live stats and time a second back-to-back inference per frame
    // to separate hardware wake-up latency from the model's own cost.
    private let logMode = CommandLine.arguments.contains("--log")
    private var emaBurst: [Double] = []
    private var lastLog = CACurrentMediaTime()

    /// Called on the worker queue.
    var onStats: ((FrameStats) -> Void)?
    var onStatus: ((String) -> Void)?

    func load(variant: ModelVariant, compute: ComputeChoice) {
        queue.async {
            self.variant = variant
            self.compute = compute
            self.engine = nil
            self.onStatus?("Loading \(variant.rawValue) on \(compute.rawValue)…")
            do {
                let e = try MixVPREngine(variant: variant, compute: compute)
                self.engine = e
                self.emaInfer = 0; self.emaPre = 0; self.frames = 0; self.windowStart = CACurrentMediaTime()
                self.onStatus?(String(format: "%@ · %@ · loaded in %.0f ms", variant.rawValue, compute.rawValue, e.loadMs))
            } catch {
                self.onStatus?("Load failed: \(error)")
            }
        }
    }

    /// Called from the camera queue; drops the frame if the previous one is still being processed.
    func handle(frame: CVPixelBuffer) {
        let acquired = inFlight.withLock { busy -> Bool in
            if busy { return false }
            busy = true
            return true
        }
        guard acquired else { return }
        queue.async {
            defer { self.inFlight.withLock { $0 = false } }
            self.process(frame)
        }
    }

    private func process(_ pb: CVPixelBuffer) {
        guard let engine, !paused else { return }
        let t0 = CACurrentMediaTime()
        engine.preprocess(pb)
        let t1 = CACurrentMediaTime()
        guard let d = try? engine.infer() else { return }
        let t2 = CACurrentMediaTime()
        if logMode {
            // Burst of 4 extra inferences: if they get progressively faster, the live slowdown is clock ramp-up.
            var burst: [Double] = []
            var tPrev = t2
            for _ in 0..<4 {
                _ = try? engine.infer()
                let tNow = CACurrentMediaTime()
                burst.append((tNow - tPrev) * 1000)
                tPrev = tNow
            }
            if emaBurst.isEmpty { emaBurst = burst } else {
                for i in 0..<4 { emaBurst[i] = emaBurst[i] * 0.9 + burst[i] * 0.1 }
            }
            if tPrev - lastLog >= 1 {
                lastLog = tPrev
                let b = emaBurst.map { String(format: "%.2f", $0) }.joined(separator: " → ")
                print(String(format: "LIVE %@ %@ fps=%.1f pre=%.2fms infer#1=%.2fms then %@ ms",
                             engine.variant.rawValue, engine.compute.rawValue, fps, emaPre, emaInfer, b))
            }
        }

        latest = d
        lastFrame = pb
        let a = 0.1
        emaPre = emaPre == 0 ? (t1 - t0) * 1000 : emaPre * (1 - a) + (t1 - t0) * 1000 * a
        emaInfer = emaInfer == 0 ? (t2 - t1) * 1000 : emaInfer * (1 - a) + (t2 - t1) * 1000 * a
        frames += 1
        if t2 - windowStart >= 1 {
            fps = Double(frames) / (t2 - windowStart)
            frames = 0
            windowStart = t2
        }

        var scores: [UUID: Float] = [:]
        var best: Match?
        for p in places {
            var s: Float = 0
            vDSP_dotpr(p.descriptor, 1, d, 1, &s, vDSP_Length(d.count))
            scores[p.id] = s
            if best == nil || s > best!.score { best = Match(id: p.id, score: s) }
        }
        onStats?(FrameStats(inferMs: emaInfer, preMs: emaPre, fps: fps, scores: scores, best: best))
    }

    func registerPlace(completion: @escaping (Place?) -> Void) {
        queue.async {
            guard let d = self.latest, let img = self.engine?.thumbnail() else { completion(nil); return }
            let p = Place(name: "Place \(self.places.count + 1)", descriptor: d, thumbnail: img)
            self.places.append(p)
            completion(p)
        }
    }

    /// `--embed`: run the bundled test_image.png through the exact live pipeline and print the descriptor
    /// so it can be compared with the PC (PyTorch / CoreML-on-Mac) result.
    func embedTestImage() {
        queue.async {
            guard let engine = self.engine,
                  let ui = UIImage(named: "test_image"), let cg = ui.cgImage,
                  let pb = makeSyntheticFrame(width: cg.width, height: cg.height) else {
                print("EMBED failed: engine/image missing"); return
            }
            CVPixelBufferLockBaseAddress(pb, [])
            let info = CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
            if let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb), width: cg.width, height: cg.height,
                                   bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
                                   space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: info) {
                ctx.draw(cg, in: CGRect(x: 0, y: 0, width: cg.width, height: cg.height))
            }
            CVPixelBufferUnlockBaseAddress(pb, [])
            engine.preprocess(pb)
            guard let d = try? engine.infer() else { print("EMBED infer failed"); return }
            let data = d.withUnsafeBufferPointer { Data(buffer: $0) }
            print("EMBED \(engine.variant.rawValue) \(engine.compute.rawValue) dim=\(d.count) first=\(d.prefix(4).map { String(format: "%.5f", $0) })")
            print("EMBEDB64 \(data.base64EncodedString())")
        }
    }

    func clearPlaces() {
        queue.async { self.places.removeAll() }
    }

    func runBenchmark(iterations: Int, progress: @escaping (BenchResult) -> Void, done: @escaping () -> Void) {
        queue.async {
            self.paused = true
            self.engine = nil
            MixVPREngine.benchmark(iterations: iterations, pixelBuffer: self.lastFrame, progress: progress)
            self.engine = try? MixVPREngine(variant: self.variant, compute: self.compute)
            self.paused = false
            done()
        }
    }
}
