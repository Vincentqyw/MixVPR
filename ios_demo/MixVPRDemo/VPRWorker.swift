import Accelerate
import CoreVideo
import Foundation
import os
import QuartzCore
import UIKit

struct Match {
    let id: UUID
    let score: Float
}

struct FrameStats {
    var inferMs = 0.0
    var preMs = 0.0
    var cameraFps = 0.0
    var sessionScores: [UUID: Float] = [:]
    var imageScores: [UUID: Float] = [:]
    var best: Match?          // best-matching session
}

/// Owns the engine, all sessions and every per-frame computation on one serial queue.
final class VPRWorker {
    private let queue = DispatchQueue(label: "vpr.work", qos: .userInteractive)
    private let inFlight = OSAllocatedUnfairLock(initialState: false)
    private let benchStop = OSAllocatedUnfairLock(initialState: false)

    // Everything below is only touched on `queue`.
    private var engine: VPREngine?
    private var model: VPRModel = .mixvprFP16
    private var compute: ComputeChoice = .all
    private var paused = false
    private var latest: [Float]?
    private var lastFrame: CVPixelBuffer?
    private var sessions: [Session] = []
    private var currentID: UUID?
    private var frames = 0
    private var windowStart = CACurrentMediaTime()
    private var fps = 0.0
    private var emaInfer = 0.0
    private var emaPre = 0.0
    // Display smoothing: scores are EMA-filtered, the best match only switches after a
    // challenger has led by `switchMargin` for `switchFrames` frames, and the UI is fed at ~4 Hz.
    private var smoothSession: [UUID: Float] = [:]
    private var smoothImage: [UUID: Float] = [:]
    private var stableBest: Match?
    private var challenger: (id: UUID, frames: Int)?
    private var lastEmit: CFTimeInterval = 0
    private let scoreAlpha: Float = 0.35
    private let switchMargin: Float = 0.03
    private let switchFrames = 4
    private let emitInterval: CFTimeInterval = 0.25
    private let logMode = CommandLine.arguments.contains("--log")
    private var emaBurst: [Double] = []
    private var lastLog = CACurrentMediaTime()

    /// All callbacks fire on the worker queue.
    var onStats: ((FrameStats) -> Void)?
    var onStatus: ((String) -> Void)?
    var onSessions: (([Session], UUID?) -> Void)?

    private var currentIndex: Int? { sessions.firstIndex { $0.id == currentID } }

    // MARK: sessions

    func bootstrap(lastSessionID: UUID?) {
        queue.async {
            self.sessions = SessionStore.loadAll()
            if self.sessions.isEmpty { self.sessions = [self.newSession()] }
            self.currentID = lastSessionID.flatMap { id in self.sessions.contains { $0.id == id } ? id : nil } ?? self.sessions.last!.id
            self.emit()
            self.reindex()
        }
    }

    private func newSession() -> Session {
        let s = Session(id: UUID(), name: "Place \(sessions.count + 1)", createdAt: Date(), images: [])
        SessionStore.save(s)
        return s
    }

    func openSession(_ id: UUID) {
        queue.async {
            guard self.sessions.contains(where: { $0.id == id }) else { return }
            self.currentID = id
            self.emit()
        }
    }

    func createSession() {
        queue.async {
            let s = self.newSession()
            self.sessions.append(s)
            self.currentID = s.id
            self.emit()
        }
    }

    func renameSession(_ id: UUID, to name: String) {
        queue.async {
            guard let i = self.sessions.firstIndex(where: { $0.id == id }) else { return }
            self.sessions[i].name = name
            SessionStore.save(self.sessions[i])
            self.emit()
        }
    }

    func deleteSession(_ id: UUID) {
        queue.async {
            SessionStore.delete(id)
            self.sessions.removeAll { $0.id == id }
            if self.sessions.isEmpty { self.sessions = [self.newSession()] }
            if self.currentID == id { self.currentID = self.sessions.last!.id }
            self.emit()
        }
    }

    private func emit() { onSessions?(sessions, currentID) }

    // MARK: engine

    func load(model: VPRModel, compute: ComputeChoice) {
        queue.async {
            self.model = model
            self.compute = compute
            self.engine = nil
            self.latest = nil
            self.onStatus?("Loading \(model.title)…")
            do {
                let e = try VPREngine(model: model, compute: compute)
                self.engine = e
                self.emaInfer = 0; self.emaPre = 0; self.frames = 0; self.fps = 0
                self.windowStart = CACurrentMediaTime()
                self.smoothSession = [:]; self.smoothImage = [:]; self.stableBest = nil; self.challenger = nil
                self.onStatus?(String(format: "%@ · %@ · ready in %.0f ms", model.title, compute.rawValue, e.loadMs))
            } catch {
                self.onStatus?("Load failed: \(error)")
            }
            self.onStats?(FrameStats())
            self.reindex()
        }
    }

    /// Computes descriptors for the current model family for every stored image that lacks one.
    private func reindex() {
        guard let engine else { return }
        let fam = model.family
        let missing = sessions.reduce(0) { $0 + $1.images.filter { $0.descriptors[fam] == nil }.count }
        guard missing > 0 else { return }
        var done = 0
        for si in sessions.indices {
            var changed = false
            for ii in sessions[si].images.indices where sessions[si].images[ii].descriptors[fam] == nil {
                done += 1
                onStatus?("Indexing \(done)/\(missing) for \(fam.rawValue)…")
                if let d = engine.embed(image: sessions[si].images[ii].thumbnail) {
                    sessions[si].images[ii].descriptors[fam] = d
                    changed = true
                }
            }
            if changed { SessionStore.save(sessions[si]) }
        }
        emit()
        onStatus?(String(format: "%@ · %@", model.title, compute.rawValue))
    }

    // MARK: frames

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
        if logMode { logBurst(engine, from: t2) }

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

        // Raw cosine scores for this frame, then EMA-smoothed for display.
        var sessionScores: [UUID: Float] = [:]
        var imageScores: [UUID: Float] = [:]
        let fam = model.family
        for s in sessions {
            var sbest: Float = -1
            for img in s.images {
                guard let v = img.descriptors[fam], v.count == d.count else { continue }
                var score: Float = 0
                vDSP_dotpr(v, 1, d, 1, &score, vDSP_Length(d.count))
                let sm = smoothImage[img.id].map { $0 + (score - $0) * scoreAlpha } ?? score
                smoothImage[img.id] = sm
                imageScores[img.id] = sm
                sbest = max(sbest, score)
            }
            guard sbest > -1 else { continue }
            let sm = smoothSession[s.id].map { $0 + (sbest - $0) * scoreAlpha } ?? sbest
            smoothSession[s.id] = sm
            sessionScores[s.id] = sm
        }
        smoothSession = smoothSession.filter { sessionScores[$0.key] != nil }
        smoothImage = smoothImage.filter { imageScores[$0.key] != nil }

        // Hysteresis on the winner.
        let leader = sessionScores.max { $0.value < $1.value }
        if let cur = stableBest, let curScore = sessionScores[cur.id] {
            if let leader, leader.key != cur.id, leader.value > curScore + switchMargin {
                challenger = (leader.key, (challenger?.id == leader.key ? challenger!.frames : 0) + 1)
                if challenger!.frames >= switchFrames { stableBest = Match(id: leader.key, score: leader.value); challenger = nil }
                else { stableBest = Match(id: cur.id, score: curScore) }
            } else {
                challenger = nil
                stableBest = Match(id: cur.id, score: curScore)
            }
        } else {
            stableBest = leader.map { Match(id: $0.key, score: $0.value) }
            challenger = nil
        }

        if t2 - lastEmit >= emitInterval {
            lastEmit = t2
            onStats?(FrameStats(inferMs: emaInfer, preMs: emaPre, cameraFps: fps,
                                sessionScores: sessionScores, imageScores: imageScores, best: stableBest))
        }
    }

    private func logBurst(_ engine: VPREngine, from t: CFTimeInterval) {
        var burst: [Double] = []
        var tPrev = t
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
                         engine.model.rawValue, engine.compute.rawValue, fps, emaPre, emaInfer, b))
        }
    }

    // MARK: images

    /// Captures the current frame into the current session.
    func capture(completion: @escaping (Bool) -> Void) {
        queue.async {
            guard let d = self.latest, let engine = self.engine, let img = engine.thumbnail(),
                  let jpeg = img.jpegData(compressionQuality: 0.85), let i = self.currentIndex else { completion(false); return }
            self.sessions[i].images.append(PlaceImage(id: UUID(), descriptors: [self.model.family: d], jpeg: jpeg,
                                                      thumbnail: img, createdAt: Date()))
            SessionStore.save(self.sessions[i])
            self.emit()
            completion(true)
        }
    }

    func deleteImage(_ imageID: UUID) {
        queue.async {
            guard let i = self.sessions.firstIndex(where: { $0.images.contains { $0.id == imageID } }) else { return }
            self.sessions[i].images.removeAll { $0.id == imageID }
            SessionStore.save(self.sessions[i])
            self.emit()
        }
    }

    // MARK: diagnostics

    /// `--embed`: run the bundled test_<N>.png through every bundled model × compute unit and print the
    /// descriptors (base64 float32) so they can be compared with the PC.
    func embedTestImages() {
        queue.async {
            self.paused = true
            for model in VPRModel.bundled {
                guard let ui = UIImage(named: "test_\(model.inputSize)"), let cg = ui.cgImage,
                      let pb = makePixelBuffer(from: cg) else { continue }
                for compute in ComputeChoice.allCases {
                    autoreleasepool {
                        guard let e = try? VPREngine(model: model, compute: compute) else {
                            print("EMBED \(model.rawValue) \(compute.rawValue) load failed"); return
                        }
                        e.preprocess(pb)
                        guard let d = try? e.infer() else { return }
                        let data = d.withUnsafeBufferPointer { Data(buffer: $0) }
                        print("EMBED \(model.rawValue) \(compute.rawValue) dim=\(d.count)")
                        print("EMBEDB64 \(data.base64EncodedString())")
                    }
                }
            }
            print("EMBED DONE")
            self.engine = try? VPREngine(model: self.model, compute: self.compute)
            self.paused = false
        }
    }

    // MARK: benchmark

    func runBenchmark(models: [VPRModel], progress: @escaping (BenchResult) -> Void, done: @escaping (Bool) -> Void) {
        benchStop.withLock { $0 = false }
        queue.async {
            self.paused = true
            self.engine = nil
            VPREngine.benchmark(models: models, pixelBuffer: self.lastFrame,
                                shouldStop: { self.benchStop.withLock { $0 } }, progress: progress)
            let stopped = self.benchStop.withLock { $0 }
            self.engine = try? VPREngine(model: self.model, compute: self.compute)
            self.paused = false
            done(stopped)
        }
    }

    /// Thread-safe; takes effect at the next iteration boundary.
    func stopBenchmark() { benchStop.withLock { $0 = true } }
}
