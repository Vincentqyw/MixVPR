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
    var fps = 0.0
    var placeScores: [UUID: Float] = [:]
    var imageScores: [UUID: Float] = [:]
    var best: Match?
}

/// Owns the engine, the open session and every per-frame computation on one serial queue.
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
    private var session: Session?
    private var frames = 0
    private var windowStart = CACurrentMediaTime()
    private var fps = 0.0
    private var emaInfer = 0.0
    private var emaPre = 0.0
    private let logMode = CommandLine.arguments.contains("--log")
    private var emaBurst: [Double] = []
    private var lastLog = CACurrentMediaTime()

    /// All callbacks fire on the worker queue.
    var onStats: ((FrameStats) -> Void)?
    var onStatus: ((String) -> Void)?
    var onSession: ((Session?) -> Void)?
    var onSessions: (([SessionSummary]) -> Void)?

    // MARK: sessions

    func bootstrap(lastSessionID: UUID?) {
        queue.async {
            var list = SessionStore.list()
            if list.isEmpty {
                SessionStore.save(Session(id: UUID(), name: "Session 1", createdAt: Date(), places: []))
                list = SessionStore.list()
            }
            let id = lastSessionID.flatMap { id in list.contains { $0.id == id } ? id : nil } ?? list[0].id
            self.session = SessionStore.load(id)
            self.onSessions?(list)
            self.onSession?(self.session)
            self.reindex()
        }
    }

    func openSession(_ id: UUID) {
        queue.async {
            guard id != self.session?.id, let s = SessionStore.load(id) else { return }
            self.session = s
            self.onSession?(s)
            self.reindex()
        }
    }

    func createSession() {
        queue.async {
            let n = SessionStore.list().count + 1
            let s = Session(id: UUID(), name: "Session \(n)", createdAt: Date(), places: [])
            SessionStore.save(s)
            self.session = s
            self.onSessions?(SessionStore.list())
            self.onSession?(s)
        }
    }

    func renameSession(_ id: UUID, to name: String) {
        queue.async {
            if self.session?.id == id {
                self.session?.name = name
                self.persist()
            } else if var s = SessionStore.load(id) {
                s.name = name
                SessionStore.save(s)
            }
            self.onSessions?(SessionStore.list())
        }
    }

    func deleteSession(_ id: UUID) {
        queue.async {
            SessionStore.delete(id)
            var list = SessionStore.list()
            if self.session?.id == id {
                if list.isEmpty {
                    SessionStore.save(Session(id: UUID(), name: "Session 1", createdAt: Date(), places: []))
                    list = SessionStore.list()
                }
                self.session = SessionStore.load(list[0].id)
                self.onSession?(self.session)
            }
            self.onSessions?(list)
        }
    }

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
        guard let engine, var s = session else { return }
        let fam = model.family
        var missing = 0
        for p in s.places { for i in p.images where i.descriptors[fam] == nil { missing += 1 } }
        guard missing > 0 else { return }
        var done = 0
        for pi in s.places.indices {
            for ii in s.places[pi].images.indices where s.places[pi].images[ii].descriptors[fam] == nil {
                done += 1
                onStatus?("Indexing \(done)/\(missing) for \(fam.rawValue)…")
                if let d = engine.embed(image: s.places[pi].images[ii].thumbnail) {
                    s.places[pi].images[ii].descriptors[fam] = d
                }
            }
        }
        session = s
        persist()
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

        var placeScores: [UUID: Float] = [:]
        var imageScores: [UUID: Float] = [:]
        var best: Match?
        let fam = model.family
        for p in session?.places ?? [] {
            var pbest: Float = -1
            for img in p.images {
                guard let v = img.descriptors[fam], v.count == d.count else { continue }
                var s: Float = 0
                vDSP_dotpr(v, 1, d, 1, &s, vDSP_Length(d.count))
                imageScores[img.id] = s
                pbest = max(pbest, s)
            }
            guard pbest > -1 else { continue }
            placeScores[p.id] = pbest
            if best == nil || pbest > best!.score { best = Match(id: p.id, score: pbest) }
        }
        onStats?(FrameStats(inferMs: emaInfer, preMs: emaPre, fps: fps,
                            placeScores: placeScores, imageScores: imageScores, best: best))
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

    // MARK: places & images

    /// Captures the current frame into `placeID`, or into a new place when nil. Returns the place id.
    func capture(into placeID: UUID?, completion: @escaping (UUID?) -> Void) {
        queue.async {
            guard let d = self.latest, let engine = self.engine, let img = engine.thumbnail(),
                  let jpeg = img.jpegData(compressionQuality: 0.85), self.session != nil else { completion(nil); return }
            let image = PlaceImage(id: UUID(), descriptors: [self.model.family: d], jpeg: jpeg, thumbnail: img, createdAt: Date())
            var target = placeID
            if let id = placeID, let i = self.session!.places.firstIndex(where: { $0.id == id }) {
                self.session!.places[i].images.append(image)
            } else {
                let n = self.session!.places.count + 1
                let p = Place(id: UUID(), name: "Place \(n)", images: [image])
                self.session!.places.append(p)
                target = p.id
            }
            self.persist()
            completion(target)
        }
    }

    func renamePlace(_ id: UUID, to name: String) {
        queue.async {
            guard let i = self.session?.places.firstIndex(where: { $0.id == id }) else { return }
            self.session!.places[i].name = name
            self.persist()
        }
    }

    func deletePlace(_ id: UUID) {
        queue.async {
            self.session?.places.removeAll { $0.id == id }
            self.persist()
        }
    }

    func deleteImage(_ imageID: UUID, from placeID: UUID) {
        queue.async {
            guard let i = self.session?.places.firstIndex(where: { $0.id == placeID }) else { return }
            self.session!.places[i].images.removeAll { $0.id == imageID }
            if self.session!.places[i].images.isEmpty { self.session!.places.remove(at: i) }
            self.persist()
        }
    }

    private func persist() {
        guard let s = session else { return }
        SessionStore.save(s)
        onSession?(s)
        onSessions?(SessionStore.list())
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
