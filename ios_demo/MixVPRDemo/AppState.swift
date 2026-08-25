import Foundation
import SwiftUI

@MainActor
final class AppState: ObservableObject {
    @Published var variant: ModelVariant = .fp16 { didSet { worker.load(variant: variant, compute: compute) } }
    @Published var compute: ComputeChoice = .all { didSet { worker.load(variant: variant, compute: compute) } }
    @Published var status = "Starting…"
    @Published var stats = FrameStats()
    @Published var places: [Place] = []
    @Published var benchResults: [BenchResult] = []
    @Published var benchRunning = false
    @Published var showBench = false
    @Published var cameraError: String?

    let camera = CameraManager()
    let worker = VPRWorker()
    private var started = false

    init() {
        worker.onStatus = { [weak self] s in Task { @MainActor in self?.status = s } }
        worker.onStats = { [weak self] st in Task { @MainActor in self?.stats = st } }
        camera.onFrame = { [weak self] pb in self?.worker.handle(frame: pb) }
    }

    func start() async {
        guard !started else { return }
        started = true
        worker.load(variant: variant, compute: compute)
        if CommandLine.arguments.contains("--bench") { runBenchmark() }
        guard await CameraManager.requestAccess() else {
            cameraError = "Camera access denied — enable it in Settings › MixVPR."
            return
        }
        do { try camera.start() } catch { cameraError = "\(error)" }
    }

    func registerPlace() {
        worker.registerPlace { [weak self] p in
            guard let p else { return }
            Task { @MainActor in self?.places.append(p) }
        }
    }

    func clearPlaces() {
        worker.clearPlaces()
        places.removeAll()
        stats.scores = [:]
        stats.best = nil
    }

    func runBenchmark() {
        guard !benchRunning else { return }
        benchRunning = true
        benchResults = []
        showBench = true
        print("BENCH device=\(deviceModelIdentifier()) ios=\(UIDevice.current.systemVersion)")
        worker.runBenchmark(iterations: 50, progress: { [weak self] r in
            print("BENCH \(r.line)")
            Task { @MainActor in self?.benchResults.append(r) }
        }, done: { [weak self] in
            print("BENCH DONE")
            Task { @MainActor in self?.benchRunning = false }
        })
    }
}
