import Foundation
import SwiftUI
import UIKit

enum MatchLevel { case none, weak, confident }

@MainActor
final class AppState: ObservableObject {
    @Published var model: VPRModel { didSet { defaults.set(model.rawValue, forKey: "model"); reload() } }
    @Published var computes: [ModelFamily: ComputeChoice] { didSet { saveComputes(); reload() } }
    @Published var thresholds: [ModelFamily: Float] { didSet { saveThresholds() } }
    @Published var crossSession: Bool { didSet { defaults.set(crossSession, forKey: "crossSession"); worker.setCrossSession(crossSession) } }
    @Published var status = "Starting…"
    @Published var stats = FrameStats()
    @Published var sessions: [Session] = []
    @Published var currentID: UUID?
    @Published var detailImageID: UUID?
    @Published var benchResults: [BenchResult] = []
    @Published var benchRunning = false
    @Published var benchStopped = false
    @Published var showBench = false
    @Published var showSettings = false
    @Published var showSessions = false
    @Published var cameraError: String?
    @Published var flash = false
    @Published var renamingID: UUID?
    @Published var renameText = ""

    let camera = CameraManager()
    let worker = VPRWorker()
    private let defaults = UserDefaults.standard
    private var started = false
    private static var launchFlagsHandled = false
    private let impact = UIImpactFeedbackGenerator(style: .medium)
    private let notify = UINotificationFeedbackGenerator()

    init() {
        let bundled = VPRModel.bundled
        let saved = defaults.string(forKey: "model").flatMap(VPRModel.init(rawValue:))
        model = saved.flatMap { bundled.contains($0) ? $0 : nil } ?? bundled.first ?? .mixvprFP16
        var c: [ModelFamily: ComputeChoice] = [:]
        var t: [ModelFamily: Float] = [:]
        for f in ModelFamily.allCases {
            c[f] = defaults.string(forKey: "compute.\(f.rawValue)").flatMap(ComputeChoice.init(rawValue:)) ?? f.defaultCompute
            let v = defaults.float(forKey: "threshold.\(f.rawValue)")
            t[f] = v > 0 ? v : f.defaultThreshold
        }
        computes = c
        thresholds = t
        crossSession = defaults.bool(forKey: "crossSession")

        worker.onStatus = { [weak self] s in Task { @MainActor in self?.status = s } }
        worker.onStats = { [weak self] st in Task { @MainActor in self?.stats = st } }
        worker.onSessions = { [weak self] list, current in
            Task { @MainActor in
                guard let self else { return }
                self.sessions = list
                self.currentID = current
                if let current { self.defaults.set(current.uuidString, forKey: "session") }
                if let d = self.detailImageID, self.locate(image: d) == nil { self.detailImageID = nil }
            }
        }
        camera.onFrame = { [weak self] pb in self?.worker.handle(frame: pb) }
    }

    // MARK: derived

    var session: Session? { currentID.flatMap { id in sessions.first { $0.id == id } } }
    /// Finds an image in any session: (session, image, 1-based index within the session).
    func locate(image id: UUID) -> (session: Session, image: PlaceImage, index: Int)? {
        for s in sessions {
            if let i = s.images.firstIndex(where: { $0.id == id }) { return (s, s.images[i], i + 1) }
        }
        return nil
    }
    var detail: (session: Session, image: PlaceImage, index: Int)? { detailImageID.flatMap(locate(image:)) }
    var bestMatch: (session: Session, image: PlaceImage, index: Int)? { stats.best.flatMap { locate(image: $0.imageID) } }
    var compute: ComputeChoice {
        get { computes[model.family] ?? model.family.defaultCompute }
        set { computes[model.family] = newValue }
    }
    var threshold: Float {
        get { thresholds[model.family] ?? model.family.defaultThreshold }
        set { thresholds[model.family] = newValue }
    }
    var hasAnyImages: Bool { sessions.contains { !$0.images.isEmpty } }
    func level(for score: Float?) -> MatchLevel {
        guard let s = score else { return .none }
        if s >= threshold { return .confident }
        if s >= threshold - 0.15 { return .weak }
        return .none
    }
    var matchLevel: MatchLevel { level(for: stats.best?.score) }
    func color(for level: MatchLevel) -> Color {
        switch level {
        case .confident: return Color(red: 0.30, green: 0.87, blue: 0.55)
        case .weak: return Color(red: 0.98, green: 0.78, blue: 0.30)
        case .none: return .white.opacity(0.55)
        }
    }
    var matchColor: Color { color(for: matchLevel) }

    // MARK: lifecycle

    func start() async {
        guard !started else { return }
        started = true
        worker.setCrossSession(crossSession)
        worker.bootstrap(lastSessionID: defaults.string(forKey: "session").flatMap(UUID.init(uuidString:)))
        reload()
        if !Self.launchFlagsHandled {
            Self.launchFlagsHandled = true
            if CommandLine.arguments.contains("--bench") { runBenchmark() }
            if CommandLine.arguments.contains("--embed") { worker.embedTestImages() }
        }
        guard await CameraManager.requestAccess() else {
            cameraError = "Camera access is off.\nEnable it in Settings › PlaceLens."
            return
        }
        do { try camera.start() } catch { cameraError = "\(error)" }
    }

    private func reload() { worker.load(model: model, compute: compute) }
    private func saveComputes() { for (f, c) in computes { defaults.set(c.rawValue, forKey: "compute.\(f.rawValue)") } }
    private func saveThresholds() { for (f, t) in thresholds { defaults.set(t, forKey: "threshold.\(f.rawValue)") } }

    // MARK: actions

    /// Shutter: adds the current frame to the current session.
    func capture() {
        impact.impactOccurred()
        flash = true
        withAnimation(.easeOut(duration: 0.4)) { flash = false }
        worker.capture { [weak self] ok in
            Task { @MainActor in self?.notify.notificationOccurred(ok ? .success : .error) }
        }
    }

    func newSession() {
        impact.impactOccurred()
        worker.createSession()
    }

    func beginRename(_ id: UUID) {
        renameText = sessions.first { $0.id == id }?.name ?? ""
        renamingID = id
    }

    func commitRename() {
        defer { renamingID = nil }
        let name = renameText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, let id = renamingID else { return }
        worker.renameSession(id, to: name)
    }

    // MARK: benchmark

    func runBenchmark() {
        guard !benchRunning else { showBench = true; return }
        benchRunning = true
        benchStopped = false
        benchResults = []
        showSettings = false
        showBench = true
        print("BENCH device=\(deviceModelIdentifier()) ios=\(UIDevice.current.systemVersion)")
        worker.runBenchmark(models: VPRModel.bundled, progress: { [weak self] r in
            print("BENCH \(r.line)")
            Task { @MainActor in self?.benchResults.append(r) }
        }, done: { [weak self] stopped in
            print(stopped ? "BENCH STOPPED" : "BENCH DONE")
            Task { @MainActor in self?.benchRunning = false; self?.benchStopped = stopped }
        })
    }

    func stopBenchmark() { worker.stopBenchmark() }
}
