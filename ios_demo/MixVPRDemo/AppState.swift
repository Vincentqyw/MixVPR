import Foundation
import SwiftUI
import UIKit

enum MatchLevel { case none, weak, confident }

enum RenameTarget: Identifiable {
    case session(UUID), place(UUID)
    var id: UUID { switch self { case .session(let i), .place(let i): return i } }
}

@MainActor
final class AppState: ObservableObject {
    @Published var model: VPRModel { didSet { defaults.set(model.rawValue, forKey: "model"); reload() } }
    @Published var computes: [ModelFamily: ComputeChoice] { didSet { saveComputes(); reload() } }
    @Published var thresholds: [ModelFamily: Float] { didSet { saveThresholds() } }
    @Published var status = "Starting…"
    @Published var stats = FrameStats()
    @Published var session: Session?
    @Published var sessions: [SessionSummary] = []
    @Published var selectedPlaceID: UUID?
    @Published var detailPlaceID: UUID?
    @Published var benchResults: [BenchResult] = []
    @Published var benchRunning = false
    @Published var benchStopped = false
    @Published var showBench = false
    @Published var showSettings = false
    @Published var showSessions = false
    @Published var cameraError: String?
    @Published var flash = false
    @Published var renameTarget: RenameTarget?
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

        worker.onStatus = { [weak self] s in Task { @MainActor in self?.status = s } }
        worker.onStats = { [weak self] st in Task { @MainActor in self?.stats = st } }
        worker.onSession = { [weak self] s in
            Task { @MainActor in
                guard let self else { return }
                withAnimation(.spring(duration: 0.35)) { self.session = s }
                if let s { self.defaults.set(s.id.uuidString, forKey: "session") }
                if let sel = self.selectedPlaceID, !(s?.places.contains { $0.id == sel } ?? false) { self.selectedPlaceID = nil }
                if let det = self.detailPlaceID, !(s?.places.contains { $0.id == det } ?? false) { self.detailPlaceID = nil }
            }
        }
        worker.onSessions = { [weak self] list in Task { @MainActor in self?.sessions = list } }
        camera.onFrame = { [weak self] pb in self?.worker.handle(frame: pb) }
    }

    // MARK: derived

    var places: [Place] { session?.places ?? [] }
    var selectedPlace: Place? { selectedPlaceID.flatMap { id in places.first { $0.id == id } } }
    var detailPlace: Place? { detailPlaceID.flatMap { id in places.first { $0.id == id } } }
    var compute: ComputeChoice {
        get { computes[model.family] ?? model.family.defaultCompute }
        set { computes[model.family] = newValue }
    }
    var threshold: Float {
        get { thresholds[model.family] ?? model.family.defaultThreshold }
        set { thresholds[model.family] = newValue }
    }
    var bestPlace: Place? {
        guard let b = stats.best else { return nil }
        return places.first { $0.id == b.id }
    }
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

    // MARK: capture

    /// Shutter: adds a view to the selected place, or creates a new place (which becomes selected).
    func capture(into placeID: UUID? = nil) {
        impact.impactOccurred()
        flash = true
        withAnimation(.easeOut(duration: 0.4)) { flash = false }
        worker.capture(into: placeID ?? selectedPlaceID) { [weak self] id in
            Task { @MainActor in
                guard let self else { return }
                if let id {
                    self.notify.notificationOccurred(.success)
                    if placeID == nil { self.selectedPlaceID = id }
                } else {
                    self.notify.notificationOccurred(.error)
                }
            }
        }
    }

    func togglePlace(_ p: Place) {
        if selectedPlaceID == p.id { detailPlaceID = p.id } else { selectedPlaceID = p.id }
    }

    // MARK: rename

    func beginRename(_ target: RenameTarget) {
        switch target {
        case .session(let id): renameText = sessions.first { $0.id == id }?.name ?? session?.name ?? ""
        case .place(let id): renameText = places.first { $0.id == id }?.name ?? ""
        }
        renameTarget = target
    }

    func commitRename() {
        defer { renameTarget = nil }
        let name = renameText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, let t = renameTarget else { return }
        switch t {
        case .session(let id): worker.renameSession(id, to: name)
        case .place(let id): worker.renamePlace(id, to: name)
        }
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
