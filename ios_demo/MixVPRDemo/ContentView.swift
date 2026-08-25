import SwiftUI

struct ContentView: View {
    @StateObject private var state = AppState()

    var body: some View {
        VStack(spacing: 12) {
            statsBar
            preview
            pickers
            placesStrip
            buttons
        }
        .padding()
        .background(Color(.systemBackground))
        .sheet(isPresented: $state.showBench) { BenchSheet(state: state) }
        .task { await state.start() }
    }

    private var statsBar: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 2) {
                Text("MixVPR on-device").font(.headline)
                Text(state.status).font(.caption).foregroundStyle(.secondary).lineLimit(2)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 2) {
                Text(String(format: "%.0f FPS", state.stats.fps))
                    .font(.system(.title2, design: .monospaced).bold())
                Text(String(format: "infer %.1f ms · pre %.1f ms", state.stats.inferMs, state.stats.preMs))
                    .font(.caption.monospaced()).foregroundStyle(.secondary)
            }
        }
    }

    private var preview: some View {
        ZStack(alignment: .bottom) {
            if let err = state.cameraError {
                Color.black
                Text(err).foregroundStyle(.white).padding().frame(maxHeight: .infinity)
            } else {
                CameraPreview(session: state.camera.session)
            }
            matchBadge
        }
        .aspectRatio(1, contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    @ViewBuilder private var matchBadge: some View {
        if let best = state.stats.best, let place = state.places.first(where: { $0.id == best.id }) {
            let s = best.score
            HStack(spacing: 8) {
                Image(uiImage: place.thumbnail).resizable().scaledToFill()
                    .frame(width: 36, height: 36).clipShape(RoundedRectangle(cornerRadius: 6))
                Text(s >= 0.70 ? "📍 \(place.name)" : s >= 0.55 ? "❓ maybe \(place.name)" : "Unknown place")
                    .font(.headline)
                Spacer()
                Text(String(format: "%.2f", s)).font(.title3.monospaced().bold()).foregroundStyle(scoreColor(s))
            }
            .padding(10)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
            .padding(10)
        } else if state.places.isEmpty {
            Text("Point at a place and tap “Register place”")
                .font(.subheadline)
                .padding(8)
                .background(.ultraThinMaterial, in: Capsule())
                .padding(10)
        }
    }

    private var pickers: some View {
        HStack {
            Picker("Model", selection: $state.variant) {
                ForEach(ModelVariant.allCases) { Text($0.rawValue).tag($0) }
            }.pickerStyle(.segmented)
            Picker("Compute", selection: $state.compute) {
                ForEach(ComputeChoice.allCases) { Text($0.rawValue).tag($0) }
            }.pickerStyle(.segmented)
        }
    }

    private var placesStrip: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 10) {
                ForEach(state.places) { p in
                    let s = state.stats.scores[p.id] ?? 0
                    VStack(spacing: 4) {
                        Image(uiImage: p.thumbnail).resizable().scaledToFill()
                            .frame(width: 72, height: 72)
                            .clipShape(RoundedRectangle(cornerRadius: 10))
                            .overlay(RoundedRectangle(cornerRadius: 10)
                                .stroke(scoreColor(s), lineWidth: state.stats.best?.id == p.id ? 3 : 1))
                        Text(p.name).font(.caption2)
                        Text(String(format: "%.2f", s)).font(.caption2.monospaced()).foregroundStyle(scoreColor(s))
                    }
                }
                if state.places.isEmpty {
                    Text("No places registered yet").font(.caption).foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal, 4)
        }
        .frame(height: 110)
    }

    private var buttons: some View {
        HStack {
            Button { state.registerPlace() } label: {
                Label("Register place", systemImage: "mappin.and.ellipse").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            Button(role: .destructive) { state.clearPlaces() } label: {
                Image(systemName: "trash")
            }
            .buttonStyle(.bordered)
            Button { state.runBenchmark() } label: {
                Label("Bench", systemImage: "speedometer")
            }
            .buttonStyle(.bordered)
        }
    }

    private func scoreColor(_ s: Float) -> Color {
        s >= 0.70 ? .green : s >= 0.55 ? .yellow : .secondary
    }
}

struct BenchSheet: View {
    @ObservedObject var state: AppState

    var body: some View {
        NavigationStack {
            List {
                Section("\(deviceModelIdentifier()) · iOS \(UIDevice.current.systemVersion) · 50 iterations") {
                    ForEach(state.benchResults) { r in
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("\(r.variant.rawValue) · \(r.compute.rawValue)").font(.headline)
                                if let e = r.error {
                                    Text(e).font(.caption).foregroundStyle(.red)
                                } else {
                                    Text(String(format: "median %.2f ms · p95 %.2f ms · pre %.2f ms · load %.0f ms",
                                                r.medianMs, r.p95Ms, r.preMs, r.loadMs))
                                        .font(.caption.monospaced()).foregroundStyle(.secondary)
                                }
                            }
                            Spacer()
                            Text(String(format: "%.0f FPS", r.fps)).font(.title3.monospaced().bold())
                        }
                    }
                    if state.benchRunning {
                        HStack { ProgressView(); Text("Running…").padding(.leading, 8) }
                    }
                }
            }
            .navigationTitle("Benchmark")
        }
    }
}
