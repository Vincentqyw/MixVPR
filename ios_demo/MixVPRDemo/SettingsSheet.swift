import SwiftUI

struct SettingsSheet: View {
    @ObservedObject var state: AppState
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section("Model") {
                    ForEach(VPRModel.bundled) { m in
                        Button { state.model = m } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(m.title).foregroundStyle(.primary)
                                    Text(m.subtitle).font(.caption).foregroundStyle(.secondary)
                                }
                                Spacer()
                                if m == state.model { Image(systemName: "checkmark").foregroundStyle(.tint) }
                            }
                        }
                    }
                }

                Section {
                    Picker("Compute", selection: Binding(get: { state.compute }, set: { state.compute = $0 })) {
                        ForEach(ComputeChoice.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    if state.model.family == .megaloc && state.compute == .all {
                        Label("On this ANE generation MegaLoc's ViT loses accuracy (cos ≈ 0.95 vs PyTorch) and is no faster than the GPU.",
                              systemImage: "exclamationmark.triangle")
                            .font(.caption).foregroundStyle(.orange)
                    }
                } header: {
                    Text("Compute unit")
                } footer: {
                    Text("ANE = Apple Neural Engine (fp16). MixVPR is fastest and accurate on the ANE; MegaLoc should use the GPU.")
                }

                Section {
                    HStack {
                        Slider(value: Binding(get: { Double(state.threshold) }, set: { state.threshold = Float($0) }),
                               in: 0.3...0.95, step: 0.01)
                        Text(String(format: "%.2f", state.threshold))
                            .font(.body.monospacedDigit())
                            .frame(width: 44, alignment: .trailing)
                    }
                } header: {
                    Text("Match threshold · \(state.model.family.rawValue)")
                } footer: {
                    Text("Cosine similarity to the closest stored view. Above the threshold the brackets turn green; within 0.15 below, amber.")
                }

                Section {
                    Button { state.runBenchmark() } label: { Label("Run benchmark", systemImage: "speedometer") }
                } footer: {
                    Text("\(deviceModelIdentifier()) · iOS \(UIDevice.current.systemVersion)")
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }
    }
}

struct BenchSheet: View {
    @ObservedObject var state: AppState
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                ForEach(VPRModel.bundled) { m in
                    let rows = state.benchResults.filter { $0.model == m }
                    if !rows.isEmpty {
                        Section(m.title) {
                            ForEach(rows) { r in
                                HStack {
                                    Text(r.compute.rawValue)
                                        .font(.subheadline.weight(.semibold))
                                        .frame(width: 44, alignment: .leading)
                                    if let e = r.error {
                                        Text(e).font(.caption).foregroundStyle(.red)
                                    } else {
                                        Text(String(format: "median %.1f · p95 %.1f ms · load %.0f ms · n=%d",
                                                    r.medianMs, r.p95Ms, r.loadMs, r.iterations))
                                            .font(.caption.monospacedDigit())
                                            .foregroundStyle(.secondary)
                                    }
                                    Spacer()
                                    Text(String(format: "%.0f", r.fps))
                                        .font(.title3.monospacedDigit().weight(.semibold))
                                    Text("fps").font(.caption).foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }
                Section {
                    if state.benchRunning {
                        HStack {
                            ProgressView()
                            Text("Running…").padding(.leading, 8).foregroundStyle(.secondary)
                            Spacer()
                            Button(role: .destructive) { state.stopBenchmark() } label: {
                                Label("Stop", systemImage: "stop.fill")
                            }
                            .buttonStyle(.bordered)
                        }
                    } else if state.benchStopped {
                        Text("Stopped").foregroundStyle(.secondary)
                    } else if state.benchResults.isEmpty {
                        Text("No results").foregroundStyle(.secondary)
                    } else {
                        Text("Done · \(deviceModelIdentifier()) · iOS \(UIDevice.current.systemVersion)")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Benchmark")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }
    }
}
