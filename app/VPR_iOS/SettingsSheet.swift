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
                    Toggle("Search across all sessions", isOn: $state.crossSession)
                } header: {
                    Text("Retrieval")
                } footer: {
                    Text(state.crossSession
                         ? "The live view is matched against the images of every session."
                         : "The live view is matched only against the images of the current session.")
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
                    Text("Cosine similarity to the closest stored image. Above the threshold the brackets turn green; within 0.15 below, amber.")
                }

                Section {
                    Button { state.runBenchmark() } label: { Label("Run benchmark", systemImage: "speedometer") }
                    NavigationLink { AboutView() } label: { Label("About", systemImage: "info.circle") }
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

struct AboutView: View {
    private var version: String {
        let v = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0"
        let b = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "1"
        return "\(v) (\(b))"
    }

    var body: some View {
        List {
            Section {
                VStack(alignment: .leading, spacing: 10) {
                    HStack(spacing: 12) {
                        Image(uiImage: UIImage(named: "AppIcon") ?? UIImage())
                            .resizable().frame(width: 56, height: 56)
                            .clipShape(RoundedRectangle(cornerRadius: 13, style: .continuous))
                        VStack(alignment: .leading, spacing: 2) {
                            Text("PlaceLens").font(.title3.weight(.semibold))
                            Text("VPR_iOS · version \(version)").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    Text("On-device visual place recognition. The live camera frame is turned into a global descriptor by a CoreML model and compared, by cosine similarity, with the images you captured in a session — entirely on the phone, nothing leaves the device. Built to measure how VPR models behave on Apple silicon (ANE / GPU / CPU) and as a small field tool for indoor/outdoor relocalisation experiments.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 4)
            }

            Section("Models") {
                aboutLink("MixVPR", "ResNet-50 + feature mixing · 320² · 4096-d (Ali-bey, Chaib-draa, Giguère; WACV 2023)",
                          "https://github.com/amaralibey/MixVPR")
                aboutLink("MegaLoc", "DINOv2 ViT-B + SALAD aggregation · 322² · 8448-d (Berton, Masone; 2025)",
                          "https://github.com/gmberton/MegaLoc")
                aboutLink("CoreML / ONNX checkpoints", "Realcat/image_retrieval_checkpoints on Hugging Face (mixvpr/, megaloc/)",
                          "https://huggingface.co/Realcat/image_retrieval_checkpoints")
            }

            Section("Source") {
                aboutLink("Vincentqyw/MixVPR", "This app (app/), export scripts and mobile benchmarks",
                          "https://github.com/Vincentqyw/MixVPR")
                aboutLink("Vincentqyw/megaloc", "MegaLoc iOS / Neural Engine export recipe",
                          "https://github.com/Vincentqyw/megaloc")
            }

            Section("Author") {
                aboutLink("Vincent Qin", "github.com/Vincentqyw", "https://github.com/Vincentqyw")
            }

            Section {
                Text("MixVPR and MegaLoc are used under their respective open-source licenses; model weights belong to their authors. This app is provided as-is for research and evaluation.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .navigationTitle("About")
        .navigationBarTitleDisplayMode(.inline)
    }

    private func aboutLink(_ title: String, _ subtitle: String, _ url: String) -> some View {
        Link(destination: URL(string: url)!) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).foregroundStyle(.primary)
                    Text(subtitle).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Image(systemName: "arrow.up.right").font(.caption).foregroundStyle(.tertiary)
            }
        }
    }
}
