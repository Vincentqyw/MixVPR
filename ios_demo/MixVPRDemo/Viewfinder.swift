import SwiftUI

/// Camera view. The corner brackets mark the square crop the model sees and double as the
/// match indicator: white → nothing, amber → weak, green → confident.
struct Viewfinder: View {
    @ObservedObject var state: AppState

    var body: some View {
        GeometryReader { geo in
            let side = min(geo.size.width, geo.size.height)
            ZStack {
                if let err = state.cameraError {
                    Color(white: 0.07)
                    Text(err).font(.callout).multilineTextAlignment(.center).foregroundStyle(.secondary).padding(32)
                } else {
                    CameraPreview(session: state.camera.session)
                }
                Brackets(color: state.matchColor)
                    .frame(width: side, height: side)
                    .animation(.easeInOut(duration: 0.25), value: state.matchLevel)
                VStack {
                    HStack {
                        Spacer()
                        StatsPill(state: state)
                    }
                    .padding(12)
                    Spacer()
                    MatchChip(state: state).padding(.bottom, 14)
                }
                Color.white.opacity(state.flash ? 0.8 : 0).allowsHitTesting(false)
            }
            .frame(width: geo.size.width, height: geo.size.height)
            .clipShape(RoundedRectangle(cornerRadius: 26, style: .continuous))
            .frame(maxWidth: .infinity, maxHeight: .infinity)   // GeometryReader aligns top-leading; keep it centred
        }
    }
}

struct StatsPill: View {
    @ObservedObject var state: AppState

    var body: some View {
        Text(text)
            .font(.caption2.monospacedDigit())
            .foregroundStyle(.white.opacity(0.85))
            .padding(.horizontal, 9)
            .padding(.vertical, 5)
            .background(.black.opacity(0.35), in: Capsule())
            .contentTransition(.numericText())
    }

    /// "model 9.1 ms" = inference time per frame; "camera 24 fps" = frames the camera delivers.
    private var text: String {
        state.stats.cameraFps > 0
            ? String(format: "model %.1f ms · camera %.0f fps", state.stats.inferMs, state.stats.cameraFps)
            : state.status
    }
}

struct Brackets: View {
    var color: Color
    var length: CGFloat = 30
    var radius: CGFloat = 16
    var inset: CGFloat = 16

    var body: some View {
        GeometryReader { g in
            let w = g.size.width, h = g.size.height
            Path { p in
                p.move(to: CGPoint(x: inset, y: inset + length))
                p.addLine(to: CGPoint(x: inset, y: inset + radius))
                p.addQuadCurve(to: CGPoint(x: inset + radius, y: inset), control: CGPoint(x: inset, y: inset))
                p.addLine(to: CGPoint(x: inset + length, y: inset))

                p.move(to: CGPoint(x: w - inset - length, y: inset))
                p.addLine(to: CGPoint(x: w - inset - radius, y: inset))
                p.addQuadCurve(to: CGPoint(x: w - inset, y: inset + radius), control: CGPoint(x: w - inset, y: inset))
                p.addLine(to: CGPoint(x: w - inset, y: inset + length))

                p.move(to: CGPoint(x: w - inset, y: h - inset - length))
                p.addLine(to: CGPoint(x: w - inset, y: h - inset - radius))
                p.addQuadCurve(to: CGPoint(x: w - inset - radius, y: h - inset), control: CGPoint(x: w - inset, y: h - inset))
                p.addLine(to: CGPoint(x: w - inset - length, y: h - inset))

                p.move(to: CGPoint(x: inset + length, y: h - inset))
                p.addLine(to: CGPoint(x: inset + radius, y: h - inset))
                p.addQuadCurve(to: CGPoint(x: inset, y: h - inset - radius), control: CGPoint(x: inset, y: h - inset))
                p.addLine(to: CGPoint(x: inset, y: h - inset - length))
            }
            .stroke(color, style: StrokeStyle(lineWidth: 3, lineCap: .round))
            .shadow(color: .black.opacity(0.35), radius: 2)
        }
    }
}

/// Best-matching stored image: "Session · #index" plus score. Tap to preview it.
struct MatchChip: View {
    @ObservedObject var state: AppState

    var body: some View {
        Group {
            if !state.hasAnyImages {
                Text("Tap the shutter to add images to this session")
                    .font(.footnote)
                    .foregroundStyle(.white.opacity(0.9))
                    .padding(.horizontal, 14).padding(.vertical, 9)
                    .background(.ultraThinMaterial, in: Capsule())
            } else {
                Button {
                    if let m = state.bestMatch { state.detailImageID = m.image.id }
                } label: {
                    HStack(spacing: 10) {
                        if let m = state.bestMatch {
                            Image(uiImage: m.image.thumbnail)
                                .resizable().scaledToFill()
                                .frame(width: 34, height: 34)
                                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                            VStack(alignment: .leading, spacing: 1) {
                                Text(title(for: m)).font(.subheadline.weight(.semibold)).lineLimit(1)
                                Text("\(m.session.name) · #\(m.index)").font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                            }
                        } else {
                            Text(state.crossSession ? "Looking…" : "Nothing to match in this session")
                                .font(.subheadline).foregroundStyle(.secondary)
                        }
                        if let s = state.stats.best?.score {
                            Text(String(format: "%.2f", s))
                                .font(.subheadline.monospacedDigit().weight(.semibold))
                                .foregroundStyle(state.matchColor)
                                .contentTransition(.numericText())
                        }
                        if state.bestMatch != nil {
                            Image(systemName: "chevron.right").font(.caption2).foregroundStyle(.tertiary)
                        }
                    }
                    .padding(.horizontal, 12).padding(.vertical, 8)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                    .animation(.easeInOut(duration: 0.3), value: state.stats.best?.score)
                    .animation(.easeInOut(duration: 0.3), value: state.stats.best?.imageID)
                }
                .buttonStyle(.plain)
            }
        }
    }

    private func title(for m: (session: Session, image: PlaceImage, index: Int)) -> String {
        switch state.matchLevel {
        case .confident: return "Match"
        case .weak: return "Possible match"
        case .none: return "No match"
        }
    }
}
