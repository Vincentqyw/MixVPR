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

    private var text: String {
        state.stats.fps > 0 ? String(format: "%.1f ms · %.0f fps", state.stats.inferMs, state.stats.fps) : state.status
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

struct MatchChip: View {
    @ObservedObject var state: AppState

    var body: some View {
        Group {
            if state.places.isEmpty {
                Text("Point at a place, then tap the shutter")
                    .font(.footnote)
                    .foregroundStyle(.white.opacity(0.9))
            } else {
                HStack(spacing: 10) {
                    if let p = state.bestPlace, let cover = p.cover {
                        Image(uiImage: cover)
                            .resizable().scaledToFill()
                            .frame(width: 30, height: 30)
                            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                        Text(label(for: p)).font(.subheadline.weight(.semibold)).lineLimit(1)
                    } else {
                        Text("Looking…").font(.subheadline).foregroundStyle(.secondary)
                    }
                    if let s = state.stats.best?.score {
                        Text(String(format: "%.2f", s))
                            .font(.subheadline.monospacedDigit().weight(.semibold))
                            .foregroundStyle(state.matchColor)
                            .contentTransition(.numericText())
                    }
                }
                .animation(.easeInOut(duration: 0.2), value: state.stats.best?.score)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 9)
        .background(.ultraThinMaterial, in: Capsule())
    }

    private func label(for p: Place) -> String {
        switch state.matchLevel {
        case .confident: return p.name
        case .weak: return "\(p.name)?"
        case .none: return "No match"
        }
    }
}
