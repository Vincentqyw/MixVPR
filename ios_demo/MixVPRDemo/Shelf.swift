import SwiftUI

/// Images of the current session, newest first, each with its live similarity bar; the best match is ringed.
/// Tap to inspect, long-press to delete.
struct Shelf: View {
    @ObservedObject var state: AppState

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 12) {
                ForEach((state.session?.images ?? []).reversed()) { img in
                    let score = state.stats.imageScores[img.id]
                    ShelfItem(image: img, score: score, isBest: state.stats.best?.imageID == img.id,
                              color: state.color(for: state.level(for: score)))
                        .onTapGesture { state.detailImageID = img.id }
                        .contextMenu {
                            Button(role: .destructive) { state.worker.deleteImage(img.id) } label: { Label("Delete", systemImage: "trash") }
                        }
                        .transition(.scale(scale: 0.6).combined(with: .opacity))
                }
                if state.session?.images.isEmpty ?? true {
                    Text("No images in \(state.session?.name ?? "this session") yet")
                        .font(.footnote)
                        .foregroundStyle(.tertiary)
                        .frame(maxWidth: .infinity)
                }
            }
            .padding(.horizontal, 24)
            .frame(minWidth: UIScreen.main.bounds.width)
        }
    }
}

struct ShelfItem: View {
    let image: PlaceImage
    let score: Float?
    let isBest: Bool
    let color: Color

    var body: some View {
        VStack(spacing: 6) {
            Image(uiImage: image.thumbnail)
                .resizable().scaledToFill()
                .frame(width: 64, height: 64)
                .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .stroke(isBest ? color : .white.opacity(0.12), lineWidth: isBest ? 2.5 : 1))
                .animation(.easeInOut(duration: 0.2), value: isBest)
            ZStack(alignment: .leading) {
                Capsule().fill(.white.opacity(0.12))
                Capsule().fill(color).frame(width: 64 * CGFloat(max(0, min(1, score ?? 0))))
            }
            .frame(width: 64, height: 3)
            .animation(.easeOut(duration: 0.2), value: score)
            Text(score.map { String(format: "%.2f", $0) } ?? "–")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(color)
        }
    }
}
