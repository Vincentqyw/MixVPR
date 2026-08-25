import SwiftUI

/// Places of the open session. Tap to select (the shutter then adds views to it), tap the
/// selected place again to open its images, long-press for rename / delete.
struct Shelf: View {
    @ObservedObject var state: AppState

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 16) {
                ForEach(state.places) { p in
                    let score = state.stats.placeScores[p.id]
                    ShelfItem(place: p,
                              score: score,
                              isBest: state.stats.best?.id == p.id,
                              isSelected: state.selectedPlaceID == p.id,
                              color: state.color(for: state.level(for: score)))
                        .onTapGesture { state.togglePlace(p) }
                        .contextMenu {
                            Button { state.detailPlaceID = p.id } label: { Label("Views (\(p.images.count))", systemImage: "photo.on.rectangle") }
                            Button { state.beginRename(.place(p.id)) } label: { Label("Rename", systemImage: "pencil") }
                            Button(role: .destructive) { state.worker.deletePlace(p.id) } label: { Label("Delete", systemImage: "trash") }
                        }
                        .transition(.scale(scale: 0.6).combined(with: .opacity))
                }
                if state.places.isEmpty {
                    Text("No places in this session yet")
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
    let place: Place
    let score: Float?
    let isBest: Bool
    let isSelected: Bool
    let color: Color

    var body: some View {
        VStack(spacing: 6) {
            ZStack(alignment: .bottomTrailing) {
                if let cover = place.cover {
                    Image(uiImage: cover)
                        .resizable().scaledToFill()
                        .frame(width: 64, height: 64)
                        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                }
                if place.images.count > 1 {
                    Text("\(place.images.count)")
                        .font(.caption2.weight(.semibold).monospacedDigit())
                        .padding(.horizontal, 5).padding(.vertical, 1)
                        .background(.black.opacity(0.6), in: Capsule())
                        .padding(4)
                }
            }
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .stroke(isSelected ? .white : (isBest ? color : .white.opacity(0.12)), lineWidth: isSelected || isBest ? 2.5 : 1)
            )
            .animation(.easeInOut(duration: 0.2), value: isSelected || isBest)
            Text(place.name)
                .font(.caption2)
                .foregroundStyle(isBest || isSelected ? .primary : .secondary)
                .lineLimit(1)
                .frame(width: 68)
            ZStack(alignment: .leading) {
                Capsule().fill(.white.opacity(0.12))
                Capsule().fill(color).frame(width: 64 * CGFloat(max(0, min(1, score ?? 0))))
            }
            .frame(width: 64, height: 3)
            .animation(.easeOut(duration: 0.2), value: score)
        }
    }
}
