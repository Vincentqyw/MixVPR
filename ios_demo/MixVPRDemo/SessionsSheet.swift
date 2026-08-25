import SwiftUI

struct SessionsSheet: View {
    @ObservedObject var state: AppState
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                ForEach(state.sessions) { s in
                    Button {
                        state.worker.openSession(s.id)
                        state.selectedPlaceID = nil
                        dismiss()
                    } label: {
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(s.name).foregroundStyle(.primary)
                                Text("\(s.placeCount) places · \(s.imageCount) views · \(s.createdAt.formatted(date: .abbreviated, time: .shortened))")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                            Spacer()
                            if s.id == state.session?.id { Image(systemName: "checkmark").foregroundStyle(.tint) }
                        }
                    }
                    .contextMenu {
                        Button { state.beginRename(.session(s.id)) } label: { Label("Rename", systemImage: "pencil") }
                        Button(role: .destructive) { state.worker.deleteSession(s.id) } label: { Label("Delete", systemImage: "trash") }
                    }
                }
                .onDelete { idx in idx.map { state.sessions[$0].id }.forEach { state.worker.deleteSession($0) } }
            }
            .navigationTitle("Sessions")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Done") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button { state.worker.createSession(); state.selectedPlaceID = nil; dismiss() } label: { Image(systemName: "plus") }
                }
            }
        }
    }
}

struct PlaceDetailSheet: View {
    @ObservedObject var state: AppState
    @Environment(\.dismiss) private var dismiss
    private let columns = [GridItem(.adaptive(minimum: 96), spacing: 10)]

    var body: some View {
        NavigationStack {
            Group {
                if let p = state.detailPlace {
                    ScrollView {
                        LazyVGrid(columns: columns, spacing: 10) {
                            ForEach(p.images) { img in
                                let s = state.stats.imageScores[img.id]
                                ZStack(alignment: .bottomLeading) {
                                    Image(uiImage: img.thumbnail)
                                        .resizable().scaledToFill()
                                        .aspectRatio(1, contentMode: .fill)
                                        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                                    if let s {
                                        Text(String(format: "%.2f", s))
                                            .font(.caption2.monospacedDigit().weight(.semibold))
                                            .foregroundStyle(state.color(for: state.level(for: s)))
                                            .padding(.horizontal, 6).padding(.vertical, 3)
                                            .background(.black.opacity(0.55), in: Capsule())
                                            .padding(6)
                                    }
                                }
                                .contextMenu {
                                    Button(role: .destructive) { state.worker.deleteImage(img.id, from: p.id) } label: {
                                        Label("Delete view", systemImage: "trash")
                                    }
                                }
                            }
                            Button { state.capture(into: p.id) } label: {
                                VStack(spacing: 6) {
                                    Image(systemName: "camera.fill").font(.title3)
                                    Text("Add view").font(.caption)
                                }
                                .frame(maxWidth: .infinity)
                                .aspectRatio(1, contentMode: .fit)
                                .background(.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                            }
                            .buttonStyle(.plain)
                        }
                        .padding()
                        Text("Long-press a view to delete it. A place's score is the best of its views.")
                            .font(.caption).foregroundStyle(.tertiary).padding(.horizontal)
                    }
                    .navigationTitle(p.name)
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar {
                        ToolbarItem(placement: .cancellationAction) { Button("Done") { dismiss() } }
                        ToolbarItem(placement: .confirmationAction) {
                            Menu {
                                Button { state.beginRename(.place(p.id)) } label: { Label("Rename", systemImage: "pencil") }
                                Button(role: .destructive) { state.worker.deletePlace(p.id); dismiss() } label: { Label("Delete place", systemImage: "trash") }
                            } label: { Image(systemName: "ellipsis.circle") }
                        }
                    }
                } else {
                    ContentUnavailableView("Place removed", systemImage: "mappin.slash")
                }
            }
        }
    }
}
