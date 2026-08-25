import SwiftUI

/// All sessions; live best-image similarity is shown for the sessions being searched. Tap to switch, swipe to delete.
struct SessionsSheet: View {
    @ObservedObject var state: AppState
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                ForEach(state.sessions.reversed()) { s in
                    let score = state.stats.sessionScores[s.id]
                    Button {
                        state.worker.openSession(s.id)
                        dismiss()
                    } label: {
                        HStack(spacing: 12) {
                            Group {
                                if let c = s.cover {
                                    Image(uiImage: c).resizable().scaledToFill()
                                } else {
                                    Color.white.opacity(0.08)
                                }
                            }
                            .frame(width: 44, height: 44)
                            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                            VStack(alignment: .leading, spacing: 2) {
                                Text(s.name).foregroundStyle(.primary)
                                Text("\(s.images.count) images · \(s.createdAt.formatted(date: .abbreviated, time: .shortened))")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                            Spacer()
                            if let score {
                                Text(String(format: "%.2f", score))
                                    .font(.subheadline.monospacedDigit().weight(.semibold))
                                    .foregroundStyle(state.color(for: state.level(for: score)))
                            }
                            if s.id == state.currentID { Image(systemName: "checkmark").foregroundStyle(.tint) }
                        }
                    }
                    .contextMenu {
                        Button { state.beginRename(s.id) } label: { Label("Rename", systemImage: "pencil") }
                        Button(role: .destructive) { state.worker.deleteSession(s.id) } label: { Label("Delete", systemImage: "trash") }
                    }
                }
                .onDelete { idx in
                    let list = Array(state.sessions.reversed())
                    idx.map { list[$0].id }.forEach { state.worker.deleteSession($0) }
                }
            }
            .navigationTitle("Sessions")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Done") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button { state.newSession(); dismiss() } label: { Image(systemName: "plus") }
                }
            }
        }
    }
}

struct ImageDetailSheet: View {
    @ObservedObject var state: AppState
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Group {
                if let d = state.detail {
                    let score = state.stats.imageScores[d.image.id]
                    VStack(spacing: 14) {
                        Image(uiImage: d.image.thumbnail)
                            .resizable().scaledToFit()
                            .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                            .padding(.horizontal)
                        HStack {
                            Text(d.image.createdAt.formatted(date: .abbreviated, time: .shortened))
                                .font(.footnote).foregroundStyle(.secondary)
                            Spacer()
                            if let score {
                                Text(String(format: "live similarity %.2f", score))
                                    .font(.footnote.monospacedDigit())
                                    .foregroundStyle(state.color(for: state.level(for: score)))
                            }
                        }
                        .padding(.horizontal, 24)
                        if d.session.id != state.currentID {
                            Button("Switch to \(d.session.name)") { state.worker.openSession(d.session.id); dismiss() }
                                .font(.footnote)
                        }
                        Spacer()
                    }
                    .padding(.top)
                    .navigationTitle("\(d.session.name) · #\(d.index)")
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar {
                        ToolbarItem(placement: .cancellationAction) { Button("Done") { dismiss() } }
                        ToolbarItem(placement: .destructiveAction) {
                            Button(role: .destructive) { state.worker.deleteImage(d.image.id); dismiss() } label: { Image(systemName: "trash") }
                        }
                    }
                } else {
                    ContentUnavailableView("Image removed", systemImage: "photo")
                }
            }
        }
    }
}
