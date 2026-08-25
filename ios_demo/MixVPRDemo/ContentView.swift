import SwiftUI

struct ContentView: View {
    @StateObject private var state = AppState()

    var body: some View {
        VStack(spacing: 0) {
            TopBar(state: state)
                .padding(.horizontal, 16)
                .frame(height: 48)
            Viewfinder(state: state)
                .aspectRatio(3 / 4, contentMode: .fit)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
            Shelf(state: state)
                .frame(height: 104)
            Controls(state: state)
                .frame(height: 104)
        }
        .background(Color.black.ignoresSafeArea())
        .sheet(isPresented: $state.showSessions) {
            SessionsSheet(state: state)
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
        }
        .sheet(isPresented: $state.showSettings) {
            SettingsSheet(state: state)
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
        }
        .sheet(isPresented: $state.showBench) { BenchSheet(state: state) }
        .sheet(isPresented: Binding(get: { state.detailPlaceID != nil }, set: { if !$0 { state.detailPlaceID = nil } })) {
            PlaceDetailSheet(state: state)
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
        }
        .alert("Rename", isPresented: Binding(get: { state.renameTarget != nil },
                                              set: { if !$0 { state.renameTarget = nil } })) {
            TextField("Name", text: $state.renameText)
            Button("Save") { state.commitRename() }
            Button("Cancel", role: .cancel) { state.renameTarget = nil }
        }
        .task { await state.start() }
    }
}

// MARK: - Top bar: session on the left, model on the right

struct TopBar: View {
    @ObservedObject var state: AppState

    var body: some View {
        HStack {
            Chip {
                state.showSessions = true
            } content: {
                Image(systemName: "square.stack").font(.caption)
                Text(state.session?.name ?? "…").fontWeight(.semibold).lineLimit(1)
                Image(systemName: "chevron.down").font(.caption2).foregroundStyle(.tertiary)
            }
            Spacer()
            Chip {
                state.showSettings = true
            } content: {
                Text(state.model.family.rawValue).fontWeight(.semibold)
                Text(state.compute.rawValue).foregroundStyle(.secondary)
            }
        }
    }
}

struct Chip<Content: View>: View {
    let action: () -> Void
    @ViewBuilder let content: () -> Content

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6, content: content)
                .font(.subheadline)
                .foregroundStyle(.white)
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .background(.white.opacity(0.08), in: Capsule())
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Controls: one shutter, one caption

struct Controls: View {
    @ObservedObject var state: AppState

    var body: some View {
        VStack(spacing: 10) {
            ShutterButton { state.capture() }
            caption
        }
    }

    @ViewBuilder private var caption: some View {
        if let p = state.selectedPlace {
            HStack(spacing: 6) {
                Text("Add view to **\(p.name)**")
                Button { state.selectedPlaceID = nil } label: {
                    Image(systemName: "xmark.circle.fill").foregroundStyle(.tertiary)
                }
                .buttonStyle(.plain)
            }
            .font(.footnote)
            .foregroundStyle(.secondary)
        } else {
            Text("New place").font(.footnote).foregroundStyle(.tertiary)
        }
    }
}

struct ShutterButton: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle().stroke(.white, lineWidth: 3.5).frame(width: 70, height: 70)
                Circle().fill(.white).frame(width: 57, height: 57)
            }
        }
        .buttonStyle(ShutterStyle())
        .accessibilityLabel("Capture")
    }

    private struct ShutterStyle: ButtonStyle {
        func makeBody(configuration: Configuration) -> some View {
            configuration.label
                .scaleEffect(configuration.isPressed ? 0.88 : 1)
                .animation(.spring(duration: 0.2), value: configuration.isPressed)
        }
    }
}
