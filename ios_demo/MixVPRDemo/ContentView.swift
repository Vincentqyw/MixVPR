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
                .frame(height: 96)
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
        .sheet(isPresented: Binding(get: { state.detailImageID != nil }, set: { if !$0 { state.detailImageID = nil } })) {
            ImageDetailSheet(state: state)
                .presentationDetents([.medium])
                .presentationDragIndicator(.visible)
        }
        .alert("Rename place", isPresented: Binding(get: { state.renamingID != nil },
                                                    set: { if !$0 { state.renamingID = nil } })) {
            TextField("Name", text: $state.renameText)
            Button("Save") { state.commitRename() }
            Button("Cancel", role: .cancel) { state.renamingID = nil }
        }
        .task { await state.start() }
    }
}

// MARK: - Top bar: current place (session) on the left, model on the right

struct TopBar: View {
    @ObservedObject var state: AppState

    var body: some View {
        HStack {
            Chip {
                if let id = state.currentID { state.beginRename(id) }
            } content: {
                Image(systemName: "mappin.and.ellipse").font(.caption)
                Text(state.session?.name ?? "…").fontWeight(.semibold).lineLimit(1)
                if let n = state.session?.images.count, n > 0 {
                    Text("\(n)").foregroundStyle(.secondary).monospacedDigit()
                }
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

// MARK: - Controls: new place · shutter · places list

struct Controls: View {
    @ObservedObject var state: AppState

    var body: some View {
        ZStack {
            HStack {
                RoundIconButton(systemName: "plus", label: "New") { state.newSession() }
                Spacer()
                RoundIconButton(systemName: "square.stack", label: "Places") { state.showSessions = true }
            }
            .padding(.horizontal, 44)
            VStack(spacing: 8) {
                ShutterButton { state.capture() }
                Text("Add to **\(state.session?.name ?? "…")**")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
        }
    }
}

struct RoundIconButton: View {
    let systemName: String
    let label: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 5) {
                Image(systemName: systemName)
                    .font(.system(size: 17, weight: .medium))
                    .foregroundStyle(.white)
                    .frame(width: 46, height: 46)
                    .background(.white.opacity(0.10), in: Circle())
                Text(label).font(.caption2).foregroundStyle(.tertiary)
            }
        }
        .buttonStyle(.plain)
    }
}

struct ShutterButton: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle().stroke(.white, lineWidth: 3.5).frame(width: 66, height: 66)
                Circle().fill(.white).frame(width: 54, height: 54)
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
