import Foundation
import UIKit

/// One captured view of a place. `descriptors` is filled lazily per model family so a
/// session captured with MixVPR can be re-indexed for MegaLoc (and vice versa).
struct PlaceImage: Identifiable {
    let id: UUID
    var descriptors: [ModelFamily: [Float]]
    let jpeg: Data
    let thumbnail: UIImage
    let createdAt: Date
}

struct Place: Identifiable {
    let id: UUID
    var name: String
    var images: [PlaceImage]
    var cover: UIImage? { images.last?.thumbnail }
}

struct Session: Identifiable {
    let id: UUID
    var name: String
    let createdAt: Date
    var places: [Place]
    var imageCount: Int { places.reduce(0) { $0 + $1.images.count } }
}

struct SessionSummary: Identifiable, Equatable {
    let id: UUID
    let name: String
    let createdAt: Date
    let placeCount: Int
    let imageCount: Int
}

/// Binary plist per session under Documents/sessions/.
enum SessionStore {
    private struct ImageDTO: Codable { let id: UUID; var descriptors: [String: [Float]]; let jpeg: Data; let createdAt: Date }
    private struct PlaceDTO: Codable { let id: UUID; var name: String; var images: [ImageDTO] }
    private struct SessionDTO: Codable { let id: UUID; var name: String; let createdAt: Date; var places: [PlaceDTO] }

    static var directory: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("sessions", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private static func url(_ id: UUID) -> URL { directory.appendingPathComponent("\(id.uuidString).plist") }

    private static func loadDTO(_ url: URL) -> SessionDTO? {
        guard let d = try? Data(contentsOf: url) else { return nil }
        return try? PropertyListDecoder().decode(SessionDTO.self, from: d)
    }

    static func list() -> [SessionSummary] {
        let files = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        return files.compactMap { f -> SessionSummary? in
            guard f.pathExtension == "plist", let s = loadDTO(f) else { return nil }
            return SessionSummary(id: s.id, name: s.name, createdAt: s.createdAt, placeCount: s.places.count,
                                  imageCount: s.places.reduce(0) { $0 + $1.images.count })
        }
        .sorted { $0.createdAt > $1.createdAt }
    }

    static func load(_ id: UUID) -> Session? {
        guard let s = loadDTO(url(id)) else { return nil }
        let places = s.places.map { p in
            Place(id: p.id, name: p.name, images: p.images.compactMap { i in
                guard let img = UIImage(data: i.jpeg) else { return nil }
                var descs: [ModelFamily: [Float]] = [:]
                for (k, v) in i.descriptors { if let f = ModelFamily(rawValue: k) { descs[f] = v } }
                return PlaceImage(id: i.id, descriptors: descs, jpeg: i.jpeg, thumbnail: img, createdAt: i.createdAt)
            })
        }
        return Session(id: s.id, name: s.name, createdAt: s.createdAt, places: places)
    }

    static func save(_ s: Session) {
        let dto = SessionDTO(id: s.id, name: s.name, createdAt: s.createdAt, places: s.places.map { p in
            PlaceDTO(id: p.id, name: p.name, images: p.images.map { i in
                ImageDTO(id: i.id,
                         descriptors: Dictionary(uniqueKeysWithValues: i.descriptors.map { ($0.key.rawValue, $0.value) }),
                         jpeg: i.jpeg, createdAt: i.createdAt)
            })
        })
        let enc = PropertyListEncoder()
        enc.outputFormat = .binary
        if let d = try? enc.encode(dto) { try? d.write(to: url(s.id), options: .atomic) }
    }

    static func delete(_ id: UUID) { try? FileManager.default.removeItem(at: url(id)) }
}
