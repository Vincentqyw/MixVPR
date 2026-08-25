import Foundation
import UIKit

/// One captured image. `descriptors` is filled lazily per model family so a session captured
/// with MixVPR can be re-indexed for MegaLoc (and vice versa).
struct PlaceImage: Identifiable {
    let id: UUID
    var descriptors: [ModelFamily: [Float]]
    let jpeg: Data
    let thumbnail: UIImage
    let createdAt: Date
}

/// A session is one place: a named set of images taken around it.
struct Session: Identifiable {
    let id: UUID
    var name: String
    let createdAt: Date
    var images: [PlaceImage]
    var cover: UIImage? { images.last?.thumbnail }
}

/// Binary plist per session under Documents/sessions/.
enum SessionStore {
    private struct ImageDTO: Codable { let id: UUID; var descriptors: [String: [Float]]; let jpeg: Data; let createdAt: Date }
    private struct SessionDTO: Codable { let id: UUID; var name: String; let createdAt: Date; var images: [ImageDTO] }

    static var directory: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("sessions", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private static func url(_ id: UUID) -> URL { directory.appendingPathComponent("\(id.uuidString).plist") }

    static func loadAll() -> [Session] {
        let files = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        return files.compactMap { f -> Session? in
            guard f.pathExtension == "plist", let d = try? Data(contentsOf: f),
                  let s = try? PropertyListDecoder().decode(SessionDTO.self, from: d) else { return nil }
            let images = s.images.compactMap { i -> PlaceImage? in
                guard let img = UIImage(data: i.jpeg) else { return nil }
                var descs: [ModelFamily: [Float]] = [:]
                for (k, v) in i.descriptors { if let f = ModelFamily(rawValue: k) { descs[f] = v } }
                return PlaceImage(id: i.id, descriptors: descs, jpeg: i.jpeg, thumbnail: img, createdAt: i.createdAt)
            }
            // Sessions created by an earlier build were named "Place N"; a session is a capture session.
            var name = s.name
            if let r = name.range(of: #"^Place (\d+)$"#, options: .regularExpression) {
                name = "Session " + name[r].dropFirst(6)
            }
            let session = Session(id: s.id, name: name, createdAt: s.createdAt, images: images)
            if name != s.name { save(session) }
            return session
        }
        .sorted { $0.createdAt < $1.createdAt }
    }

    static func save(_ s: Session) {
        let dto = SessionDTO(id: s.id, name: s.name, createdAt: s.createdAt, images: s.images.map { i in
            ImageDTO(id: i.id,
                     descriptors: Dictionary(uniqueKeysWithValues: i.descriptors.map { ($0.key.rawValue, $0.value) }),
                     jpeg: i.jpeg, createdAt: i.createdAt)
        })
        let enc = PropertyListEncoder()
        enc.outputFormat = .binary
        if let d = try? enc.encode(dto) { try? d.write(to: url(s.id), options: .atomic) }
    }

    static func delete(_ id: UUID) { try? FileManager.default.removeItem(at: url(id)) }
}
