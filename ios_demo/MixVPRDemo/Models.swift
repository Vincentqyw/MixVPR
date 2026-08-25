import CoreML
import Foundation

enum ModelFamily: String, CaseIterable, Identifiable, Codable {
    case mixvpr = "MixVPR"
    case megaloc = "MegaLoc"
    var id: String { rawValue }
    var defaultThreshold: Float { self == .mixvpr ? 0.70 : 0.80 }
    var defaultCompute: ComputeChoice { self == .mixvpr ? .all : .cpuGPU }
}

enum VPRModel: String, CaseIterable, Identifiable {
    case mixvprFP16, mixvprINT8, megalocFP16, megalocINT8
    var id: String { rawValue }

    var family: ModelFamily {
        switch self {
        case .mixvprFP16, .mixvprINT8: return .mixvpr
        case .megalocFP16, .megalocINT8: return .megaloc
        }
    }
    var precision: String {
        switch self {
        case .mixvprFP16, .megalocFP16: return "FP16"
        case .mixvprINT8, .megalocINT8: return "INT8"
        }
    }
    var title: String { "\(family.rawValue) \(precision)" }
    var subtitle: String {
        switch self {
        case .mixvprFP16: return "ResNet-50 · 320² · 4096-d · 21 MB"
        case .mixvprINT8: return "ResNet-50 · 320² · 4096-d · 11 MB"
        case .megalocFP16: return "DINOv2 ViT-B · 322² · 8448-d · 437 MB"
        case .megalocINT8: return "DINOv2 ViT-B · 322² · 8448-d · 219 MB"
        }
    }
    var resource: String {
        switch self {
        case .mixvprFP16: return "mixvpr_fp16"
        case .mixvprINT8: return "mixvpr_int8"
        case .megalocFP16: return "megaloc_fp16"
        case .megalocINT8: return "megaloc_int8"
        }
    }
    /// MegaLoc's paper resolution is 518²; the bundled mobile export uses 322² (2.7× faster).
    /// Change MEGALOC_SIZE in prepare_models.sh and here together.
    var inputSize: Int { family == .mixvpr ? 320 : 322 }
    /// Heavy models get fewer benchmark iterations.
    var benchIterations: Int { family == .mixvpr ? 50 : 8 }
    var bundleURL: URL? { Bundle.main.url(forResource: resource, withExtension: "mlmodelc", subdirectory: "Models") }
    static var bundled: [VPRModel] { allCases.filter { $0.bundleURL != nil } }
}

enum ComputeChoice: String, CaseIterable, Identifiable, Codable {
    case all = "ANE"
    case cpuGPU = "GPU"
    case cpu = "CPU"
    var id: String { rawValue }
    var units: MLComputeUnits {
        switch self {
        case .all: return .all
        case .cpuGPU: return .cpuAndGPU
        case .cpu: return .cpuOnly
        }
    }
}
