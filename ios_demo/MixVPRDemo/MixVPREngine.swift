import Accelerate
import CoreML
import CoreVideo
import QuartzCore
import UIKit

enum ModelVariant: String, CaseIterable, Identifiable {
    case fp16 = "FP16"
    case int8 = "INT8"
    var id: String { rawValue }
    var resourceName: String { self == .fp16 ? "mixvpr_fp16" : "mixvpr_int8" }
}

enum ComputeChoice: String, CaseIterable, Identifiable {
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

struct EngineError: Error, CustomStringConvertible {
    let description: String
}

/// One compiled MixVPR CoreML model plus a reusable preprocessing pipeline
/// (center-crop → 320×320 → ImageNet normalisation → fp16 NCHW tensor).
/// Not thread-safe: use from a single serial queue.
final class MixVPREngine {
    static let inputSize = 320
    static let descriptorDim = 4096

    let variant: ModelVariant
    let compute: ComputeChoice
    private(set) var loadMs: Double = 0
    private let model: MLModel
    private let input: MLMultiArray
    private var resized: vImage_Buffer   // 320×320 BGRA copy of the last preprocessed frame

    init(variant: ModelVariant, compute: ComputeChoice) throws {
        self.variant = variant
        self.compute = compute
        guard let url = Bundle.main.url(forResource: variant.resourceName, withExtension: "mlmodelc", subdirectory: "Models") else {
            throw EngineError(description: "Missing \(variant.resourceName).mlmodelc — run ios_demo/prepare_models.sh")
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = compute.units
        let t0 = CACurrentMediaTime()
        model = try MLModel(contentsOf: url, configuration: cfg)
        let n = Self.inputSize
        input = try MLMultiArray(shape: [1, 3, NSNumber(value: n), NSNumber(value: n)], dataType: .float16)
        resized = try vImage_Buffer(width: n, height: n, bitsPerPixel: 32)
        // Warm-up: the first prediction pays for ANE/GPU program compilation.
        fillSynthetic()
        _ = try infer()
        loadMs = (CACurrentMediaTime() - t0) * 1000
    }

    deinit { resized.free() }

    /// Deterministic test pattern, used by the benchmark before the camera delivers a frame.
    func fillSynthetic() {
        let n = Self.inputSize
        let px = resized.data.bindMemory(to: UInt8.self, capacity: resized.rowBytes * n)
        for y in 0..<n {
            let row = px + y * resized.rowBytes
            for x in 0..<n {
                let p = row + x * 4
                p[0] = UInt8((x * 7 + y * 3) & 255)
                p[1] = UInt8((x ^ y) & 255)
                p[2] = UInt8((x * y / 41) & 255)
                p[3] = 255
            }
        }
        fillTensorFromResized()
    }

    /// Center-crops the BGRA pixel buffer to a square, scales it to 320×320 and fills the input tensor.
    func preprocess(_ pb: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        let rowBytes = CVPixelBufferGetBytesPerRow(pb)
        let side = min(w, h)
        let x0 = (w - side) / 2, y0 = (h - side) / 2
        var src = vImage_Buffer(data: base.advanced(by: y0 * rowBytes + x0 * 4),
                                height: vImagePixelCount(side), width: vImagePixelCount(side), rowBytes: rowBytes)
        vImageScale_ARGB8888(&src, &resized, nil, vImage_Flags(kvImageHighQualityResampling))
        fillTensorFromResized()
    }

    private func fillTensorFromResized() {
        let n = Self.inputSize
        let plane = n * n
        // (v/255 - mean) / std  ==  v * scale - offset   (RGB order)
        let scale: [Float] = [1 / (255 * 0.229), 1 / (255 * 0.224), 1 / (255 * 0.225)]
        let offset: [Float] = [0.485 / 0.229, 0.456 / 0.224, 0.406 / 0.225]
        let px = resized.data.bindMemory(to: UInt8.self, capacity: resized.rowBytes * n)
        let rb = resized.rowBytes
        input.withUnsafeMutableBytes { raw, _ in
            let out = raw.bindMemory(to: Float16.self)
            for y in 0..<n {
                let row = px + y * rb
                let o = y * n
                for x in 0..<n {
                    let p = row + x * 4            // B G R A
                    let i = o + x
                    out[i]             = Float16(Float(p[2]) * scale[0] - offset[0])
                    out[plane + i]     = Float16(Float(p[1]) * scale[1] - offset[1])
                    out[2 * plane + i] = Float16(Float(p[0]) * scale[2] - offset[2])
                }
            }
        }
    }

    /// Runs the model on the current tensor and returns an L2-normalised 4096-d descriptor.
    func infer() throws -> [Float] {
        let provider = try MLDictionaryFeatureProvider(dictionary: ["images": MLFeatureValue(multiArray: input)])
        let result = try model.prediction(from: provider)
        guard let arr = result.featureValue(for: "descriptor")?.multiArrayValue else {
            throw EngineError(description: "model returned no 'descriptor' output")
        }
        var v = [Float](repeating: 0, count: arr.count)
        arr.withUnsafeBytes { raw in
            switch arr.dataType {
            case .float16:
                let p = raw.bindMemory(to: Float16.self)
                for i in 0..<v.count { v[i] = Float(p[i]) }
            case .float32:
                let p = raw.bindMemory(to: Float.self)
                for i in 0..<v.count { v[i] = p[i] }
            case .double:
                let p = raw.bindMemory(to: Double.self)
                for i in 0..<v.count { v[i] = Float(p[i]) }
            default:
                break
            }
        }
        var norm: Float = 0
        vDSP_dotpr(v, 1, v, 1, &norm, vDSP_Length(v.count))
        if norm > 0 { v = vDSP.multiply(1 / norm.squareRoot(), v) }
        return v
    }

    /// The last preprocessed 320×320 frame as an image (for place thumbnails).
    func thumbnail() -> UIImage? {
        let n = Self.inputSize
        let info = CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        guard let ctx = CGContext(data: resized.data, width: n, height: n, bitsPerComponent: 8,
                                  bytesPerRow: resized.rowBytes, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: info),
              let cg = ctx.makeImage() else { return nil }
        return UIImage(cgImage: cg)
    }
}

// MARK: - Benchmark

struct BenchResult: Identifiable {
    let id = UUID()
    let variant: ModelVariant
    let compute: ComputeChoice
    let loadMs: Double
    let preMs: Double
    let meanMs: Double
    let medianMs: Double
    let p95Ms: Double
    let error: String?

    var fps: Double { meanMs > 0 ? 1000 / meanMs : 0 }
    var line: String {
        if let error { return "\(variant.rawValue) \(compute.rawValue) ERROR \(error)" }
        return String(format: "%@ %@ load=%.0fms pre=%.2fms mean=%.2fms median=%.2fms p95=%.2fms fps=%.1f",
                      variant.rawValue, compute.rawValue, loadMs, preMs, meanMs, medianMs, p95Ms, fps)
    }
}

extension MixVPREngine {
    /// Times every model × compute-unit combination. `pixelBuffer` (a real camera frame) is optional;
    /// without it a synthetic 480×640 BGRA frame stands in so preprocessing is still measured.
    static func benchmark(iterations: Int, pixelBuffer: CVPixelBuffer?, progress: (BenchResult) -> Void) {
        let pixelBuffer = pixelBuffer ?? makeSyntheticFrame(width: 480, height: 640)
        for variant in ModelVariant.allCases {
            for compute in ComputeChoice.allCases {
                do {
                    let e = try MixVPREngine(variant: variant, compute: compute)
                    var preMs = 0.0
                    if let pb = pixelBuffer {
                        e.preprocess(pb)
                        let t = CACurrentMediaTime()
                        for _ in 0..<20 { e.preprocess(pb) }
                        preMs = (CACurrentMediaTime() - t) * 50
                    } else {
                        e.fillSynthetic()
                    }
                    for _ in 0..<5 { _ = try e.infer() }   // settle clocks
                    var times: [Double] = []
                    times.reserveCapacity(iterations)
                    for _ in 0..<iterations {
                        let t = CACurrentMediaTime()
                        _ = try e.infer()
                        times.append((CACurrentMediaTime() - t) * 1000)
                    }
                    let sorted = times.sorted()
                    progress(BenchResult(variant: variant, compute: compute, loadMs: e.loadMs, preMs: preMs,
                                         meanMs: times.reduce(0, +) / Double(times.count),
                                         medianMs: sorted[sorted.count / 2],
                                         p95Ms: sorted[min(sorted.count - 1, Int(Double(sorted.count) * 0.95))],
                                         error: nil))
                } catch {
                    progress(BenchResult(variant: variant, compute: compute, loadMs: 0, preMs: 0,
                                         meanMs: 0, medianMs: 0, p95Ms: 0, error: "\(error)"))
                }
            }
        }
    }
}

/// A BGRA pixel buffer filled with a gradient, used when the camera has not delivered a frame yet.
func makeSyntheticFrame(width: Int, height: Int) -> CVPixelBuffer? {
    var pb: CVPixelBuffer?
    let attrs = [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary
    guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs, &pb) == kCVReturnSuccess,
          let buf = pb else { return nil }
    CVPixelBufferLockBaseAddress(buf, [])
    defer { CVPixelBufferUnlockBaseAddress(buf, []) }
    guard let base = CVPixelBufferGetBaseAddress(buf) else { return nil }
    let rb = CVPixelBufferGetBytesPerRow(buf)
    let px = base.bindMemory(to: UInt8.self, capacity: rb * height)
    for y in 0..<height {
        for x in 0..<width {
            let p = px + y * rb + x * 4
            p[0] = UInt8(x * 255 / width); p[1] = UInt8(y * 255 / height); p[2] = UInt8((x + y) & 255); p[3] = 255
        }
    }
    return buf
}

func deviceModelIdentifier() -> String {
    var sys = utsname()
    uname(&sys)
    return withUnsafePointer(to: &sys.machine) {
        $0.withMemoryRebound(to: CChar.self, capacity: Int(_SYS_NAMELEN)) { String(cString: $0) }
    }
}
