import Metal
import simd

class VelocityVerletIntegrator {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let integrationPipeline: MTLComputePipelineState

    // prepared state
    private var positionBuffer: MTLBuffer?
    private var momentaBuffer: MTLBuffer?
    private var massBuffer: MTLBuffer?
    private var trajectoryCount: Int = 0
    private var atomCount: Int = 0

    public init(kernelName: String, device: MTLDevice? = nil) {
        guard let _device = device ?? MTLCreateSystemDefaultDevice() else {
            fatalError()
        }

        self.device = _device

        guard let _commandQueue = _device.makeCommandQueue() else {
            fatalError()
        }

        self.commandQueue = _commandQueue

        guard
            let library = try? _device.makeDefaultLibrary(bundle: Bundle.module)
        else {
            fatalError()
        }

        guard
            let integrationFunction = library.makeFunction(
                name: kernelName
            )
        else {
            fatalError()
        }

        self.integrationPipeline = try! _device.makeComputePipelineState(
            function: integrationFunction
        )
    }

    public func prepare(
        positions: [simd_float3],
        momenta: [simd_float3],
        masses: [Float]
    ) {
        atomCount = masses.count
        trajectoryCount = positions.count / atomCount

        guard trajectoryCount > 0 else { return }

        positionBuffer = device.makeBuffer(
            bytes: positions,
            length: MemoryLayout<simd_float3>.stride * positions.count,
            options: .storageModeShared
        )

        momentaBuffer = device.makeBuffer(
            bytes: momenta,
            length: MemoryLayout<simd_float3>.stride * momenta.count,
            options: .storageModeShared
        )

        massBuffer = device.makeBuffer(
            bytes: masses,
            length: MemoryLayout<Float>.stride * masses.count,
            options: .storageModeShared
        )
    }

    public func dispatch(
        parameters: LEPSParameters,
        timeStep: Float,
        totalSteps: UInt32
    ) {
        guard let positionBuffer, let momentaBuffer, let massBuffer,
            trajectoryCount > 0
        else { return }

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let commandEncoder = commandBuffer.makeComputeCommandEncoder()!

        commandEncoder.setComputePipelineState(integrationPipeline)

        commandEncoder.setBuffer(positionBuffer, offset: 0, index: 0)
        commandEncoder.setBuffer(momentaBuffer, offset: 0, index: 1)

        var params = parameters
        commandEncoder.setBytes(
            &params,
            length: MemoryLayout<LEPSParameters>.stride,
            index: 2
        )

        commandEncoder.setBuffer(massBuffer, offset: 0, index: 3)

        var ts = timeStep
        commandEncoder.setBytes(
            &ts,
            length: MemoryLayout<Float>.stride,
            index: 4
        )

        var steps = totalSteps
        commandEncoder.setBytes(
            &steps,
            length: MemoryLayout<UInt32>.stride,
            index: 5
        )

        let gridSize = MTLSize(width: trajectoryCount, height: 1, depth: 1)

        let threadgroupWidth = min(
            integrationPipeline.threadExecutionWidth,
            trajectoryCount
        )

        let threadgroupSize = MTLSize(
            width: threadgroupWidth,
            height: 1,
            depth: 1
        )

        commandEncoder.dispatchThreads(
            gridSize,
            threadsPerThreadgroup: threadgroupSize
        )
        commandEncoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    public func finalize() -> (
        positions: [simd_float3], momenta: [simd_float3]
    ) {
        guard let positionBuffer, let momentaBuffer else {
            return ([], [])
        }

        let count = trajectoryCount * atomCount

        let positionPointer = positionBuffer.contents().assumingMemoryBound(
            to: simd_float3.self
        )

        let positions = Array(
            UnsafeBufferPointer(start: positionPointer, count: count)
        )

        let momentaPointer = momentaBuffer.contents().assumingMemoryBound(
            to: simd_float3.self
        )

        let momenta = Array(
            UnsafeBufferPointer(start: momentaPointer, count: count)
        )

        return (positions, momenta)
    }
}
