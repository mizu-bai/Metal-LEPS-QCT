import Metal
import simd

class VelocityVerletIntegrator {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    private let integrationPipeline: MTLComputePipelineState

    public init(kernelName: String, device: MTLDevice? = nil) {
        // create device
        guard let _device = device ?? MTLCreateSystemDefaultDevice() else {
            fatalError()
        }

        self.device = _device

        // make command queue
        guard let _commandQueue = _device.makeCommandQueue() else {
            fatalError()
        }

        self.commandQueue = _commandQueue

        // load metal library
        guard
            let library = try? _device.makeDefaultLibrary(bundle: Bundle.module)
        else {
            fatalError()
        }

        // find kernel functions
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

    public func integrate<ParameterType>(
        positions: inout [simd_float3],
        momenta: inout [simd_float3],
        parameters: ParameterType,
        masses: [Float],
        timeStep: Float,
        totalSteps: UInt
    ) {
        let atomCount = masses.count
        let trajectoryCount = positions.count / atomCount

        guard trajectoryCount > 0 else { return }

        let positionBuffer = device.makeBuffer(
            bytes: positions,
            length: MemoryLayout<simd_float3>.stride * positions.count,
            options: .storageModeShared
        )!

        let momentaBuffer = device.makeBuffer(
            bytes: momenta,
            length: MemoryLayout<simd_float3>.stride * momenta.count,
            options: .storageModeShared
        )!

        let massBuffer = device.makeBuffer(
            bytes: masses,
            length: MemoryLayout<Float>.stride * masses.count,
            options: .storageModeShared
        )!

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let commandEncoder = commandBuffer.makeComputeCommandEncoder()!

        commandEncoder.setComputePipelineState(integrationPipeline)

        commandEncoder.setBuffer(positionBuffer, offset: 0, index: 0)
        commandEncoder.setBuffer(momentaBuffer, offset: 0, index: 1)

        var _parameters = parameters

        withUnsafeBytes(of: &_parameters) { rawBuffer in
            commandEncoder.setBytes(
                rawBuffer.baseAddress!,
                length: MemoryLayout<ParameterType>.stride,
                index: 2
            )
        }

        commandEncoder.setBuffer(massBuffer, offset: 0, index: 3)

        var _timeStep = timeStep
        commandEncoder.setBytes(
            &_timeStep,
            length: MemoryLayout<Float>.stride,
            index: 4
        )

        var _totalSteps = totalSteps
        commandEncoder.setBytes(
            &_totalSteps,
            length: MemoryLayout<UInt>.stride,
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

        let positionPointer = positionBuffer.contents().assumingMemoryBound(
            to: simd_float3.self
        )

        positions = Array(
            UnsafeBufferPointer(start: positionPointer, count: positions.count)
        )

        let momentaPointer = momentaBuffer.contents().assumingMemoryBound(
            to: simd_float3.self
        )

        momenta = Array(
            UnsafeBufferPointer(start: momentaPointer, count: momenta.count)
        )
    }
}
