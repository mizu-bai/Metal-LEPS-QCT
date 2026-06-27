import Metal
import simd

struct LEPSParameters {
    var De: Float  // kJ/mol
    var alpha: Float  // Angstrom^-1
    var r_e: Float  // Angstrom
    var delta: Float  // Sato parameter related value: (1.0f - S) / (1.0f + S)
}

struct LEPSPotentialCalculator {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    private let energyPipeline: MTLComputePipelineState
    private let energyAndForcesPipeline: MTLComputePipelineState

    public let atomCount: Int
    public var parameters: LEPSParameters

    public init(parameters: LEPSParameters, device: MTLDevice? = nil) {
        // constant
        self.atomCount = 3

        // store parameters
        self.parameters = parameters

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
            let energyFunction = library.makeFunction(
                name: "leps_energy_kernel"
            ),
            let energyAndForcesFunction = library.makeFunction(
                name: "leps_energy_and_forces_kernel"
            )
        else {
            fatalError()
        }

        self.energyPipeline = try! _device.makeComputePipelineState(
            function: energyFunction
        )

        self.energyAndForcesPipeline = try! _device.makeComputePipelineState(
            function: energyAndForcesFunction
        )
    }

    public func calculateEnergy(_ positions: [simd_float3]) -> [Float] {
        let configurationCount = UInt32(positions.count / atomCount)

        guard configurationCount > 0 else { return [Float]() }

        var energies = [Float](repeating: 0, count: Int(configurationCount))

        let positionBuffer = device.makeBuffer(
            bytes: positions,
            length: MemoryLayout<simd_float3>.stride * positions.count,
            options: .storageModeShared
        )!

        let energyBuffer = device.makeBuffer(
            bytes: &energies,
            length: MemoryLayout<Float>.stride * energies.count,
            options: .storageModeShared
        )!

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let commandEncoder = commandBuffer.makeComputeCommandEncoder()!

        commandEncoder.setComputePipelineState(energyPipeline)

        commandEncoder.setBuffer(positionBuffer, offset: 0, index: 0)
        commandEncoder.setBuffer(energyBuffer, offset: 0, index: 1)

        var _parameters = self.parameters
        commandEncoder.setBytes(
            &_parameters,
            length: MemoryLayout<LEPSParameters>.stride,
            index: 2
        )

        var _configurationCount = configurationCount
        commandEncoder.setBytes(
            &_configurationCount,
            length: MemoryLayout<UInt32>.stride,
            index: 3
        )

        let gridSize = MTLSize(
            width: Int(configurationCount),
            height: 1,
            depth: 1
        )

        let threadgroupWidth = min(
            energyPipeline.threadExecutionWidth,
            Int(configurationCount)
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

        let energyPointer = energyBuffer.contents().assumingMemoryBound(
            to: Float.self
        )

        energies = Array(
            UnsafeBufferPointer(
                start: energyPointer,
                count: energies.count
            )
        )

        return energies
    }

    public func calculateEnergyAndForces(_ positions: [simd_float3]) -> (
        energies: [Float], forces: [simd_float3]
    ) {
        let configurationCount = UInt32(positions.count / atomCount)

        guard configurationCount > 0 else {
            return ([Float](), [simd_float3]())
        }

        var energies = [Float](repeating: 0, count: Int(configurationCount))
        var forces = [simd_float3](
            repeating: .zero,
            count: Int(configurationCount * 3)
        )

        let positionBuffer = device.makeBuffer(
            bytes: positions,
            length: MemoryLayout<simd_float3>.stride * positions.count,
            options: .storageModeShared
        )!

        let energyBuffer = device.makeBuffer(
            bytes: &energies,
            length: MemoryLayout<Float>.stride * energies.count,
            options: .storageModeShared
        )!

        let forceBuffer = device.makeBuffer(
            bytes: &forces,
            length: MemoryLayout<simd_float3>.stride * forces.count,
            options: .storageModeShared
        )!

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let commandEncoder = commandBuffer.makeComputeCommandEncoder()!

        commandEncoder.setComputePipelineState(energyAndForcesPipeline)

        commandEncoder.setBuffer(positionBuffer, offset: 0, index: 0)
        commandEncoder.setBuffer(energyBuffer, offset: 0, index: 1)
        commandEncoder.setBuffer(forceBuffer, offset: 0, index: 2)

        var params = self.parameters

        commandEncoder.setBytes(
            &params,
            length: MemoryLayout<LEPSParameters>.stride,
            index: 3
        )

        var count = configurationCount

        commandEncoder.setBytes(
            &count,
            length: MemoryLayout<UInt32>.stride,
            index: 4
        )

        let gridSize = MTLSize(
            width: Int(configurationCount),
            height: 1,
            depth: 1
        )

        let threadgroupWidth = min(
            energyAndForcesPipeline.threadExecutionWidth,
            Int(configurationCount)
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

        let energyPointer = energyBuffer.contents().assumingMemoryBound(
            to: Float.self
        )

        energies = Array(
            UnsafeBufferPointer(start: energyPointer, count: energies.count)
        )

        let forcePointer = forceBuffer.contents().assumingMemoryBound(
            to: simd_float3.self
        )

        forces = Array(
            UnsafeBufferPointer(start: forcePointer, count: forces.count)
        )

        return (energies, forces)
    }
}
