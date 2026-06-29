import Foundation
import simd

// LEPS parameters for H + H2 System
let S: Float = 0.106  // Sato parameter

var parameters = LEPSParameters(
    De: 457.7,  // kJ/mol
    alpha: 1.942,  // Angstrom^-1
    r_e: 0.741,  // Angstrom
    delta: (1.0 - S) / (1.0 + S)  // Sato parameter related delta value
)

// LEPS calculator
let calculator = LEPSPotentialCalculator(parameters: parameters)

// H2 potential from LEPS
let H2PotentialHartree: (Double) -> Double = { rBohr in
    let rAngstrom = Float(rBohr * AtomicUnits.bohrToAngstrom)

    let positions: [simd_float3] = [
        simd_float3(1000.0, 0.0, 0.0),
        simd_float3(0.0, 0.0, 0.0),
        simd_float3(rAngstrom, 0.0, 0.0),
    ]

    let energies = calculator.calculateEnergy(positions)

    return Double(energies[0]) * AtomicUnits.kJPerMolToHartree
}

// initial state sampling
let vibrationalLevel = 0  // v
let rotationalLevel = 0.0  // J

let collisionEnergy = 60.0  // kJ/mol
let impactParameter = 0.0  // Angstrom
let initialSeparation = 10.0  // Angstrom

let massesAmu = [1.008, 1.008, 1.008]

let trajectoryCount = 1_000_000

nonisolated(unsafe) let sampler = InitialStateSampler()

sampler.prepare(
    v: vibrationalLevel,
    J: rotationalLevel,
    diatomicPotential: H2PotentialHartree,
    masses: massesAmu
)

nonisolated(unsafe) var positionsBatch = [simd_float3](
    repeating: simd_float3(repeating: 0.0),
    count: trajectoryCount * 3
)
nonisolated(unsafe) var momentaBatch = [simd_float3](
    repeating: simd_float3(repeating: 0.0),
    count: trajectoryCount * 3
)

DispatchQueue.concurrentPerform(
    iterations: trajectoryCount,
    execute: { t in
        autoreleasepool {
            let (initialPositions, initialMomenta) = sampler.sample(
                collisionEnergy: collisionEnergy,
                impactParameter: impactParameter,
                initialSeparation: initialSeparation
            )

            let offset = t * 3

            positionsBatch[offset + 0] = initialPositions[0]
            positionsBatch[offset + 1] = initialPositions[1]
            positionsBatch[offset + 2] = initialPositions[2]

            momentaBatch[offset + 0] = initialMomenta[0]
            momentaBatch[offset + 1] = initialMomenta[1]
            momentaBatch[offset + 2] = initialMomenta[2]
        }
    }
)

// propagation
let integrator = VelocityVerletIntegrator(
    kernelName: "velocity_verlet_leps_kernel"
)

let timeStep: Float = 0.1  // fs
let stepsPerBlock: UInt32 = 100
let blockCount: UInt32 = 100

// prepare GPU buffers once
let massesFloat = massesAmu.map { Float($0) }
integrator.prepare(
    positions: positionsBatch,
    momenta: momentaBatch,
    masses: massesFloat
)

for _ in 0..<blockCount {
    integrator.dispatch(
        parameters: parameters,
        timeStep: timeStep,
        totalSteps: stepsPerBlock
    )
}

(positionsBatch, momentaBatch) = integrator.finalize()

// analyze
var nonReactiveCount = 0  // A + BC -> A + BC
var reactiveABCount = 0  // A + BC -> AB + C
var reactiveACCount = 0  // A + BC -> AC + B
var dissociationCount = 0  // A + BC -> A + B + C

let rBonded: Float = 2.2  // Angstrom

for t in 0..<trajectoryCount {
    let offset = t * 3

    let rA = positionsBatch[offset + 0]
    let rB = positionsBatch[offset + 1]
    let rC = positionsBatch[offset + 2]

    let rAB = simd_distance(rA, rB)
    let rBC = simd_distance(rB, rC)
    let rAC = simd_distance(rA, rC)

    if rAB < rBonded {
        reactiveABCount += 1
    } else if rAC < rBonded {
        reactiveACCount += 1
    } else if rBC < rBonded {
        nonReactiveCount += 1
    } else {
        dissociationCount += 1
    }
}

let totalReactive = reactiveABCount + reactiveACCount
let reactionProbability = Double(totalReactive) / Double(trajectoryCount)

// summary
print("Trajectory Count: \(trajectoryCount)")
print("Collision Energy: \(collisionEnergy) kJ/mol")
print("Impact Parameter: \(impactParameter) Angstrom")
print()
print("A + BC -> A + BC    : \(nonReactiveCount)")
print("A + BC -> AB + C    : \(reactiveABCount)")
print("A + BC -> AC + B    : \(reactiveACCount)")
print("A + BC -> A + B + C : \(dissociationCount)")
print()
print("Reaction Probability: \(reactionProbability * 100.0) %")
