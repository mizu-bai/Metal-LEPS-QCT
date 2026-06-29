import Foundation
import simd

class InitialStateSampler {
    private var J: Double = 0.0
    private var masses: [Double] = []
    private var pool: [(r: Double, p: Double)] = []

    func prepare(
        v: Int,
        J: Double,
        diatomicPotential: (Double) -> Double,
        masses: [Double]
    ) {
        self.J = J
        self.masses = masses

        let massB = masses[1]
        let massC = masses[2]

        let BCReducedMassAmu = (massB * massC) / (massB + massC)
        let BCReducedMassAu = BCReducedMassAmu * AtomicUnits.amuToAu

        let rMinBohr = 0.3 * AtomicUnits.angstromToBohr
        let rMaxBohr = 4.0 * AtomicUnits.angstromToBohr

        let pointCount = 500
        let gridCount = pointCount - 1
        let dr = (rMaxBohr - rMinBohr) / Double(pointCount)

        var rGrid = [Double](repeating: 0.0, count: gridCount)
        var vGrid = [Double](repeating: 0.0, count: gridCount)

        for i in 0..<gridCount {
            rGrid[i] = (rMinBohr + dr) + Double(i) * dr
            vGrid[i] = diatomicPotential(rGrid[i])
        }

        let initialDiatomicState = DiatomicState(
            v: v,
            J: J,
            reducedMass: BCReducedMassAu,
            energy: 0.0
        )

        let (finalDiatomicState, _) = dvrSolve(
            diatomicState: initialDiatomicState,
            rGrid: rGrid,
            vGrid: vGrid
        )

        self.pool = generateVibrationalPool(
            state: finalDiatomicState,
            rGrid: rGrid,
            vGrid: vGrid
        )
    }

    func sample(
        collisionEnergy: Double,
        impactParameter: Double,
        initialSeparation: Double
    ) -> (positions: [simd_float3], momenta: [simd_float3]) {
        // assuming A + BC -> ...
        let massA = masses[0]
        let massB = masses[1]
        let massC = masses[2]

        let totalMass = massA + massB + massC

        let sample = pool.randomElement()!

        let r0 = sample.r * AtomicUnits.bohrToAngstrom
        let p0 =
            sample.p * AtomicUnits.auToAmu * AtomicUnits.bohrToAngstrom
            / AtomicUnits.auToFs

        // spherical coordinates
        let cosTheta = Double.random(in: -1.0...1.0)
        let sinTheta = sqrt(1.0 - cosTheta * cosTheta)
        let phi = Double.random(in: 0.0...(2.0 * Double.pi))

        let e = simd_double3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta)

        let relativePositionB = -e * (massC / (massB + massC)) * r0
        let relativePositionC = e * (massB / (massB + massC)) * r0

        let vibrationalMomentaB = -e * p0
        let vibrationalMomentaC = e * p0

        var rotationalMomentaB = simd_double3.zero
        var rotationalMomentaC = simd_double3.zero

        if J > 0.0 {
            var randomVector = simd_double3(
                Double.random(in: -1.0...1.0),
                Double.random(in: -1.0...1.0),
                Double.random(in: -1.0...1.0)
            )

            while simd_length(randomVector) < 1.0e-04 {
                randomVector = simd_double3(
                    Double.random(in: -1.0...1.0),
                    Double.random(in: -1.0...1.0),
                    Double.random(in: -1.0...1.0)
                )
            }

            let normal = simd_normalize(simd_cross(e, randomVector))

            let L =
                sqrt(J * (J + 1.0)) * AtomicUnits.auToAmu
                * AtomicUnits.bohrToAngstrom * AtomicUnits.bohrToAngstrom
                / AtomicUnits.auToFs

            let relativeMomenta = simd_cross(normal, e) * (L / r0)

            rotationalMomentaB = -relativeMomenta
            rotationalMomentaC = relativeMomenta
        }

        let relativeMomentaB = vibrationalMomentaB + rotationalMomentaB
        let relativeMomentaC = vibrationalMomentaC + rotationalMomentaC

        // A
        let impactPhi = Double.random(in: 0...(2.0 * Double.pi))

        let relativePositionA = simd_double3(
            impactParameter * cos(impactPhi),
            impactParameter * sin(impactPhi),
            initialSeparation
        )

        let reducedMassCollision = massA * (massB + massC) / totalMass

        let relativeVelocity = sqrt(
            2.0 * collisionEnergy * 1.0e-04 / reducedMassCollision
        )

        let relativeMomentaA = simd_double3(0.0, 0.0, -massA * relativeVelocity)

        // COM

        let centerOfMassPosition =
            (relativePositionA * massA
                + relativePositionB * massB
                + relativePositionC * massC) / totalMass
        let centerOfMassMomentum =
            relativeMomentaA + relativeMomentaB + relativeMomentaC
        let centerOfMassVelocity = centerOfMassMomentum / totalMass

        let finalPositionA = relativePositionA - centerOfMassPosition
        let finalPositionB = relativePositionB - centerOfMassPosition
        let finalPositionC = relativePositionC - centerOfMassPosition

        let finalMomentaA = relativeMomentaA - centerOfMassVelocity * massA
        let finalMomentaB = relativeMomentaB - centerOfMassVelocity * massB
        let finalMomentaC = relativeMomentaC - centerOfMassVelocity * massC

        let positions = [
            simd_float3(finalPositionA),
            simd_float3(finalPositionB),
            simd_float3(finalPositionC),
        ]

        let momenta = [
            simd_float3(finalMomentaA),
            simd_float3(finalMomentaB),
            simd_float3(finalMomentaC),
        ]

        return (positions: positions, momenta: momenta)
    }

    private func interpolatePotential(
        r: Double,
        rGrid: [Double],
        vGrid: [Double]
    ) -> Double {
        if r <= rGrid.first! {
            return vGrid.first!
        }

        if r >= rGrid.last! {
            return vGrid.last!
        }

        var i = 0

        while i < rGrid.count - 1 && r > rGrid[i + 1] {
            i += 1
        }

        let rLeft = rGrid[i]
        let rRight = rGrid[i + 1]

        let t = (r - rLeft) / (rRight - rLeft)

        return vGrid[i] * (1.0 - t) + vGrid[i + 1] * t
    }

    private func calculateRadialForce(
        r: Double,
        rGrid: [Double],
        vGrid: [Double],
        J: Double,
        reducedMass: Double
    ) -> Double {
        let eps: Double = 1.0e-05

        let vEffective: (Double) -> Double = { radius in
            let vPotential = self.interpolatePotential(
                r: radius,
                rGrid: rGrid,
                vGrid: vGrid
            )
            let vCentrifugal =
                (J * (J + 1.0)) / (2.0 * reducedMass * radius * radius)

            return vPotential + vCentrifugal
        }

        return -(vEffective(r + eps) - vEffective(r - eps)) / (2.0 * eps)
    }

    private func generateVibrationalPool(
        state: DiatomicState,
        rGrid: [Double],
        vGrid: [Double]
    ) -> [(r: Double, p: Double)] {
        // unpack
        let reducedMass = state.reducedMass
        let J = state.J
        let totalEnergy = state.energy

        // find minima on effective potential
        var minIndex = 0
        var minValue = Double.greatestFiniteMagnitude

        let vEffective: (Double) -> Double = { radius in
            let vPotential = self.interpolatePotential(
                r: radius,
                rGrid: rGrid,
                vGrid: vGrid
            )
            let vCentrifugal =
                (J * (J + 1.0)) / (2.0 * reducedMass * radius * radius)

            return vPotential + vCentrifugal
        }

        for i in 0..<rGrid.count {
            let value = vEffective(rGrid[i])

            if value < minValue {
                minValue = value
                minIndex = i
            }
        }

        let rEq = rGrid[minIndex]

        // initialize semiclassical state
        var r = rEq

        let kineticEnergyAtEq = totalEnergy - minValue

        guard kineticEnergyAtEq > 0 else {
            fatalError()
        }

        var momenta = sqrt(2.0 * reducedMass * kineticEnergyAtEq)

        var pool: [(r: Double, p: Double)] = []

        // propagate
        let timeStep = 0.2  // time in atomic unit
        var hasCrossedNegative = false

        for _ in 0..<20000 {
            pool.append((r, momenta))

            let radicalForce = calculateRadialForce(
                r: r,
                rGrid: rGrid,
                vGrid: vGrid,
                J: J,
                reducedMass: reducedMass
            )

            // a(t) = f(t) / m
            let acceleration = radicalForce / reducedMass

            // x(t + dt) = x(t) + v * dt + 1/2 * a * dt * dt
            let rNext =
                r + (momenta / reducedMass) * timeStep + 0.5 * acceleration
                * timeStep * timeStep

            let radicalForceNext = calculateRadialForce(
                r: rNext,
                rGrid: rGrid,
                vGrid: vGrid,
                J: J,
                reducedMass: reducedMass
            )

            // p(t + dt) = p(t) + 0.5 * (f(t) + f(t + dt)) * dt
            let momentaNext =
                momenta + 0.5 * (radicalForce + radicalForceNext) * timeStep

            if r <= rEq && rNext > rEq {
                if hasCrossedNegative {
                    break
                }
            } else if r >= rEq && rNext < rEq {
                hasCrossedNegative = true
            }

            r = rNext
            momenta = momentaNext
        }

        return pool
    }
}
