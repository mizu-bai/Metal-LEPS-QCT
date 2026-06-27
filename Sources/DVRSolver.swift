import Accelerate
import Foundation

struct DiatomicState {
    let v: Int
    let J: Double
    let reducedMass: Double
    let energy: Double
}

struct DVRState {
    let rGrid: [Double]
    let energies: [Double]
    let wavefunctions: [Double]
}

func dvrSolve(diatomicState: DiatomicState, rGrid: [Double], vGrid: [Double])
    -> (DiatomicState, DVRState)
{
    // 1. grid
    let gridCount = rGrid.count
    let dr: Double = rGrid[1] - rGrid[0]

    // 2. Hamiltonian matrix
    var hMatrix = [Double](repeating: 0.0, count: gridCount * gridCount)

    let coefficient = 1.0 / (2.0 * diatomicState.reducedMass * dr * dr)

    for i in 0..<gridCount {
        let r = rGrid[i]

        // effective potential
        let vEfficient =
            vGrid[i] + (diatomicState.J * (diatomicState.J + 1.0))
            / (2.0 * diatomicState.reducedMass * r * r)

        for j in 0..<gridCount {
            let index = i * gridCount + j

            if i == j {
                // diag terms
                let tDiag = coefficient * (Double.pi * Double.pi / 3.0)

                hMatrix[index] = tDiag + vEfficient
            } else {
                // off-diag terms
                let diff = Double(i - j)
                let sign = ((i - j) % 2 == 0) ? 1.0 : -1.0
                let tOff = coefficient * sign * (2.0 / (diff * diff))

                hMatrix[index] = tOff
            }
        }
    }

    var jobz = Int8(Character("V").asciiValue!)
    var uplo = Int8(Character("U").asciiValue!)
    var nInt = __CLPK_integer(gridCount)
    var a = hMatrix
    var lda = nInt
    var eigenValues = [Double](repeating: 0.0, count: gridCount)

    var lworkQuery: Double = 0.0
    var lworkQueryInt = __CLPK_integer(-1)
    var info = __CLPK_integer(0)

    dsyev_(
        &jobz,
        &uplo,
        &nInt,
        &a,
        &lda,
        &eigenValues,
        &lworkQuery,
        &lworkQueryInt,
        &info
    )

    let lwork = Int(lworkQuery)
    var work = [Double](repeating: 0.0, count: lwork)
    var lworkInt = __CLPK_integer(lwork)

    dsyev_(
        &jobz,
        &uplo,
        &nInt,
        &a,
        &lda,
        &eigenValues,
        &work,
        &lworkInt,
        &info
    )

    if info != 0 {
        fatalError()
    }

    var wavefunctions = [Double](repeating: 0.0, count: gridCount * gridCount)

    for i in 0..<(gridCount * gridCount) {
        wavefunctions[i] = a[i] * 1.0 / sqrt(dr)
    }

    let updatedDiatomic = DiatomicState(
        v: diatomicState.v,
        J: diatomicState.J,
        reducedMass: diatomicState.reducedMass,
        energy: eigenValues[diatomicState.v]
    )

    let dvrState = DVRState(
        rGrid: rGrid,
        energies: eigenValues,
        wavefunctions: wavefunctions
    )

    return (updatedDiatomic, dvrState)
}
