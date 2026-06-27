#pragma once

#include <metal_stdlib>

using namespace metal;

// Unit System
// - distance: Angstrom
// - force: kJ/mol/Angstrom
// - time: fs
// - mass: amu
static constexpr constant float kUnitFactor = 1.0e-04f;

template <unsigned int N> struct PhaseSpace {
    float3 positions[N];
    float3 momenta[N];
    float3 positionCorrections[N]; // Kahan Summation
};

template <typename PotentialType>
inline void
performVelocityVerletIntegration(device float3* globalPositions, device float3* globalMomenta,
                                 constant typename PotentialType::Parameters& parameters,
                                 constant float* masses, float timeStep, uint totalSteps,
                                 uint index) {
    constexpr unsigned int N = PotentialType::atomCount;
    uint offset = index * N;

    PhaseSpace<N> phaseSpace;
    float3 forces[N];
    float energy = 0.0f;

    for (unsigned int i = 0; i < N; ++i) {
        phaseSpace.positions[i] = globalPositions[offset + i];
        phaseSpace.momenta[i] = globalMomenta[offset + i];
        phaseSpace.positionCorrections[i] = float3(0.0f);
    }

    PotentialType potential = {parameters};

    potential.calculateEnergyAndForces(phaseSpace.positions, &energy, forces, 0);

    for (uint step = 0; step < totalSteps; ++step) {
        // 1. positions update
        for (unsigned int i = 0; i < N; ++i) {
            float m = masses[i];

            // v = p / m
            float3 velocity = phaseSpace.momenta[i] / m;

            // a = F / m
            float3 acceleration = forces[i] / m * kUnitFactor;

            // x(t + dt) = x(t) + v * dt + 0.5 * a * dt^2
            float3 dx = velocity * timeStep + 0.5f * acceleration * timeStep * timeStep;
            float3 y = dx - phaseSpace.positionCorrections[i];
            float3 temp = phaseSpace.positions[i] + y;

            phaseSpace.positionCorrections[i] = (temp - phaseSpace.positions[i]) - y;
            phaseSpace.positions[i] = temp;
        }

        // 2. momenta update half
        for (unsigned int i = 0; i < N; ++i) {
            // p(t + dt / 2) = p(t) + 0.5 * F * dt
            phaseSpace.momenta[i] += 0.5f * forces[i] * timeStep * kUnitFactor;
        }

        // 3. new force
        float3 forcesNew[N];
        potential.calculateEnergyAndForces(phaseSpace.positions, &energy, forcesNew, 0);

        // 4. momenta update last
        for (unsigned int i = 0; i < N; ++i) {
            // p(t + dt) = p(t + dt / 2) + 0.5 * F_new * dt
            phaseSpace.momenta[i] += 0.5f * forcesNew[i] * timeStep * kUnitFactor;

            forces[i] = forcesNew[i];
        }
    }

    for (unsigned int i = 0; i < N; ++i) {
        globalPositions[offset + i] = phaseSpace.positions[i];
        globalMomenta[offset + i] = phaseSpace.momenta[i];
    }
}
