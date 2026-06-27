#include <metal_stdlib>

using namespace metal;

#include "LEPSPotential.h"

// kernel to calculate energy
kernel void leps_energy_kernel(constant float3* positions [[buffer(0)]],
                               device float* energies [[buffer(1)]],
                               constant LEPSParameters& params [[buffer(2)]],
                               constant uint& configurationCount [[buffer(3)]],
                               uint index [[thread_position_in_grid]]) {
    if (index >= configurationCount) {
        return;
    }

    LEPSPotential potential = {params};

    potential.calculateEnergy(positions, energies, index);
}

// kernel to calculate energy and forces
kernel void leps_energy_and_forces_kernel(constant float3* positions [[buffer(0)]],
                                          device float* energies [[buffer(1)]],
                                          device float3* forces [[buffer(2)]],
                                          constant LEPSParameters& params [[buffer(3)]],
                                          constant uint& configurationCount [[buffer(4)]],
                                          uint index [[thread_position_in_grid]]) {
    if (index >= configurationCount) {
        return;
    }

    LEPSPotential potential = {params};

    potential.calculateEnergyAndForces(positions, energies, forces, index);
}
