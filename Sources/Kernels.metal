#include <metal_stdlib>
using namespace metal;

// integrators
#include "VelocityVerletIntegrator.h"

// potentials
#include "LEPSPotential.h"

// VelocityVerlet + LEPSPotential
kernel void velocity_verlet_leps_kernel(device float3* globalPositions [[buffer(0)]],
                                        device float3* globalMomenta [[buffer(1)]],
                                        constant LEPSParameters& parameters [[buffer(2)]],
                                        constant float* masses [[buffer(3)]],
                                        constant float& timeStep [[buffer(4)]],
                                        constant uint& totalSteps [[buffer(5)]],
                                        uint index [[thread_position_in_grid]]) {
    performVelocityVerletIntegration<LEPSPotential>(globalPositions, globalMomenta, parameters,
                                                    masses, timeStep, totalSteps, index);
}
