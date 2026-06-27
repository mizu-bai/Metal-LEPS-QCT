#pragma once

#include <metal_stdlib>

using namespace metal;

// London-Eyring-Polanyi-Sato Potential
struct LEPSParameters {
    float De;    // kJ/mol
    float alpha; // Angstorm^-1
    float r_e;   // Angstorm
    float delta; // Sato parameter related value: (1.0f - S) / (1.0f + S)
};

struct LEPSPotential {
    typedef LEPSParameters Parameters;
    LEPSParameters parameters;

    static constexpr constant uint atomCount = 3;

    // calculate the Coulomb (Q) and Exchange (J) integrals
    inline void calculatePairIntegrals(float r, thread float& Q, thread float& J) const {
        // unpack
        const float De = parameters.De;
        const float alpha = parameters.alpha;
        const float r_e = parameters.r_e;
        const float delta = parameters.delta;

        float x = r - r_e;
        float exp1 = exp(-alpha * x);
        float exp2 = exp1 * exp1;

        // integrals
        float EM = De * (exp2 - 2.0f * exp1);
        float EA = 0.5f * De * (exp2 + 2.0f * exp1);

        Q = 0.5f * (EM + delta * EA);
        J = 0.5f * (EM - delta * EA);
    }

    // calculate the Coulomb (Q) and Exchange (J) integrals and derivatives (dQ, dJ)
    inline void calculatePairIntegralsAndDerivatives(float r, thread float& Q, thread float& J,
                                                     thread float& dQ, thread float& dJ) const {
        // unpack
        const float De = parameters.De;
        const float alpha = parameters.alpha;
        const float r_e = parameters.r_e;
        const float delta = parameters.delta;

        float x = r - r_e;
        float exp1 = exp(-alpha * x);
        float exp2 = exp1 * exp1;

        // integrals
        float EM = De * (exp2 - 2.0f * exp1);
        float EA = 0.5f * De * (exp2 + 2.0f * exp1);

        Q = 0.5f * (EM + delta * EA);
        J = 0.5f * (EM - delta * EA);

        // derivatives
        float dEM = 2.0f * alpha * De * (exp1 - exp2);
        float dEA = -alpha * De * (exp2 + exp1);

        dQ = 0.5f * (dEM + delta * dEA);
        dJ = 0.5f * (dEM - delta * dEA);
    }

    // calculate LEPS potential energy
    template <typename PosPtr, typename EnergyPtr>
    inline void calculateEnergy(PosPtr positions, EnergyPtr energies, uint index) const {
        uint offset = index * atomCount;

        float3 rA = positions[offset + 0];
        float3 rB = positions[offset + 1];
        float3 rC = positions[offset + 2];

        // calculate distances
        float3 drAB = rA - rB;
        float rAB = max(length(drAB), 1.0e-6f);

        float3 drBC = rB - rC;
        float rBC = max(length(drBC), 1.0e-6f);

        float3 drAC = rA - rC;
        float rAC = max(length(drAC), 1.0e-6f);

        // calculate integrals
        float Q_AB, J_AB;
        float Q_BC, J_BC;
        float Q_AC, J_AC;

        calculatePairIntegrals(rAB, Q_AB, J_AB);
        calculatePairIntegrals(rBC, Q_BC, J_BC);
        calculatePairIntegrals(rAC, Q_AC, J_AC);

        float sum_term = 0.0f;

        sum_term += (J_AB - J_BC) * (J_AB - J_BC);
        sum_term += (J_BC - J_AC) * (J_BC - J_AC);
        sum_term += (J_AB - J_AC) * (J_AB - J_AC);

        float W = sqrt(0.5f * sum_term + 1e-6f);

        // energy
        energies[index] = Q_AB + Q_BC + Q_AC - W;
    }

    // calculate LEPS potential energy and forces
    template <typename PosPtr, typename EnergyPtr, typename ForcePtr>
    inline void calculateEnergyAndForces(PosPtr positions, EnergyPtr energies, ForcePtr forces,
                                         uint index) const {
        uint offset = index * atomCount;

        float3 rA = positions[offset + 0];
        float3 rB = positions[offset + 1];
        float3 rC = positions[offset + 2];

        // calculate distances
        float3 drAB = rA - rB;
        float rAB = max(length(drAB), 1.0e-6f);

        float3 drBC = rB - rC;
        float rBC = max(length(drBC), 1.0e-6f);

        float3 drAC = rA - rC;
        float rAC = max(length(drAC), 1.0e-6f);

        // calculate integrals and derivatives
        float Q_AB, J_AB, dQ_AB, dJ_AB;
        float Q_BC, J_BC, dQ_BC, dJ_BC;
        float Q_AC, J_AC, dQ_AC, dJ_AC;

        calculatePairIntegralsAndDerivatives(rAB, Q_AB, J_AB, dQ_AB, dJ_AB);
        calculatePairIntegralsAndDerivatives(rBC, Q_BC, J_BC, dQ_BC, dJ_BC);
        calculatePairIntegralsAndDerivatives(rAC, Q_AC, J_AC, dQ_AC, dJ_AC);

        float sum_term = 0.0f;

        sum_term += (J_AB - J_BC) * (J_AB - J_BC);
        sum_term += (J_BC - J_AC) * (J_BC - J_AC);
        sum_term += (J_AB - J_AC) * (J_AB - J_AC);

        float W = sqrt(0.5f * sum_term + 1e-6f);

        // energy
        energies[index] = Q_AB + Q_BC + Q_AC - W;

        // forces
        float W_inv = 1.0f / (2.0f * W);

        float F_AB = -(dQ_AB - (2.0f * J_AB - J_BC - J_AC) * W_inv * dJ_AB);
        float F_BC = -(dQ_BC - (2.0f * J_BC - J_AB - J_AC) * W_inv * dJ_BC);
        float F_AC = -(dQ_AC - (2.0f * J_AC - J_AB - J_BC) * W_inv * dJ_AC);

        float3 fA = +F_AB / rAB * drAB + F_AC / rAC * drAC;
        float3 fB = -F_AB / rAB * drAB + F_BC / rBC * drBC;
        float3 fC = -F_AC / rAC * drAC - F_BC / rBC * drBC;

        // net force correction
        float3 f_net = fA + fB + fC;
        float3 f_corr = f_net * 0.33333333f;

        forces[offset + 0] = fA - f_corr; // A
        forces[offset + 1] = fB - f_corr; // B
        forces[offset + 2] = fC - f_corr; // C
    }
};
