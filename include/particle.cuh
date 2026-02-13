#pragma once

struct Particle
{
    float x, y, z;
    float vx, vy, vz;
    float m, q;

    Particle() = default;
    Particle(float x_, float y_, float z_, float vx_, float vy_, float vz_, float m_, float q_)
        : x(x_), y(y_), z(z_), vx(vx_), vy(vy_), vz(vz_), m(m_), q(q_)
    {
    }
};