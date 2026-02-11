#include <particles.cuh>

int main()
{
    auto initiation = Particles::read_from_file("particles.txt");
    Particles particles(initiation);

    return 0;
}