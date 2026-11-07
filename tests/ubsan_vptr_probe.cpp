// Member call through a pointer whose object has an unrelated dynamic type. Compiled with the
// project's options under NPU_SANITIZERS=undefined, UBSan's vptr check stops it before the
// call; reaching the return means project code lost the check.
#include <cstdio>

struct Shape {
    virtual ~Shape() = default;
    virtual int sides() const { return 0; }
};

struct Square : Shape {
    int sides() const override { return 4; }
    int corners() const { return 4; }
};

struct Packet {
    virtual ~Packet() = default;
};

int main() {
    Packet packet;
    Packet* volatile hidden = &packet;  // volatile: the optimiser cannot see the dynamic type
    void* raw = hidden;
    const auto* square = static_cast<const Square*>(raw);
    std::printf("vptr check missing: corners() returned %d\n", square->corners());
    return 0;
}
