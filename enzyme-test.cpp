#include <fstream>
#include <Eigen/Dense>
#include <cstdint>
#include <vector>
#include <string>

struct Layer {
    Eigen::MatrixXf W;
    Eigen::MatrixXf x;
    Eigen::MatrixXf b;

    Layer(Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf);
};

Layer::Layer(Eigen::MatrixXf _w, Eigen::MatrixXf _x, Eigen::MatrixXf _b)
    :W{_w}, x{_x}, b{_b} {}

class Network {
    std::ifstream file;
    std::vector<Layer> layers;
public:
    Network(char* path, size_t depth, size_t width, size_t batch_size);
    float forward(float in, float label);
    void step();
};

Network::Network(char* path, size_t depth, size_t width, size_t batch_size)
    :file{path, std::ios::binary}
{
    layers.emplace_back(
	 Eigen::MatrixXf::Zero(1, width),
	 Eigen::MatrixXf::Zero(batch_size, 1),
	 Eigen::MatrixXf::Zero(batch_size, 1)
    );
    for (int i = 0; i < depth; i++) {
	layers.emplace_back(
	    Eigen::MatrixXf::Zero(width, width),
	    Eigen::MatrixXf::Zero(batch_size, width),
	    Eigen::MatrixXf::Zero(batch_size, width)
        );
    }
    layers.emplace_back(
	 Eigen::MatrixXf::Zero(width, 1),
	 Eigen::MatrixXf::Zero(batch_size, 1),
	 Eigen::MatrixXf::Zero(batch_size, 1)
    );    
}

float Network::forward(float in, float label) {
    layers[0].x(0,0) = in;
    for (int i = 1; i < layers.size(); i++) {
	layers[i].x = (layers[i-1].W * layers[i-1].x) + layers[i-1].b;
    }
    return label-in;
}

extern int enzyme_const;
template<typename Return, typename... T>
Return __enzyme_autodiff(T...);

static void deriv(int in, const Network* network, Network* gradient, int label) {
    __enzyme_autodiff<void>(&Network::forward, enzyme_const, in, network, gradient, enzyme_const, label);
}

void Network::step() {
    Network gradient();    
}
