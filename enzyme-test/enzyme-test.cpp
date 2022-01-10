// #include <iostream>
// #include <fstream>
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

struct Network {
    //std::ifstream file;
    std::vector<Layer> layers;
  
    Network(char* path, size_t depth, size_t width, size_t batch_size);
    Network(size_t depth, size_t width, size_t batch_size);
    float forward(float in, float label);
};

Network::Network(char* path, size_t depth, size_t width, size_t batch_size)
//    :file{path, std::ios::binary}
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

Network::Network(size_t depth, size_t width, size_t batch_size) {
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

float forward(Network* net, float in, float label) {
    net->layers[0].x(0,0) = in;
    for (int i = 1; i < net->layers.size(); i++) {
	net->layers[i].x = (net->layers[i-1].W * net->layers[i-1].x) + net->layers[i-1].b;
    }
    return label-in;
}

extern int enzyme_const;
template<typename Return, typename... T>
Return __enzyme_autodiff(T...);

static void deriv(int in, const Network* network, Network* gradient, int label) {
    __enzyme_autodiff<void>(&Network::forward, enzyme_const, in, network, gradient, enzyme_const, label);
}

void step(Network* net) {
    while (true) {
	Network grad(net->layers.size()-2, net->layers[1].x.cols(), net->layers[1].x.rows()); 
	float loss = forward(net, 3, 6);
	deriv(3, net, &grad, 6);
	for (int i = 0; i < net->layers.size(); i++) {
	    net->layers[i].b -= 0.01 * grad.layers[i].b;
	    net->layers[i].W -= 0.01 * grad.layers[i].W;
	}
	// std::cout << loss << "\n";
    }
}

int main() {
    Network n(const_cast<char*>("./ref"), 1, 4, 1);
    step(&n);
}
