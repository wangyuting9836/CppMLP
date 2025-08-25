#include <iostream>
#include <random>
#include <numeric>

#include "Tensor.h"
#include "Net.h"
#include "utils.h"
#include "include/mnist/mnist_reader.hpp"

struct MNIST_DATA_set
{
	std::vector<std::vector<float>> training_images; ///< The training images
	std::vector<std::vector<float>> test_images;     ///< The test images
	std::vector<std::vector<float>> training_labels; ///< The training labels
	std::vector<std::vector<float>> test_labels;     ///< The test labels
};

MNIST_DATA_set read_MNIST_DATA()
{
	std::string MNIST_DATA_LOCATION = "../MMISTdataset";
	mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> dataset =
		mnist::read_dataset<std::vector, std::vector, float, uint8_t>(MNIST_DATA_LOCATION);

	//std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
	//std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
	//std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
	//std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

	std::for_each(dataset.training_images.begin(), dataset.training_images.end(), [](auto& m)
	{
		std::for_each(m.begin(), m.end(), [](auto& v)
		{
			v = v / 255.0f;
		});
	});

	std::for_each(dataset.test_images.begin(), dataset.test_images.end(), [](auto& m)
	{
		std::for_each(m.begin(), m.end(), [](auto& v)
		{
			v = v / 255.0f;
		});
	});

	int train_num = dataset.training_images.size();
	int test_num = dataset.test_images.size();

	std::vector<std::vector<float>> training_labels(train_num, std::vector<float>(10, 0));
	for (int i = 0; i < train_num; ++i)
	{
		training_labels[i][dataset.training_labels[i]] = 1;
	}
	std::vector<std::vector<float>> test_labels(train_num, std::vector<float>(10, 0));
	for (int i = 0; i < test_num; ++i)
	{
		test_labels[i][dataset.test_labels[i]] = 1;
	}

	return { dataset.training_images, dataset.test_images, training_labels, test_labels };
}

//线性回归
void linear_regression()
{
	std::uniform_real_distribution<float> u_dist(0.0f, 10.0f);
	std::normal_distribution<float> n_dist(0.0f, 2.0f);

	int train_num = 40000;
	int test_num = 10000;

	std::vector<std::vector<float>> training_datas(train_num);
	std::vector<std::vector<float>> training_labels(train_num);
	std::vector<std::vector<float>> test_datas(test_num);
	std::vector<std::vector<float>> test_labels(test_num);

	float true_w1 = 2.0;
	float true_w2 = 3.0;
	float true_b = 7.0;
	for (int i = 0; i < train_num; ++i)
	{
		float x1 = u_dist(rand_generator());
		float x2 = u_dist(rand_generator());
		float y = true_w1 * x1 + true_w2 * x2 + true_b + n_dist(rand_generator());
		training_datas[i].emplace_back(x1);
		training_datas[i].emplace_back(x2);
		training_labels[i].emplace_back(y);
	}

	for (int i = 0; i < test_num; ++i)
	{
		float x1 = u_dist(rand_generator());
		float x2 = u_dist(rand_generator());
		float y = true_w1 * x1 + true_w2 * x2 + true_b + n_dist(rand_generator());
		test_datas[i].emplace_back(x1);
		test_datas[i].emplace_back(x2);
		test_labels[i].emplace_back(y);
	}

	const int batch_size = 10;
	float lr = 0.001;

	Net nn;

	nn.add_layer(2, 1, FCLayer::ActivationFun::NONE);
	nn.set_loss_function(Net::LossFun::MEAN_SQUARED);
	nn.init_parameters(0.0f, 0.02f);

	std::vector<std::vector<float>> batch_training_data;
	std::vector<std::vector<float>> batch_training_label;

	std::vector<int> data_index_vec(train_num, 0);
	std::iota(data_index_vec.begin(), data_index_vec.end(), 0);

	for (int epoch = 0; epoch < 10; ++epoch)
	{
		float train_loss_value = 0.0f;
		std::shuffle(begin(data_index_vec), end(data_index_vec), rand_generator());

		int remain_data_num = train_num;
		int data_index = 0;
		while (remain_data_num > 0)
		{
			batch_training_data.clear();
			batch_training_label.clear();
			for (int k = 0; k < std::min(batch_size, remain_data_num); ++k)
			{
				int i = data_index_vec[data_index];
				batch_training_data.emplace_back(training_datas[i]);
				batch_training_label.emplace_back(training_labels[i]);
				++data_index;
				remain_data_num -= batch_size;
			}
			train_loss_value += nn.train(batch_training_data, batch_training_label);
			nn.SGD(lr);
			nn.zero_grad();
		}
		nn.printParameters();
		std::cout << "loss: " << train_loss_value / static_cast<float >(train_num) << std::endl;
	}

	float train_lose = 0.0f;
	int train_acc_num = 0;
	for (int i = 0; i < train_num; ++i)
	{
		train_lose += nn.forward(training_datas[i], training_labels[i]);
		const std::vector<float>& y_hat = nn.get_y_hat();
		if (fabs(y_hat[0] - training_labels[i][0]) < 2)
		{
			++train_acc_num;
		}
	}
	std::cout << "train_acc: " << static_cast<float>(train_acc_num) / static_cast<float>(train_num) << ", train_loss: "
			  << train_lose / static_cast<float>(train_num) << std::endl;

	float test_lose = 0.0f;
	int test_acc_num = 0;
	for (int i = 0; i < test_num; ++i)
	{
		test_lose += nn.forward(test_datas[i], test_labels[i]);
		const std::vector<float>& y_hat = nn.get_y_hat();
		if (fabs(y_hat[0] - test_labels[i][0]) < 2)
		{
			++test_acc_num;
		}
	}
	std::cout << "test_acc: " << static_cast<float>(test_acc_num) / static_cast<float>(test_num) << ", test_loss: "
			  << test_lose / static_cast<float>(test_num) << std::endl;
}

void MNIST_DATA_softmax()
{
	MNIST_DATA_set dataset = read_MNIST_DATA();

	int train_num = static_cast<int>(dataset.training_images.size());
	int test_num = static_cast<int>(dataset.test_images.size());

	const int batch_size = 10;
	float lr = 0.001;

	Net nn;

	nn.add_layer(784, 10, FCLayer::ActivationFun::NONE);
	nn.set_loss_function(Net::LossFun::CROSS_ENTROPY);
	nn.init_parameters(0.0f, 0.02f);

	std::vector<std::vector<float>> batch_training_data;
	std::vector<std::vector<float>> batch_training_label;

	std::vector<int> data_index_vec(train_num, 0);
	std::iota(data_index_vec.begin(), data_index_vec.end(), 0);

	for (int epoch = 0; epoch < 10; ++epoch)
	{
		float train_loss_value = 0.0f;
		std::shuffle(begin(data_index_vec), end(data_index_vec), rand_generator());

		int remain_data_num = train_num;
		int data_index = 0;
		while (remain_data_num > 0)
		{
			batch_training_data.clear();
			batch_training_label.clear();
			for (int k = 0; k < std::min(batch_size, remain_data_num); ++k)
			{
				int i = data_index_vec[data_index];
				batch_training_data.emplace_back(dataset.training_images[i]);
				batch_training_label.emplace_back(dataset.training_labels[i]);
				++data_index;
				remain_data_num -= batch_size;
			}
			train_loss_value += nn.train(batch_training_data, batch_training_label);
			nn.SGD(lr);
			nn.zero_grad();
		}
		std::cout << "loss: " << train_loss_value / static_cast<float >(train_num) << std::endl;
	}

	float train_lose = 0.0f;
	int train_acc_num = 0;
	for (int i = 0; i < train_num; ++i)
	{
		nn.forward(dataset.training_images[i], dataset.training_labels[i]);
		train_lose += nn.get_loss_value();
		const std::vector<float>& y_hat = nn.get_y_hat_softmax();
		int type1 = static_cast<int>(std::max_element(y_hat.begin(), y_hat.end()) - y_hat.begin());
		int type2 = static_cast<int>(
			std::max_element(dataset.training_labels[i].begin(), dataset.training_labels[i].end())
			- dataset.training_labels[i].begin());

		if (type1 == type2)
		{
			++train_acc_num;
		}
	}
	std::cout << "train_acc: " << static_cast<float>(train_acc_num) / static_cast<float>(train_num) << ", train_loss: "
			  << train_lose / static_cast<float>(train_num) << std::endl;

	float test_lose = 0.0f;
	int test_acc_num = 0;
	for (int i = 0; i < test_num; ++i)
	{
		nn.forward(dataset.test_images[i], dataset.test_labels[i]);
		test_lose += nn.get_loss_value();
		const std::vector<float>& y_hat = nn.get_y_hat_softmax();
		int type1 = static_cast<int>(std::max_element(y_hat.begin(), y_hat.end()) - y_hat.begin());
		int type2 = static_cast<int>(std::max_element(dataset.test_labels[i].begin(), dataset.test_labels[i].end())
									 - dataset.test_labels[i].begin());
		if (type1 == type2)
		{
			++test_acc_num;
		}
	}
	std::cout << "test_acc: " << static_cast<float>(test_acc_num) / static_cast<float>(test_num) << ", test_loss: "
			  << test_lose / static_cast<float>(test_num) << std::endl;
}

//识别手写数字，使用MINST数据集训练
void MNIST_DATA_mlp()
{
	MNIST_DATA_set dataset = read_MNIST_DATA();

	int train_num = static_cast<int>(dataset.training_images.size());
	int test_num = static_cast<int>(dataset.test_images.size());

	const int batch_size = 10;
	float lr = 0.001;

	//构建网络
	Net nn;

	nn.add_layer(784, 256, FCLayer::ActivationFun::RELU);
	nn.add_layer(256, 10, FCLayer::ActivationFun::NONE);
	nn.set_loss_function(Net::LossFun::CROSS_ENTROPY);
	nn.init_parameters(0.0f, 0.02f);

	std::vector<std::vector<float>> batch_training_data;
	std::vector<std::vector<float>> batch_training_label;

	std::vector<int> data_index_vec(train_num, 0);
	std::iota(data_index_vec.begin(), data_index_vec.end(), 0);

	//训练
	for (int epoch = 0; epoch < 10; ++epoch)
	{
		float train_loss_value = 0.0f;
		std::shuffle(begin(data_index_vec), end(data_index_vec), rand_generator());

		int remain_data_num = train_num;
		int data_index = 0;
		while (remain_data_num > 0)
		{
			batch_training_data.clear();
			batch_training_label.clear();
			for (int k = 0; k < std::min(batch_size, remain_data_num); ++k)
			{
				int i = data_index_vec[data_index];
				batch_training_data.emplace_back(dataset.training_images[i]);
				batch_training_label.emplace_back(dataset.training_labels[i]);
				++data_index;
				remain_data_num -= batch_size;
			}
			train_loss_value += nn.train(batch_training_data, batch_training_label);
			nn.SGD(lr);
			nn.zero_grad();
		}
		std::cout << "loss: " << train_loss_value / static_cast<float >(train_num) << std::endl;
	}

	//在训练集上的准确率
	float train_lose = 0.0f;
	int train_acc_num = 0;
	for (int i = 0; i < train_num; ++i)
	{
		nn.forward(dataset.training_images[i], dataset.training_labels[i]);
		train_lose += nn.get_loss_value();
		const std::vector<float>& y_hat = nn.get_y_hat_softmax();
		int type1 = static_cast<int>(std::max_element(y_hat.begin(), y_hat.end()) - y_hat.begin());
		int type2 = static_cast<int>(
			std::max_element(dataset.training_labels[i].begin(), dataset.training_labels[i].end())
			- dataset.training_labels[i].begin());

		if (type1 == type2)
		{
			++train_acc_num;
		}
	}
	std::cout << "train_acc: " << static_cast<float>(train_acc_num) / static_cast<float>(train_num) << ", train_loss: "
			  << train_lose / static_cast<float>(train_num) << std::endl;

	//在测试集上的准确率
	float test_lose = 0.0f;
	int test_acc_num = 0;
	for (int i = 0; i < test_num; ++i)
	{
		nn.forward(dataset.test_images[i], dataset.test_labels[i]);
		test_lose += nn.get_loss_value();
		const std::vector<float>& y_hat = nn.get_y_hat_softmax();
		int type1 = static_cast<int>(std::max_element(y_hat.begin(), y_hat.end()) - y_hat.begin());
		int type2 = static_cast<int>(std::max_element(dataset.test_labels[i].begin(), dataset.test_labels[i].end())
									 - dataset.test_labels[i].begin());
		if (type1 == type2)
		{
			++test_acc_num;
		}
	}
	std::cout << "test_acc: " << static_cast<float>(test_acc_num) / static_cast<float>(test_num) << ", test_loss: "
			  << test_lose / static_cast<float>(test_num) << std::endl;
}

int main()
{
	linear_regression();
	std::cout << "--------------" << std::endl;
	MNIST_DATA_softmax();
	std::cout << "--------------" << std::endl;
	MNIST_DATA_mlp();
	return 0;
}
