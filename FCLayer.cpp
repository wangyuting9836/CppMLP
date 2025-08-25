//
// Created by wangy on 2023/12/2.
//

//
// Created by wangy on 2023/11/30.
//

#include <stack>
#include <iostream>
#include "utils.h"
#include "Net.h"

FCLayer::FCLayer(int num_of_inputs, int num_of_outputs, ActivationFun fn ) : activation_fn(fn)
{
	// 创建各种Tensor
	inputs = new Tensor(1, num_of_inputs);
	outputs = new Tensor(1, num_of_outputs, true);
	weights = new Tensor(num_of_outputs, num_of_inputs, true);
	bias = new Tensor(1, num_of_outputs, true);

	// 构建计算图：Wx
	auto* w_x = new Tensor(1, num_of_outputs, true);
	w_x->set_left(weights);
	w_x->set_right(inputs);
	w_x->set_calculate_fn(Tensor::CalculateFun::MATRIX_MUL_VECTOR);
	intermediate_results.emplace_back(w_x);

	// 构建计算图：Wx + b
	auto* w_x_b = new Tensor(1, num_of_outputs, true);
	w_x_b->set_left(w_x);
	w_x_b->set_right(bias);
	w_x_b->set_calculate_fn(Tensor::CalculateFun::VECTOR_ADD_VECTOR);
	intermediate_results.emplace_back(w_x_b);

	// 添加激活函数
	switch (activation_fn)
	{
	case RELU:
		outputs->set_left(w_x_b);
		outputs->set_calculate_fn(Tensor::CalculateFun::VECTOR_RELU);
		break;
	case NONE:
		outputs->set_left(w_x_b);
		outputs->set_calculate_fn(Tensor::CalculateFun::VECTOR_ASSIGN);
		break;
	}
}

FCLayer::~FCLayer()
{
	delete inputs;
	delete outputs;
	delete weights;
	delete bias;
	for(auto it : intermediate_results)
	{
		delete it;
	}
}
void FCLayer::zero_grad() const
{
	inputs->zero_grad();
	outputs->zero_grad();
	weights->zero_grad();
	bias->zero_grad();
	for(auto it : intermediate_results)
	{
		it->zero_grad();
	}
}

void FCLayer::zero_grad_without_parameters()
{
	inputs->zero_grad();
	outputs->zero_grad();
	for(auto it : intermediate_results)
	{
		it->zero_grad();
	}
}

void FCLayer::calculate() const
{
	inputs->forward_calculate_data();
	for(auto it : intermediate_results)
	{
		it->forward_calculate_data();
	}
	outputs->forward_calculate_data();
}

void FCLayer::update_parameters(float learn_rate) const
{
	std::vector<std::vector<float>>& weight_data = weights->get_data();
	const std::vector<std::vector<float>>& weight_grad_data = weights->get_grad();

	for(int i = 0; i < weights->get_row_dimension() ;++i)
	{
		for(int j = 0; j < weights->get_col_dimension(); ++j)
		{
			weight_data[i][j] -= learn_rate*weight_grad_data[i][j];
		}
	}

	std::vector<std::vector<float>>& bias_data = bias->get_data();
	const std::vector<std::vector<float>>& bias_grad_data = bias->get_grad();
	for(int i = 0; i < bias->get_row_dimension() ;++i)
	{
		for(int j = 0; j < bias->get_col_dimension(); ++j)
		{
			bias_data[i][j] -= learn_rate*bias_grad_data[i][j];
		}
	}
}

void FCLayer::init_parameters(float mean, float variance) const
{
	std::normal_distribution<float> n_dist(mean, variance);

	std::vector<std::vector<float>>& weight_data = weights->get_data();

	for(int i = 0; i < weights->get_row_dimension() ;++i)
	{
		for(int j = 0; j < weights->get_col_dimension(); ++j)
		{
			weight_data[i][j] = n_dist(rand_generator());
		}
	}

	std::vector<std::vector<float>>& bias_data = bias->get_data();
	for(int i = 0; i < bias->get_row_dimension() ;++i)
	{
		for(int j = 0; j < bias->get_col_dimension(); ++j)
		{
			bias_data[i][j] = n_dist(rand_generator());;
		}
	}
}

void FCLayer::printParameters() const
{
	const std::vector<std::vector<float>>& weight_data = weights->get_data();
	std::cout << "wights: ";
	for(int i = 0; i < weights->get_row_dimension() ;++i)
	{
		for(int j = 0; j < weights->get_col_dimension(); ++j)
		{
			std::cout << weight_data[i][j] << ", ";
		}
	}

	std::cout << "bias: ";
	const std::vector<std::vector<float>>& bias_data = bias->get_data();
	for(int i = 0; i < bias->get_row_dimension() ;++i)
	{
		for(int j = 0; j < bias->get_col_dimension(); ++j)
		{
			std::cout << bias_data[i][j] << ",";
		}
	}
}



