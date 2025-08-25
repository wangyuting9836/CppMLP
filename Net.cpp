//
// Created by wangy on 2023/11/30.
//

#include <stack>
#include <iostream>
#include "Net.h"

Net::Net() = default;

Net::~Net()
{
	delete loss;
	delete y_label;
	for (const auto& it : layers)
	{
		delete it;
	}
}

void Net::SGD(float learn_rate) const
{
	for (const auto& it : layers)
	{
		it->update_parameters(learn_rate);
	}
}

void Net::init_parameters(float mean, float variance) const
{
	for (const auto& it : layers)
	{
		it->init_parameters(mean, variance);
	}
}

void Net::printParameters() const
{
	int l_no = 1;
	for (const auto& it : layers)
	{
		std::cout << "layer:" << l_no << ", ";
		it->printParameters();
		std::cout << std::endl;
		++l_no;
	}
}

float Net::train(const std::vector<std::vector<float>>& input_value, const std::vector<std::vector<float>>& label_value) const
{
	float loss_value = 0.0f;
	for (int i = 0; i < input_value.size(); ++i)
	{
		forward(input_value[i], label_value[i]);
		loss_value += get_loss_value();
		zero_grad_without_parameters();
		backward();
	}
	return loss_value;
}

float Net::forward(const std::vector<float>& input_value, const std::vector<float>& label_value) const
{
	layers[0]->GetInputs()->set_data(input_value);
	y_label->set_data(label_value);
	for (const auto& it : layers)
	{
		it->calculate();
	}
	loss->forward_calculate_data();
	return get_loss_value();
}

void Net::backward() const
{
	if (!loss->is_require_grad() || loss->get_calculate_fn() == Tensor::LEAF)
	{
		return;
	}

	std::stack<Tensor*> tensor_stack;
	loss->one_grad(); // 设置损失函数的梯度为1
	tensor_stack.push(loss);
	while (true)
	{
		if (tensor_stack.empty())
		{
			break;
		}
		Tensor* cur_tensor = tensor_stack.top();
		tensor_stack.pop();
		Tensor* left = cur_tensor->get_left();
		Tensor* right = cur_tensor->get_right();
		// 向左子节点传播梯度
		if (left != nullptr && left->is_require_grad())
		{
			cur_tensor->backward_calculate_left_grad();
			tensor_stack.push(left);
		}
		// 向右子节点传播梯度
		if (right != nullptr && right->is_require_grad())
		{
			cur_tensor->backward_calculate_right_grad();
			tensor_stack.push(right);
		}
	}
}

void Net::zero_grad() const
{
	for (const auto& it : layers)
	{
		it->zero_grad();
	}
}

void Net::zero_grad_without_parameters() const
{
	for (const auto& it : layers)
	{
		it->zero_grad_without_parameters();
	}
}

void Net::add_layer(int num_of_inputs, int num_of_outputs, FCLayer::ActivationFun activation_fn)
{
	if (layers.empty())
	{
		// 第一层
		layers.emplace_back(new FCLayer(num_of_inputs, num_of_outputs, activation_fn));
	}
	else
	{
		// 后续层：将前一层的输出作为当前层的输入
		layers.emplace_back(new FCLayer(num_of_inputs, num_of_outputs, activation_fn));

		Tensor* cur_layer_inputs = (*layers.rbegin())->GetInputs();
		Tensor* pre_layer_outputs = (*(layers.rbegin() + 1))->GetOutputs();
		cur_layer_inputs->set_left(pre_layer_outputs);
		cur_layer_inputs->set_calculate_fn(Tensor::CalculateFun::VECTOR_ASSIGN);
	}
}

void Net::set_loss_function(LossFun fn)
{
	loss = new Tensor(1, 1, true);
	Tensor* last_layer_outputs = (*layers.rbegin())->GetOutputs();
	y_label = new Tensor(1, last_layer_outputs->get_col_dimension());
	loss->set_left(last_layer_outputs);
	loss->set_right(y_label);
	switch (fn)
	{
	case CROSS_ENTROPY:
		loss->set_calculate_fn(Tensor::CalculateFun::VECTOR_CROSSENTROPY);
		break;
	case MEAN_SQUARED:
		loss->set_calculate_fn(Tensor::CalculateFun::VECTOR_SQUARED);
		break;
	}
}




