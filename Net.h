//
// Created by wangy on 2023/11/30.
//

#ifndef _NET_H_
#define _NET_H_
#include <vector>
#include "Tensor.h"
#include "FCLayer.h"

class Net
{
 public:
	enum LossFun{MEAN_SQUARED, CROSS_ENTROPY}; // 损失函数类型
	Net();

	// 添加网络层
	void add_layer(int num_of_inputs, int num_of_outputs, FCLayer::ActivationFun fn);

	// 设置损失函数
	void set_loss_function(LossFun fn);

	// 训练和预测
	float train(const std::vector<std::vector<float>>& input_value, const std::vector<std::vector<float>>& label_value) const;

	// 前向传播
	float forward(const std::vector<float>& input_value, const std::vector<float>& label_value) const;

	// 反向传播
	void backward() const;

	void zero_grad() const;

	// 参数更新
	void SGD(float learn_rate) const;

	void init_parameters(float mean, float variance) const;
	void printParameters() const;
	[[nodiscard]] float get_loss_value() const;
	[[nodiscard]] const std::vector<float>& get_y_hat_softmax() const;
	[[nodiscard]] inline const std::vector<float>& get_y_hat() const;
	~Net();
 private:
	std::vector<FCLayer*> layers; 				// 网络层集合
	Tensor* y_label{};							// 标签张量
	Tensor* loss{};								// 损失张量
	void zero_grad_without_parameters() const;
};

inline float Net::get_loss_value() const
{
	return loss->get_data()[0][0];
}

inline const std::vector<float>& Net::get_y_hat_softmax() const
{
	return (*layers.rbegin())->GetOutputs()->get_softmax_data();
}

inline const std::vector<float>& Net::get_y_hat() const
{
	return (*layers.rbegin())->GetOutputs()->get_data()[0];
}

#endif //_NET_H_
