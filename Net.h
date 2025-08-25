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
	enum LossFun{MEAN_SQUARED, CROSS_ENTROPY}; // ��ʧ��������
	Net();

	// ��������
	void add_layer(int num_of_inputs, int num_of_outputs, FCLayer::ActivationFun fn);

	// ������ʧ����
	void set_loss_function(LossFun fn);

	// ѵ����Ԥ��
	float train(const std::vector<std::vector<float>>& input_value, const std::vector<std::vector<float>>& label_value) const;

	// ǰ�򴫲�
	float forward(const std::vector<float>& input_value, const std::vector<float>& label_value) const;

	// ���򴫲�
	void backward() const;

	void zero_grad() const;

	// ��������
	void SGD(float learn_rate) const;

	void init_parameters(float mean, float variance) const;
	void printParameters() const;
	[[nodiscard]] float get_loss_value() const;
	[[nodiscard]] const std::vector<float>& get_y_hat_softmax() const;
	[[nodiscard]] inline const std::vector<float>& get_y_hat() const;
	~Net();
 private:
	std::vector<FCLayer*> layers; 				// ����㼯��
	Tensor* y_label{};							// ��ǩ����
	Tensor* loss{};								// ��ʧ����
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
