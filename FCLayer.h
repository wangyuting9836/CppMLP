//
// Created by wangy on 2023/12/2.
//

#ifndef _FCLAYER_H_
#define _FCLAYER_H_

#include <vector>
#include "Tensor.h"

class FCLayer
{
 public:
	enum ActivationFun{NONE, RELU}; // �����

	// ���캯���������㲢��������ͼ
	FCLayer(int num_of_inputs, int num_of_outputs, ActivationFun fn);

	~FCLayer();

	void zero_grad() const;
	void zero_grad_without_parameters();
	void calculate() const;
	void printParameters() const;
	[[nodiscard]] Tensor* GetInputs() const;
	[[nodiscard]] Tensor* GetOutputs() const;
	[[nodiscard]] Tensor* GetWeights() const;
	[[nodiscard]] Tensor* GetBias() const;
	[[nodiscard]] ActivationFun GetActivationFn() const;

	// ��������
	void update_parameters(float learn_rate) const;

	// ��ʼ������
	void init_parameters(float mean, float variance) const;
 private:
	Tensor* inputs;  //n*1 ��������
	Tensor* outputs; //q*1 �������
	Tensor* weights; //n*q Ȩ�ؾ���
	Tensor* bias;	 //q*1 ƫ������
	std::vector<Tensor*> intermediate_results; // �м������
	ActivationFun activation_fn; // ���������
};

inline Tensor* FCLayer::GetInputs() const
{
	return inputs;
}

inline Tensor* FCLayer::GetOutputs() const
{
	return outputs;
}

inline Tensor* FCLayer::GetWeights() const
{
	return weights;
}

inline Tensor* FCLayer::GetBias() const
{
	return bias;
}

inline FCLayer::ActivationFun FCLayer::GetActivationFn() const
{
	return activation_fn;
}


#endif //_FCLAYER_H_
