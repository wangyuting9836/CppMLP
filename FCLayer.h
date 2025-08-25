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
	enum ActivationFun{NONE, RELU}; // 激活函数

	// 构造函数：创建层并构建计算图
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

	// 参数更新
	void update_parameters(float learn_rate) const;

	// 初始化参数
	void init_parameters(float mean, float variance) const;
 private:
	Tensor* inputs;  //n*1 输入张量
	Tensor* outputs; //q*1 输出张量
	Tensor* weights; //n*q 权重矩阵
	Tensor* bias;	 //q*1 偏置向量
	std::vector<Tensor*> intermediate_results; // 中间计算结果
	ActivationFun activation_fn; // 激活函数类型
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
