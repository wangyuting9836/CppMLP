//
// Created by wangy on 2023/11/29.
//

#ifndef _TENSOR_H_
#define _TENSOR_H_
#include <vector>
#include <algorithm>
class Tensor
{

 public:
	// 计算操作类型枚举
	enum CalculateFun {
		LEAF,               // 叶子节点（输入或参数）
		VECTOR_ADD_VECTOR,  // 向量加法
		VECTOR_SUB_VECTOR,  // 向量减法
		VECTOR_EXP,         // 指数运算
		VECTOR_LOG,         // 对数运算
		VECTOR_RELU,        // ReLU激活
		MATRIX_MUL_VECTOR,  // 矩阵乘向量
		VECTOR_ASSIGN,      // 向量赋值
		VECTOR_CROSSENTROPY,// 交叉熵损失
		VECTOR_SQUARED      // 均方误差损失
	};

	// 构造函数
	explicit Tensor(int row_dim, int col_dim, bool require_grad = false);

	// 反向传播：计算左子节点的梯度
	void backward_calculate_left_grad();

	// 反向传播：计算左子节点的梯度
	void backward_calculate_right_grad();

	// 前向计算：根据操作类型计算当前节点的值
	void forward_calculate_data();

	// 获取和设置数据、梯度
	[[nodiscard]] const std::vector<std::vector<float>>& get_grad() const;
	[[nodiscard]] std::vector<std::vector<float>>& get_data();
	[[nodiscard]] const std::vector<std::vector<float>>& get_data() const;
	[[nodiscard]] const std::vector<float>& get_softmax_data() const;
	void set_data(const std::vector<float>& value);

	// 清零梯度
	void zero_grad();

	// 梯度设为 1，根节点用
	void one_grad();

	[[nodiscard]] Tensor* get_left() const;
	[[nodiscard]] Tensor* get_right() const;
	void set_left(Tensor* l);
	void set_right(Tensor* r);
	void set_calculate_fn(CalculateFun fn);
	[[nodiscard]] int get_row_dimension() const;
	[[nodiscard]] int get_col_dimension() const;
	[[nodiscard]] bool is_require_grad() const;
	[[nodiscard]] CalculateFun get_calculate_fn() const;
 private:
	std::vector<std::vector<float>> data; 	// 数据存储
	std::vector<float> soft_max_data; 		// Softmax结果，用交叉熵做损失函数，反向求导时用到，其他不用
	std::vector<std::vector<float>> grad;	// 梯度存储
	int row_dimension;						// 维度信息
	int col_dimension;						// 维度信息
	Tensor* left = nullptr;					// 左子节点
	Tensor* right = nullptr;				// 右子节点
	bool require_grad = false;
	CalculateFun calculate_fn = LEAF;		// 是否需要计算梯度
};

inline const std::vector<std::vector<float>>& Tensor::get_grad() const
{
	return grad;
}

inline std::vector<std::vector<float>>& Tensor::get_data()
{
	return data;
}

inline const std::vector<std::vector<float>>& Tensor::get_data() const
{
	return data;
}

inline const std::vector<float>& Tensor::get_softmax_data() const
{
	return soft_max_data;
}

inline void Tensor::set_data(const std::vector<float>& value)
{
	std::copy(value.begin(), value.end(), data[0].begin());
}

inline void Tensor::zero_grad()
{
	for(auto& it:grad )
	{
		std::fill(it.begin(), it.end(), 0.0f);
	}
}

inline void Tensor::one_grad()
{
	for(auto& it:grad )
	{
		std::fill(it.begin(), it.end(), 1.0f);
	}
}

inline Tensor* Tensor::get_left() const
{
	return left;
}

inline Tensor* Tensor::get_right() const
{
	return right;
}

inline void Tensor::set_left(Tensor* l)
{
	left = l;
}
inline void Tensor::set_right(Tensor* r)
{
	right = r;
}
inline void Tensor::set_calculate_fn(CalculateFun fn)
{
	calculate_fn = fn;
}
inline int Tensor::get_row_dimension() const
{
	return row_dimension;
}
inline int Tensor::get_col_dimension() const
{
	return col_dimension;
}

inline bool Tensor::is_require_grad() const
{
	return require_grad;
}

inline Tensor::CalculateFun Tensor::get_calculate_fn() const
{
	return calculate_fn;
}
#endif //_TENSOR_H_
