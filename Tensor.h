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
	// �����������ö��
	enum CalculateFun {
		LEAF,               // Ҷ�ӽڵ㣨����������
		VECTOR_ADD_VECTOR,  // �����ӷ�
		VECTOR_SUB_VECTOR,  // ��������
		VECTOR_EXP,         // ָ������
		VECTOR_LOG,         // ��������
		VECTOR_RELU,        // ReLU����
		MATRIX_MUL_VECTOR,  // ���������
		VECTOR_ASSIGN,      // ������ֵ
		VECTOR_CROSSENTROPY,// ��������ʧ
		VECTOR_SQUARED      // ���������ʧ
	};

	// ���캯��
	explicit Tensor(int row_dim, int col_dim, bool require_grad = false);

	// ���򴫲����������ӽڵ���ݶ�
	void backward_calculate_left_grad();

	// ���򴫲����������ӽڵ���ݶ�
	void backward_calculate_right_grad();

	// ǰ����㣺���ݲ������ͼ��㵱ǰ�ڵ��ֵ
	void forward_calculate_data();

	// ��ȡ���������ݡ��ݶ�
	[[nodiscard]] const std::vector<std::vector<float>>& get_grad() const;
	[[nodiscard]] std::vector<std::vector<float>>& get_data();
	[[nodiscard]] const std::vector<std::vector<float>>& get_data() const;
	[[nodiscard]] const std::vector<float>& get_softmax_data() const;
	void set_data(const std::vector<float>& value);

	// �����ݶ�
	void zero_grad();

	// �ݶ���Ϊ 1�����ڵ���
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
	std::vector<std::vector<float>> data; 	// ���ݴ洢
	std::vector<float> soft_max_data; 		// Softmax������ý���������ʧ������������ʱ�õ�����������
	std::vector<std::vector<float>> grad;	// �ݶȴ洢
	int row_dimension;						// ά����Ϣ
	int col_dimension;						// ά����Ϣ
	Tensor* left = nullptr;					// ���ӽڵ�
	Tensor* right = nullptr;				// ���ӽڵ�
	bool require_grad = false;
	CalculateFun calculate_fn = LEAF;		// �Ƿ���Ҫ�����ݶ�
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
