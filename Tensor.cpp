//
// Created by wangy on 2023/11/29.
//
#include <algorithm>
#include <numeric>
#include <cmath>
#include "Tensor.h"
Tensor::Tensor(int row_dim, int col_dim, bool require_grad)
	: row_dimension(row_dim), col_dimension(col_dim), require_grad(require_grad)
{
	data.resize(row_dimension);
	for (auto& it : data)
	{
		it.resize(col_dimension);
	}

	soft_max_data.resize(col_dimension);

	grad.resize(row_dimension);
	for (auto& it : grad)
	{
		it.resize(col_dimension);
	}
}

void Tensor::backward_calculate_left_grad()
{
	switch (this->calculate_fn)
	{
	case CalculateFun::VECTOR_ADD_VECTOR:
	case CalculateFun::VECTOR_SUB_VECTOR:
		for (int i = 0; i < col_dimension; ++i)
		{
			this->left->grad[0][i] += this->grad[0][i];
		}
		break;
	case CalculateFun::VECTOR_EXP:
		for (int i = 0; i < this->col_dimension; ++i)
		{
			this->left->grad[0][i] += this->grad[0][i] * std::expf(this->left->data[0][i]);
		}
		break;
	case CalculateFun::VECTOR_LOG:
		for (int i = 0; i < this->col_dimension; ++i)
		{
			this->left->grad[0][i] += this->grad[0][i] / this->left->data[0][i];
		}
		break;
	case CalculateFun::VECTOR_RELU:
		for (int i = 0; i < this->col_dimension; ++i)
		{
			this->left->grad[0][i] += this->grad[0][i] * (this->left->data[0][i] < 0.0f ? 0.0f : 1.0f);
		}
		break;
	case CalculateFun::MATRIX_MUL_VECTOR:
		for (int i = 0; i < this->col_dimension; ++i)
		{
			for (int j = 0; j < this->left->col_dimension; ++j)
			{
				this->left->grad[i][j] += this->grad[0][i] * this->right->data[0][j];
			}
		}
		break;
	case CalculateFun::VECTOR_ASSIGN:
		for (int i = 0; i < this->col_dimension; ++i)
		{
			this->left->grad[0][i] += this->grad[0][i];
		}
		break;
	case CalculateFun::VECTOR_CROSSENTROPY:
		for (int j = 0; j < this->left->col_dimension; ++j)
		{
			if (this->right->data[0][j] == 0.0f)
			{
				this->left->grad[0][j] += this->grad[0][0] * this->left->soft_max_data[j];
			}
			else
			{
				this->left->grad[0][j] += this->grad[0][0] * (this->left->soft_max_data[j] - 1);
			}
		}
		break;
	case CalculateFun::VECTOR_SQUARED:
		for (int j = 0; j < this->left->col_dimension; ++j)
		{
			this->left->grad[0][j] = this->left->data[0][j] - this->right->data[0][j];
		}
		break;
	case LEAF:
		break;
	}
}

void Tensor::backward_calculate_right_grad()
{
	switch (this->calculate_fn)
	{
	case CalculateFun::VECTOR_ADD_VECTOR:
		for (int i = 0; i < this->col_dimension; ++i)
		{
			this->right->grad[0][i] += this->grad[0][i];
		}
		break;
	case CalculateFun::VECTOR_SUB_VECTOR:
		for (int i = 0; i < this->col_dimension; ++i)
		{
			this->right->grad[0][i] += -this->grad[0][i];
		}
		break;
	case MATRIX_MUL_VECTOR:
		for (int i = 0; i < this->right->col_dimension; ++i)
		{
			for (int j = 0; j < this->col_dimension; ++j)
			{
				this->right->grad[0][i] += this->grad[0][j] * this->left->data[j][i];
			}
		}
		break;
	case VECTOR_EXP:
	case VECTOR_LOG:
	case VECTOR_RELU:
	case VECTOR_ASSIGN:
	case VECTOR_CROSSENTROPY:
	case VECTOR_SQUARED:
	case LEAF:
		break;
	}
}

void Tensor::forward_calculate_data()
{
	switch (calculate_fn)
	{
	case CalculateFun::LEAF:
		break;
	case CalculateFun::VECTOR_ADD_VECTOR:
		for (int i = 0; i < col_dimension; ++i)
		{
			data[0][i] = left->data[0][i] + right->data[0][i];
		}
		break;
	case CalculateFun::VECTOR_SUB_VECTOR:
		for (int i = 0; i < col_dimension; ++i)
		{
			data[0][i] = left->data[0][i] - right->data[0][i];
		}
		break;
	case CalculateFun::VECTOR_EXP:
		for (int i = 0; i < col_dimension; ++i)
		{
			data[0][i] = std::expf(left->data[0][i]);
		}
		break;
	case CalculateFun::VECTOR_LOG:
		for (int i = 0; i < col_dimension; ++i)
		{
			data[0][i] = std::log(left->data[0][i]);
		}
		break;
	case CalculateFun::VECTOR_RELU:
		for (int i = 0; i < col_dimension; ++i)
		{
			data[0][i] = std::max(0.0f, left->data[0][i]);
		}
		break;
	case CalculateFun::MATRIX_MUL_VECTOR:
		for (int i = 0; i < col_dimension; ++i)
		{
			data[0][i] = 0.0f;
			for (int j = 0; j < left->col_dimension; ++j)
			{
				data[0][i] += left->data[i][j] * right->data[0][j];
			}
		}
		break;
	case CalculateFun::VECTOR_CROSSENTROPY:
	{
		for (int i = 0; i < left->col_dimension; ++i)
		{
			left->soft_max_data[i] = std::expf(left->data[0][i]);
		}
		float sum = std::accumulate(left->soft_max_data.begin(), left->soft_max_data.end(), 0.0f);
		std::for_each(left->soft_max_data.begin(), left->soft_max_data.end(), [=](float& v)
		{
			v = v / sum;
		});
		for (int j = 0; j < left->col_dimension; ++j)
		{
			if (right->data[0][j] == 1.0f)
			{
				data[0][0] = -std::log(left->soft_max_data[j]);
				break;
			}
		}
		break;
	}
	case CalculateFun::VECTOR_SQUARED:
	{
		data[0][0] = 0.0f;
		for (int j = 0; j < left->col_dimension; ++j)
		{
			data[0][0] += (left->data[0][j] - right->data[0][j]) * (left->data[0][j] - right->data[0][j]);
		}
		data[0][0] /= 2.0f;
		break;
	}
	case CalculateFun::VECTOR_ASSIGN:
	{
		for (int i = 0; i < col_dimension; ++i)
		{
			data[0][i] = left->data[0][i];
		}
		break;
	}
	}
}

