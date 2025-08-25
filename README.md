# 基于C++实现从0实现全连接神经网络

**Author:** Yuting Wang

**Date:** 2025-08-25

**Link:** https://zhuanlan.zhihu.com/p/1943256818953467258



目录

收起

1\. 前言

2\. 整体设计架构

2.1 Tensor类：计算图的核心

2.2 FCLayer类：全连接层的实现

2.3 Net类：神经网络的组合与管理

3\. 关键技术点

3.1 计算图与自动微分

3.2 模块化设计

4\. 测试

6.1 线性回归

6.2 MNIST

## 1\. 前言

深度学习框架虽然现在大多基于Python，但其底层核心无一例外都是通过C++实现的高效计算。本文将分享我使用C++实现的一个轻量级全连接神经网络框架，重点介绍设计思路、关键技术细节以及实现方法。

## 2\. 整体设计架构

我的神经网络实现主要包含三个核心类：Tensor（张量）、FCLayer（全连接层）和Net（神经网络）。整个框架采用计算图的思想，使用二叉树结构表示前向计算和反向传播过程。

### 2.1 Tensor类：计算图的核心

Tensor类不仅是数据的容器，更是计算图的基本节点。每个Tensor节点可以记录它的操作类型、左右子节点，从而实现自动微分。

```cpp
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
    
    // 前向计算：根据操作类型计算当前节点的值
    void forward_calculate_data();
    
    // 反向传播：计算左子节点的梯度
    void backward_calculate_left_grad();
    
    // 反向传播：计算右子节点的梯度
    void backward_calculate_right_grad();
    
    // 获取和设置数据、梯度
    const std::vector<std::vector<float>>& get_grad() const;
    std::vector<std::vector<float>>& get_data();
    void set_data(const std::vector<float>& value);
    
    // 清零梯度
    void zero_grad();
    
    // 清零梯度
    void zero_grad();

    // 设置计算图关系
    void set_left(Tensor* l);
    void set_right(Tensor* r);
    void set_calculate_fn(CalculateFun fn);
    
private:
    std::vector<std::vector<float>> data;    // 数据存储
    std::vector<std::vector<float>> grad;    // 梯度存储
    std::vector<float> soft_max_data;        // Softmax结果（用于交叉熵）
    Tensor* left = nullptr;                  // 左子节点
    Tensor* right = nullptr;                 // 右子节点
    CalculateFun calculate_fn = LEAF;        // 计算操作类型
    int row_dimension, col_dimension;        // 维度信息
    bool require_grad = false;               // 是否需要计算梯度
};
```

**计算图与自动微分**

Tensor类的核心在于它构成了计算图的基本单元。每个Tensor节点知道如何根据子节点的值计算自己的值（前向传播），以及如何将梯度传播回子节点（反向传播）。

以矩阵乘法为例，前向计算如下：

```cpp
void Tensor::forward_calculate_data()
{
	switch (calculate_fn)
	{
	//...
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
	//...
	}
}
```

反向传播时，根据当前Tensor节点的信息和左右孩子Tensor节点的data计算左右孩子Tensor节点的梯度。

以矩阵乘法为例，反向传播如下：

```cpp
void Tensor::backward_calculate_left_grad()
{
	switch (this->calculate_fn)
	{
	//...
	case CalculateFun::MATRIX_MUL_VECTOR:
		for (int i = 0; i < this->col_dimension; ++i)
		{
			for (int j = 0; j < this->left->col_dimension; ++j)
			{
				this->left->grad[i][j] += this->grad[0][i] * this->right->data[0][j];
			}
		}
		break;
	//...
	}
}

void Tensor::backward_calculate_right_grad()
{
	switch (this->calculate_fn)
	{
	//...
	case MATRIX_MUL_VECTOR:
		for (int i = 0; i < this->right->col_dimension; ++i)
		{
			for (int j = 0; j < this->col_dimension; ++j)
			{
				this->right->grad[0][i] += this->grad[0][j] * this->left->data[j][i];
			}
		}
		break;
	//...
	}
}
```

### 2.2 FCLayer类：全连接层的实现

FCLayer类封装了一个完整的全连接层，包括权重矩阵、偏置向量以及激活函数。

```cpp
class FCLayer 
{
public:
    enum ActivationFun { NONE, RELU };  // 激活函数类型
    
    // 构造函数：创建层并构建计算图
    FCLayer(int num_of_inputs, int num_of_outputs, ActivationFun fn);
    
    // 前向计算
    void calculate() const;
    
    // 参数更新
    void update_parameters(float learn_rate) const;
    
    // 初始化参数
    void init_parameters(float mean, float variance) const;
    
    // 获取输入输出Tensor
    Tensor* GetInputs() const;
    Tensor* GetOutputs() const;
    
private:
    Tensor* inputs;   // 输入张量
    Tensor* outputs;  // 输出张量
    Tensor* weights;  // 权重矩阵
    Tensor* bias;     // 偏置向量
    std::vector<Tensor*> intermediate_results;  // 中间计算结果
    ActivationFun activation_fn;  // 激活函数类型
};
```

**层的计算图构建**

在构造函数中，FCLayer会构建完整的计算图：

```cpp
FCLayer::FCLayer(int num_of_inputs, int num_of_outputs, ActivationFun fn) : activation_fn(fn) 
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
```

这样，每个全连接层都形成了一个小的计算图子网，多个层连接起来就形成了完整的神经网络计算图。上面的构造过程中把所有中间节点都塞进 intermediate\_results。

### 2.3 Net类：神经网络的组合与管理

Net类负责组合多个全连接层，并添加损失函数，形成完整的神经网络。

```cpp
class Net 
{
public:
    enum LossFun { MEAN_SQUARED, CROSS_ENTROPY };  // 损失函数类型
    
    Net();
    
    // 添加网络层
    void add_layer(int num_of_inputs, int num_of_outputs, 
                  FCLayer::ActivationFun fn);
    
    // 设置损失函数
    void set_loss_function(LossFun fn);
    
    // 训练和预测
    float train(const std::vector<std::vector<float>>& input_value, 
                const std::vector<std::vector<float>>& label_value) const;
    float forward(const std::vector<float>& input_value, 
                 const std::vector<float>& label_value) const;
    
    // 反向传播
    void backward() const;
    
    // 参数更新
    void SGD(float learn_rate) const;
    
private:
    std::vector<FCLayer*> layers;  // 网络层集合
    Tensor* y_label;               // 标签张量
    Tensor* loss;                  // 损失张量
};
```

**网络构建与损失函数**

网络通过逐层添加的方式构建，并最终添加损失函数：

```cpp
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
```

**反向传播算法**

反向传播采用深度优先遍历计算图的方式，使用栈结构实现：

```cpp
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
```

这种实现方式的优点是清晰易懂，且与计算图的结构完全对应。

## 3\. 关键技术点

### 3.1 计算图与自动微分

本文实现的核心是计算图概念。每个Tensor节点不仅存储数据，还知道如何计算自己的值和如何传播梯度。这种设计使得添加新的计算操作变得容易，只需实现前向计算和反向传播方法即可。

### 3.2 模块化设计

通过将神经网络分解为Tensor、FCLayer和Net三个层次，实现了良好的模块化。Tensor负责基本计算，FCLayer负责层内计算，Net负责整体网络组合。这种设计使得代码易于理解和扩展。

## 4\. 测试

用线性回归和Minst手写数字识别进行了测试，测代码在main.cpp中。

### 4.1 线性回归

-   网络结构：`2 → 1`（无激活）
-   Loss：MSE
-   观察：`w1, w2, b` 快速逼近真实值 `2, 3, 7`

### 4.2 MNIST

-   网络结构：`784 → 256(ReLU) → 10`
-   Loss：CrossEntropy
-   Batch size：10