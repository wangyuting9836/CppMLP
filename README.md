# 基于C++实现从0实现全连接神经网络

**Author:** Yuting Wang

**Date:** 2025-08-25

**Link:** https://zhuanlan.zhihu.com/p/1943256818953467258



目录

收起

1\. 前言

2\. 整体设计架构

2.1 Tensor 类：计算图的核心

2.2 FCLayer 类：全连接层的实现

2.3 Net 类：神经网络的组合与管理

3\. 关键技术点

3.1 计算图与自动微分

3.2 模块化设计

4\. 测试

4.1 线性回归

4.2 MNIST

## 1\. 前言

深度学习框架虽然现在大多基于 Python，但其底层核心无一例外都是通过 C++实现的高效计算。本文将分享我使用 C++实现的一个轻量级全连接神经网络框架，重点介绍设计思路、关键技术细节以及实现方法。

## 2\. 整体设计架构

我的神经网络实现主要包含三个核心类：Tensor（张量）、FCLayer（全连接层）和 Net（神经网络）。整个框架采用计算图的思想，使用二叉树结构表示前向计算和反向传播过程。

### 2.1 Tensor 类：计算图的核心

Tensor 类不仅是数据的容器，更是计算图的基本节点。每个 Tensor 节点可以记录它的操作类型、左右子节点，从而实现自动微分。

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
        VECTOR_RELU,        // ReLU 激活
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
    std::vector<float> soft_max_data;        // Softmax 结果（用于交叉熵）
    Tensor* left = nullptr;                  // 左子节点
    Tensor* right = nullptr;                 // 右子节点
    CalculateFun calculate_fn = LEAF;        // 计算操作类型
    int row_dimension, col_dimension;        // 维度信息
    bool require_grad = false;               // 是否需要计算梯度
};
```

**计算图与自动微分**

Tensor 类的核心在于它构成了计算图的基本单元。每个 Tensor 节点知道如何根据子节点的值计算自己的值（前向传播），以及如何将梯度传播回子节点（反向传播）。

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

反向传播时，根据当前 Tensor 节点的信息和左右孩子 Tensor 节点的 data 计算左右孩子 Tensor 节点的梯度。

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

### 2.2 FCLayer 类：全连接层的实现

FCLayer 类封装了一个完整的全连接层，包括权重矩阵、偏置向量以及激活函数。

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
    
    // 获取输入输出 Tensor
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

在构造函数中，FCLayer 会构建完整的计算图：

```cpp
FCLayer::FCLayer(int num_of_inputs, int num_of_outputs, ActivationFun fn) : activation_fn(fn) 
{
    // 创建各种 Tensor
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

### 2.3 Net 类：神经网络的组合与管理

Net 类负责组合多个全连接层，并添加损失函数，形成完整的神经网络。

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
	loss->one_grad(); // 设置损失函数的梯度为 1
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

本文实现的核心是计算图概念。每个 Tensor 节点不仅存储数据，还知道如何计算自己的值和如何传播梯度。这种设计使得添加新的计算操作变得容易，只需实现前向计算和反向传播方法即可。

### 3.2 模块化设计

通过将神经网络分解为 Tensor、FCLayer 和 Net 三个层次，实现了良好的模块化。Tensor 负责基本计算，FCLayer 负责层内计算，Net 负责整体网络组合。这种设计使得代码易于理解和扩展。

## 4\. 测试

用线性回归和 Mnist 手写数字识别进行了测试，代码在 main.cpp 中。

### 4.1 线性回归

-   网络结构：2 → 1（无激活）
-   Loss：MSE

```cpp
// 线性回归
void linear_regression()
{
	// 均匀分布 U(0, 10)
	std::uniform_real_distribution<float> u_dist(0.0f, 10.0f);
	// 正态分布 N(0, 1)
	std::normal_distribution<float> n_dist(0.0f, 1.0f);

	int train_num = 40000;
	int test_num = 10000;

	std::vector<std::vector<float>> training_datas(train_num);
	std::vector<std::vector<float>> training_labels(train_num);
	std::vector<std::vector<float>> test_datas(test_num);
	std::vector<std::vector<float>> test_labels(test_num);

	float true_w1 = 2.0;
	float true_w2 = 3.0;
	float true_b = 7.0;
	for (int i = 0; i < train_num; ++i)
	{
		float x1 = u_dist(rand_generator());
		float x2 = u_dist(rand_generator());
		// 带噪声的值
		float y = true_w1 * x1 + true_w2 * x2 + true_b + n_dist(rand_generator());
		training_datas[i].emplace_back(x1);
		training_datas[i].emplace_back(x2);
		training_labels[i].emplace_back(y);
	}

	for (int i = 0; i < test_num; ++i)
	{
		float x1 = u_dist(rand_generator());
		float x2 = u_dist(rand_generator());
		// 带噪声的值
		float y = true_w1 * x1 + true_w2 * x2 + true_b + n_dist(rand_generator());
		test_datas[i].emplace_back(x1);
		test_datas[i].emplace_back(x2);
		test_labels[i].emplace_back(y);
	}

	const int batch_size = 10;
	float lr = 0.001;

	// 构建网络
	Net nn;
	nn.add_layer(2, 1, FCLayer::ActivationFun::NONE);
	nn.set_loss_function(Net::LossFun::MEAN_SQUARED);
	nn.init_parameters(0.0f, 0.02f);

	std::vector<std::vector<float>> batch_training_data;
	std::vector<std::vector<float>> batch_training_label;

	std::vector<int> data_index_vec(train_num, 0);
	std::iota(data_index_vec.begin(), data_index_vec.end(), 0);

	// 训练
	for (int epoch = 0; epoch < 20; ++epoch)
	{
		float train_loss_value = 0.0f;
		std::shuffle(begin(data_index_vec), end(data_index_vec), rand_generator());

		int remain_data_num = train_num;
		int data_index = 0;
		while (remain_data_num > 0)
		{
			batch_training_data.clear();
			batch_training_label.clear();
			for (int k = 0; k < std::min(batch_size, remain_data_num); ++k)
			{
				int i = data_index_vec[data_index];
				batch_training_data.emplace_back(training_datas[i]);
				batch_training_label.emplace_back(training_labels[i]);
				++data_index;
				remain_data_num -= batch_size;
			}
			train_loss_value += nn.train(batch_training_data, batch_training_label);
			nn.SGD(lr);
			nn.zero_grad();
		}
		nn.printParameters();
		std::cout << "loss: " << train_loss_value / static_cast<float >(train_num) << std::endl;
	}

	float train_lose = 0.0f;
	int train_acc_num = 0;
	for (int i = 0; i < train_num; ++i)
	{
		train_lose += nn.forward(training_datas[i], training_labels[i]);
		const std::vector<float>& y_hat = nn.get_y_hat();
		if (fabs(y_hat[0] - training_labels[i][0]) < 2)
		{
			++train_acc_num;
		}
	}
	std::cout << "train_acc: " << static_cast<float>(train_acc_num) / static_cast<float>(train_num) << ", train_loss: "
			  << train_lose / static_cast<float>(train_num) << std::endl;

	float test_lose = 0.0f;
	int test_acc_num = 0;
	for (int i = 0; i < test_num; ++i)
	{
		test_lose += nn.forward(test_datas[i], test_labels[i]);
		const std::vector<float>& y_hat = nn.get_y_hat();
		if (fabs(y_hat[0] - test_labels[i][0]) < 2)
		{
			++test_acc_num;
		}
	}
	std::cout << "test_acc: " << static_cast<float>(test_acc_num) / static_cast<float>(test_num) << ", test_loss: "
			  << test_lose / static_cast<float>(test_num) << std::endl;
}
```

运行结果如下，可以发现w1, w2, b 快速逼近真实值 2, 3, 7

```powershell
--------------
layer:1, wights: 2.33403, 3.29061, bias: 3.27568,
loss: 0.416595
layer:1, wights: 2.17192, 3.12953, bias: 4.91397,
loss: 0.117269
layer:1, wights: 2.10181, 3.1077, bias: 5.78781,
loss: 0.071327
layer:1, wights: 2.03439, 2.96338, bias: 6.30105,
loss: 0.0589689
layer:1, wights: 2.05085, 3.0607, bias: 6.62082,
loss: 0.0534521
layer:1, wights: 2.02625, 3.04964, bias: 6.79094,
loss: 0.0537032
layer:1, wights: 1.97147, 2.95098, bias: 6.87258,
loss: 0.0523339
layer:1, wights: 2.06847, 3.02762, bias: 6.93528,
loss: 0.0521302
layer:1, wights: 2.03656, 2.99858, bias: 6.98488,
loss: 0.0534857
layer:1, wights: 2.02882, 3.01663, bias: 6.9726,
loss: 0.0519825
layer:1, wights: 2.00387, 3.00623, bias: 6.98899,
loss: 0.0525265
layer:1, wights: 1.98731, 2.9883, bias: 7.00187,
loss: 0.0511458
layer:1, wights: 2.01063, 2.99602, bias: 6.97834,
loss: 0.052224
layer:1, wights: 2.01435, 2.99509, bias: 6.96036,
loss: 0.0519759
layer:1, wights: 1.98023, 2.9879, bias: 6.96605,
loss: 0.0516643
layer:1, wights: 2.00794, 3.02258, bias: 7.00255,
loss: 0.0531646
layer:1, wights: 1.96356, 2.97535, bias: 7.00513,
loss: 0.0520313
layer:1, wights: 1.9812, 2.95516, bias: 7.0263,
loss: 0.0504542
layer:1, wights: 2.03982, 3.06506, bias: 7.01602,
loss: 0.0530085
layer:1, wights: 2.01129, 3.00262, bias: 7.01035,
loss: 0.051509
train_acc: 0.952925, train_loss: 0.507206
test_acc: 0.953, test_loss: 0.504345
--------------
```

### 4.2 MNIST

-   网络结构：784 → 256(ReLU) → 10
-   Loss：CrossEntropy
-   Batch size：10

```cpp
// 识别手写数字，使用MNIST数据集训练
void MNIST_DATA_mlp()
{
	MNIST_DATA_set dataset = read_MNIST_DATA();

	int train_num = static_cast<int>(dataset.training_images.size());
	int test_num = static_cast<int>(dataset.test_images.size());

	const int batch_size = 10;
	float lr = 0.001;

	// 构建网络
	Net nn;
	nn.add_layer(784, 256, FCLayer::ActivationFun::RELU);
	nn.add_layer(256, 10, FCLayer::ActivationFun::NONE);
	nn.set_loss_function(Net::LossFun::CROSS_ENTROPY);
	nn.init_parameters(0.0f, 0.02f);

	std::vector<std::vector<float>> batch_training_data;
	std::vector<std::vector<float>> batch_training_label;

	std::vector<int> data_index_vec(train_num, 0);
	std::iota(data_index_vec.begin(), data_index_vec.end(), 0);

	// 训练
	for (int epoch = 0; epoch < 10; ++epoch)
	{
		float train_loss_value = 0.0f;
		std::shuffle(begin(data_index_vec), end(data_index_vec), rand_generator());

		int remain_data_num = train_num;
		int data_index = 0;
		while (remain_data_num > 0)
		{
			batch_training_data.clear();
			batch_training_label.clear();
			for (int k = 0; k < std::min(batch_size, remain_data_num); ++k)
			{
				int i = data_index_vec[data_index];
				batch_training_data.emplace_back(dataset.training_images[i]);
				batch_training_label.emplace_back(dataset.training_labels[i]);
				++data_index;
				remain_data_num -= batch_size;
			}
			train_loss_value += nn.train(batch_training_data, batch_training_label);
			nn.SGD(lr);
			nn.zero_grad();
		}
		std::cout << "loss: " << train_loss_value / static_cast<float >(train_num) << std::endl;
	}

	// 在训练集上的准确率
	float train_lose = 0.0f;
	int train_acc_num = 0;
	for (int i = 0; i < train_num; ++i)
	{
		nn.forward(dataset.training_images[i], dataset.training_labels[i]);
		train_lose += nn.get_loss_value();
		const std::vector<float>& y_hat = nn.get_y_hat_softmax();
		int type1 = static_cast<int>(std::max_element(y_hat.begin(), y_hat.end()) - y_hat.begin());
		int type2 = static_cast<int>(
			std::max_element(dataset.training_labels[i].begin(), dataset.training_labels[i].end())
			- dataset.training_labels[i].begin());

		if (type1 == type2)
		{
			++train_acc_num;
		}
	}
	std::cout << "train_acc: " << static_cast<float>(train_acc_num) / static_cast<float>(train_num) << ", train_loss: "
			  << train_lose / static_cast<float>(train_num) << std::endl;

	// 在测试集上的准确率
	float test_lose = 0.0f;
	int test_acc_num = 0;
	for (int i = 0; i < test_num; ++i)
	{
		nn.forward(dataset.test_images[i], dataset.test_labels[i]);
		test_lose += nn.get_loss_value();
		const std::vector<float>& y_hat = nn.get_y_hat_softmax();
		int type1 = static_cast<int>(std::max_element(y_hat.begin(), y_hat.end()) - y_hat.begin());
		int type2 = static_cast<int>(std::max_element(dataset.test_labels[i].begin(), dataset.test_labels[i].end())
									 - dataset.test_labels[i].begin());
		if (type1 == type2)
		{
			++test_acc_num;
		}
	}
	std::cout << "test_acc: " << static_cast<float>(test_acc_num) / static_cast<float>(test_num) << ", test_loss: "
			  << test_lose / static_cast<float>(test_num) << std::endl;
}
```

运行结果如下

```powershell
--------------
loss: 0.219204
loss: 0.200041
loss: 0.183056
loss: 0.169217
loss: 0.158061
loss: 0.147437
loss: 0.138557
loss: 0.130949
loss: 0.124997
loss: 0.120594
train_acc: 0.781367, train_loss: 1.17369
test_acc: 0.7945, test_loss: 1.15192
```