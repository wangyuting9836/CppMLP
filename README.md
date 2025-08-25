# ����C++ʵ�ִ�0ʵ��ȫ����������

**Author:** Yuting Wang

**Date:** 2025-08-25

**Link:** https://zhuanlan.zhihu.com/p/1943256818953467258



Ŀ¼

����

1\. ǰ��

2\. ������Ƽܹ�

2.1 Tensor �ࣺ����ͼ�ĺ���

2.2 FCLayer �ࣺȫ���Ӳ��ʵ��

2.3 Net �ࣺ���������������

3\. �ؼ�������

3.1 ����ͼ���Զ�΢��

3.2 ģ�黯���

4\. ����

4.1 ���Իع�

4.2 MNIST

## 1\. ǰ��

���ѧϰ�����Ȼ���ڴ����� Python������ײ������һ���ⶼ��ͨ�� C++ʵ�ֵĸ�Ч���㡣���Ľ�������ʹ�� C++ʵ�ֵ�һ��������ȫ�����������ܣ��ص�������˼·���ؼ�����ϸ���Լ�ʵ�ַ�����

## 2\. ������Ƽܹ�

�ҵ�������ʵ����Ҫ�������������ࣺTensor����������FCLayer��ȫ���Ӳ㣩�� Net�������磩��������ܲ��ü���ͼ��˼�룬ʹ�ö������ṹ��ʾǰ�����ͷ��򴫲����̡�

### 2.1 Tensor �ࣺ����ͼ�ĺ���

Tensor �಻�������ݵ����������Ǽ���ͼ�Ļ����ڵ㡣ÿ�� Tensor �ڵ���Լ�¼���Ĳ������͡������ӽڵ㣬�Ӷ�ʵ���Զ�΢�֡�

```cpp
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
        VECTOR_RELU,        // ReLU ����
        MATRIX_MUL_VECTOR,  // ���������
        VECTOR_ASSIGN,      // ������ֵ
        VECTOR_CROSSENTROPY,// ��������ʧ
        VECTOR_SQUARED      // ���������ʧ
    };
    
    // ���캯��
    explicit Tensor(int row_dim, int col_dim, bool require_grad = false);
    
    // ǰ����㣺���ݲ������ͼ��㵱ǰ�ڵ��ֵ
    void forward_calculate_data();
    
    // ���򴫲����������ӽڵ���ݶ�
    void backward_calculate_left_grad();
    
    // ���򴫲����������ӽڵ���ݶ�
    void backward_calculate_right_grad();
    
    // ��ȡ���������ݡ��ݶ�
    const std::vector<std::vector<float>>& get_grad() const;
    std::vector<std::vector<float>>& get_data();
    void set_data(const std::vector<float>& value);
    
    // �����ݶ�
    void zero_grad();
    
    // �����ݶ�
    void zero_grad();

    // ���ü���ͼ��ϵ
    void set_left(Tensor* l);
    void set_right(Tensor* r);
    void set_calculate_fn(CalculateFun fn);
    
private:
    std::vector<std::vector<float>> data;    // ���ݴ洢
    std::vector<std::vector<float>> grad;    // �ݶȴ洢
    std::vector<float> soft_max_data;        // Softmax ��������ڽ����أ�
    Tensor* left = nullptr;                  // ���ӽڵ�
    Tensor* right = nullptr;                 // ���ӽڵ�
    CalculateFun calculate_fn = LEAF;        // �����������
    int row_dimension, col_dimension;        // ά����Ϣ
    bool require_grad = false;               // �Ƿ���Ҫ�����ݶ�
};
```

**����ͼ���Զ�΢��**

Tensor ��ĺ��������������˼���ͼ�Ļ�����Ԫ��ÿ�� Tensor �ڵ�֪����θ����ӽڵ��ֵ�����Լ���ֵ��ǰ�򴫲������Լ���ν��ݶȴ������ӽڵ㣨���򴫲�����

�Ծ���˷�Ϊ����ǰ��������£�

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

���򴫲�ʱ�����ݵ�ǰ Tensor �ڵ����Ϣ�����Һ��� Tensor �ڵ�� data �������Һ��� Tensor �ڵ���ݶȡ�

�Ծ���˷�Ϊ�������򴫲����£�

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

### 2.2 FCLayer �ࣺȫ���Ӳ��ʵ��

FCLayer ���װ��һ��������ȫ���Ӳ㣬����Ȩ�ؾ���ƫ�������Լ��������

```cpp
class FCLayer 
{
public:
    enum ActivationFun { NONE, RELU };  // ���������
    
    // ���캯���������㲢��������ͼ
    FCLayer(int num_of_inputs, int num_of_outputs, ActivationFun fn);
    
    // ǰ�����
    void calculate() const;
    
    // ��������
    void update_parameters(float learn_rate) const;
    
    // ��ʼ������
    void init_parameters(float mean, float variance) const;
    
    // ��ȡ������� Tensor
    Tensor* GetInputs() const;
    Tensor* GetOutputs() const;
    
private:
    Tensor* inputs;   // ��������
    Tensor* outputs;  // �������
    Tensor* weights;  // Ȩ�ؾ���
    Tensor* bias;     // ƫ������
    std::vector<Tensor*> intermediate_results;  // �м������
    ActivationFun activation_fn;  // ���������
};
```

**��ļ���ͼ����**

�ڹ��캯���У�FCLayer �ṹ�������ļ���ͼ��

```cpp
FCLayer::FCLayer(int num_of_inputs, int num_of_outputs, ActivationFun fn) : activation_fn(fn) 
{
    // �������� Tensor
    inputs = new Tensor(1, num_of_inputs);
    outputs = new Tensor(1, num_of_outputs, true);
    weights = new Tensor(num_of_outputs, num_of_inputs, true);
    bias = new Tensor(1, num_of_outputs, true);
    
    // ��������ͼ��Wx
    auto* w_x = new Tensor(1, num_of_outputs, true);
    w_x->set_left(weights);
    w_x->set_right(inputs);
    w_x->set_calculate_fn(Tensor::CalculateFun::MATRIX_MUL_VECTOR);
    intermediate_results.emplace_back(w_x);
    
    // ��������ͼ��Wx + b
    auto* w_x_b = new Tensor(1, num_of_outputs, true);
    w_x_b->set_left(w_x);
    w_x_b->set_right(bias);
    w_x_b->set_calculate_fn(Tensor::CalculateFun::VECTOR_ADD_VECTOR);
    intermediate_results.emplace_back(w_x_b);
    
    // ��Ӽ����
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

������ÿ��ȫ���Ӳ㶼�γ���һ��С�ļ���ͼ����������������������γ������������������ͼ������Ĺ�������а������м�ڵ㶼���� intermediate\_results��

### 2.3 Net �ࣺ���������������

Net �ฺ����϶��ȫ���Ӳ㣬�������ʧ�������γ������������硣

```cpp
class Net 
{
public:
    enum LossFun { MEAN_SQUARED, CROSS_ENTROPY };  // ��ʧ��������
    
    Net();
    
    // ��������
    void add_layer(int num_of_inputs, int num_of_outputs, 
                  FCLayer::ActivationFun fn);
    
    // ������ʧ����
    void set_loss_function(LossFun fn);
    
    // ѵ����Ԥ��
    float train(const std::vector<std::vector<float>>& input_value, 
                const std::vector<std::vector<float>>& label_value) const;
    float forward(const std::vector<float>& input_value, 
                 const std::vector<float>& label_value) const;
    
    // ���򴫲�
    void backward() const;
    
    // ��������
    void SGD(float learn_rate) const;
    
private:
    std::vector<FCLayer*> layers;  // ����㼯��
    Tensor* y_label;               // ��ǩ����
    Tensor* loss;                  // ��ʧ����
};
```

**���繹������ʧ����**

����ͨ�������ӵķ�ʽ�����������������ʧ������

```cpp
void Net::add_layer(int num_of_inputs, int num_of_outputs, FCLayer::ActivationFun activation_fn)
{
	if (layers.empty())
	{
		// ��һ��
		layers.emplace_back(new FCLayer(num_of_inputs, num_of_outputs, activation_fn));
	}
	else
	{
		// �����㣺��ǰһ��������Ϊ��ǰ�������
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

**���򴫲��㷨**

���򴫲�����������ȱ�������ͼ�ķ�ʽ��ʹ��ջ�ṹʵ�֣�

```cpp
void Net::backward() const
{
	if (!loss->is_require_grad() || loss->get_calculate_fn() == Tensor::LEAF)
	{
		return;
	}

	std::stack<Tensor*> tensor_stack;
	loss->one_grad(); // ������ʧ�������ݶ�Ϊ 1
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
		// �����ӽڵ㴫���ݶ�
		if (left != nullptr && left->is_require_grad())
		{
			cur_tensor->backward_calculate_left_grad();
			tensor_stack.push(left);
		}
		// �����ӽڵ㴫���ݶ�
		if (right != nullptr && right->is_require_grad())
		{
			cur_tensor->backward_calculate_right_grad();
			tensor_stack.push(right);
		}
	}
}
```

����ʵ�ַ�ʽ���ŵ��������׶����������ͼ�Ľṹ��ȫ��Ӧ��

## 3\. �ؼ�������

### 3.1 ����ͼ���Զ�΢��

����ʵ�ֵĺ����Ǽ���ͼ���ÿ�� Tensor �ڵ㲻���洢���ݣ���֪����μ����Լ���ֵ����δ����ݶȡ��������ʹ������µļ������������ף�ֻ��ʵ��ǰ�����ͷ��򴫲��������ɡ�

### 3.2 ģ�黯���

ͨ����������ֽ�Ϊ Tensor��FCLayer �� Net ������Σ�ʵ�������õ�ģ�黯��Tensor ����������㣬FCLayer ������ڼ��㣬Net ��������������ϡ��������ʹ�ô�������������չ��

## 4\. ����

�����Իع�� Mnist ��д����ʶ������˲��ԣ������� main.cpp �С�

### 4.1 ���Իع�

-   ����ṹ��2 �� 1���޼��
-   Loss��MSE

```cpp
// ���Իع�
void linear_regression()
{
	// ���ȷֲ� U(0, 10)
	std::uniform_real_distribution<float> u_dist(0.0f, 10.0f);
	// ��̬�ֲ� N(0, 1)
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
		// ��������ֵ
		float y = true_w1 * x1 + true_w2 * x2 + true_b + n_dist(rand_generator());
		training_datas[i].emplace_back(x1);
		training_datas[i].emplace_back(x2);
		training_labels[i].emplace_back(y);
	}

	for (int i = 0; i < test_num; ++i)
	{
		float x1 = u_dist(rand_generator());
		float x2 = u_dist(rand_generator());
		// ��������ֵ
		float y = true_w1 * x1 + true_w2 * x2 + true_b + n_dist(rand_generator());
		test_datas[i].emplace_back(x1);
		test_datas[i].emplace_back(x2);
		test_labels[i].emplace_back(y);
	}

	const int batch_size = 10;
	float lr = 0.001;

	// ��������
	Net nn;
	nn.add_layer(2, 1, FCLayer::ActivationFun::NONE);
	nn.set_loss_function(Net::LossFun::MEAN_SQUARED);
	nn.init_parameters(0.0f, 0.02f);

	std::vector<std::vector<float>> batch_training_data;
	std::vector<std::vector<float>> batch_training_label;

	std::vector<int> data_index_vec(train_num, 0);
	std::iota(data_index_vec.begin(), data_index_vec.end(), 0);

	// ѵ��
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

���н�����£����Է���w1, w2, b ���ٱƽ���ʵֵ 2, 3, 7

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

-   ����ṹ��784 �� 256(ReLU) �� 10
-   Loss��CrossEntropy
-   Batch size��10

```cpp
// ʶ����д���֣�ʹ��MNIST���ݼ�ѵ��
void MNIST_DATA_mlp()
{
	MNIST_DATA_set dataset = read_MNIST_DATA();

	int train_num = static_cast<int>(dataset.training_images.size());
	int test_num = static_cast<int>(dataset.test_images.size());

	const int batch_size = 10;
	float lr = 0.001;

	// ��������
	Net nn;
	nn.add_layer(784, 256, FCLayer::ActivationFun::RELU);
	nn.add_layer(256, 10, FCLayer::ActivationFun::NONE);
	nn.set_loss_function(Net::LossFun::CROSS_ENTROPY);
	nn.init_parameters(0.0f, 0.02f);

	std::vector<std::vector<float>> batch_training_data;
	std::vector<std::vector<float>> batch_training_label;

	std::vector<int> data_index_vec(train_num, 0);
	std::iota(data_index_vec.begin(), data_index_vec.end(), 0);

	// ѵ��
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

	// ��ѵ�����ϵ�׼ȷ��
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

	// �ڲ��Լ��ϵ�׼ȷ��
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

���н������

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