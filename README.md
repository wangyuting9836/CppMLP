# ����C++ʵ�ִ�0ʵ��ȫ����������

**Author:** Yuting Wang

**Date:** 2025-08-25

**Link:** https://zhuanlan.zhihu.com/p/1943256818953467258



Ŀ¼

����

1\. ǰ��

2\. ������Ƽܹ�

2.1 Tensor�ࣺ����ͼ�ĺ���

2.2 FCLayer�ࣺȫ���Ӳ��ʵ��

2.3 Net�ࣺ���������������

3\. �ؼ�������

3.1 ����ͼ���Զ�΢��

3.2 ģ�黯���

4\. ����

6.1 ���Իع�

6.2 MNIST

## 1\. ǰ��

���ѧϰ�����Ȼ���ڴ�����Python������ײ������һ���ⶼ��ͨ��C++ʵ�ֵĸ�Ч���㡣���Ľ�������ʹ��C++ʵ�ֵ�һ��������ȫ�����������ܣ��ص�������˼·���ؼ�����ϸ���Լ�ʵ�ַ�����

## 2\. ������Ƽܹ�

�ҵ�������ʵ����Ҫ�������������ࣺTensor����������FCLayer��ȫ���Ӳ㣩��Net�������磩��������ܲ��ü���ͼ��˼�룬ʹ�ö������ṹ��ʾǰ�����ͷ��򴫲����̡�

### 2.1 Tensor�ࣺ����ͼ�ĺ���

Tensor�಻�������ݵ����������Ǽ���ͼ�Ļ����ڵ㡣ÿ��Tensor�ڵ���Լ�¼���Ĳ������͡������ӽڵ㣬�Ӷ�ʵ���Զ�΢�֡�

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
        VECTOR_RELU,        // ReLU����
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
    std::vector<float> soft_max_data;        // Softmax��������ڽ����أ�
    Tensor* left = nullptr;                  // ���ӽڵ�
    Tensor* right = nullptr;                 // ���ӽڵ�
    CalculateFun calculate_fn = LEAF;        // �����������
    int row_dimension, col_dimension;        // ά����Ϣ
    bool require_grad = false;               // �Ƿ���Ҫ�����ݶ�
};
```

**����ͼ���Զ�΢��**

Tensor��ĺ��������������˼���ͼ�Ļ�����Ԫ��ÿ��Tensor�ڵ�֪����θ����ӽڵ��ֵ�����Լ���ֵ��ǰ�򴫲������Լ���ν��ݶȴ������ӽڵ㣨���򴫲�����

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

���򴫲�ʱ�����ݵ�ǰTensor�ڵ����Ϣ�����Һ���Tensor�ڵ��data�������Һ���Tensor�ڵ���ݶȡ�

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

### 2.2 FCLayer�ࣺȫ���Ӳ��ʵ��

FCLayer���װ��һ��������ȫ���Ӳ㣬����Ȩ�ؾ���ƫ�������Լ��������

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
    
    // ��ȡ�������Tensor
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

�ڹ��캯���У�FCLayer�ṹ�������ļ���ͼ��

```cpp
FCLayer::FCLayer(int num_of_inputs, int num_of_outputs, ActivationFun fn) : activation_fn(fn) 
{
    // ��������Tensor
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

### 2.3 Net�ࣺ���������������

Net�ฺ����϶��ȫ���Ӳ㣬�������ʧ�������γ������������硣

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
	loss->one_grad(); // ������ʧ�������ݶ�Ϊ1
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

����ʵ�ֵĺ����Ǽ���ͼ���ÿ��Tensor�ڵ㲻���洢���ݣ���֪����μ����Լ���ֵ����δ����ݶȡ��������ʹ������µļ������������ף�ֻ��ʵ��ǰ�����ͷ��򴫲��������ɡ�

### 3.2 ģ�黯���

ͨ����������ֽ�ΪTensor��FCLayer��Net������Σ�ʵ�������õ�ģ�黯��Tensor����������㣬FCLayer������ڼ��㣬Net��������������ϡ��������ʹ�ô�������������չ��

## 4\. ����

�����Իع��Minst��д����ʶ������˲��ԣ��������main.cpp�С�

### 4.1 ���Իع�

-   ����ṹ��`2 �� 1`���޼��
-   Loss��MSE
-   �۲죺`w1, w2, b` ���ٱƽ���ʵֵ `2, 3, 7`

### 4.2 MNIST

-   ����ṹ��`784 �� 256(ReLU) �� 10`
-   Loss��CrossEntropy
-   Batch size��10