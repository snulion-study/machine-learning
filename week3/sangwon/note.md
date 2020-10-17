# Week 3 노트

#### 왜 선형 함수를 사용하지 않는가 

- 선형함수 여러개를 합성하면 하나의 함수와 다를 바가 없다
- 이러면 hidden layer 를 늘리는 의미가 없으므로
- Activation function 은 비선형함수를 사용한다(sigmoid, tanh, ReLu, eLu, Leakly Relu 등)
- [Activation function 종류](https://reniew.github.io/12/)

#### Backpropagation

각 activation function 에 대한 backpropagation 식을 다 계산해보자.

