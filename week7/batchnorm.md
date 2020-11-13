# Batch normalization

deep learning에는 여러 번의 계단식 발전이 있었습니다.  
첫 번째는 ReLU의 발견이죠. activation function의 성능을 비약적으로 끌어올렸습니다.
두 번째는 CNN layer입니다. CNN은 아직도 대다수의 deeplearning network에서 압도적인 성능을 뽐내고 있습니다.  
세 번째는 Adam을 비롯한 optimization function입니다. 초기 deep learning은 hyperparameter를 찾는 것이 굉장히 어려웠는데, 이 문제를 빠르게 해결해 줄 수 있게 되었습니다.  
물론 이외에도 다양한 발전들이 많았습니다. 하지만 오늘 시간에는 그 중에서도 network가 엄청나게 빠른 속도로 학습하는 것을 도와주는, Batch Normalization에 대해 알아보겠습니다.  

## Covariance Shift

Deep learning에 있어서 가장 큰 문제 중 하나는 Covariance shift 문제입니다.  
아래 그림을 보면 왼쪽과 오른쪽은 같은 class에 대한 분류를 실시하지만, 모습은 많이 다르죠.  

<img width="1154" alt="스크린샷 2020-11-13 오전 10 51 46" src="https://user-images.githubusercontent.com/57203764/99018917-89afea80-259e-11eb-8e75-84ab5cf8e0a5.png">
  
이를 data plot 해보면 다음과 같이 나타날 수 있을 것입니다.
  
![IMG_1469](https://user-images.githubusercontent.com/57203764/99019048-de536580-259e-11eb-964f-482cd5352f0f.JPG)
  
두 데이터는 파란색 decision boundary를 통해 나눠지지만, 네트워크에게 이러한 학습과정은 쉽지 않습니다. 즉, 데이터의 분포 성질(distribution characteristic)이 다른 데이터를 학습하기 어렵다는 것입니다.  
  
이를 Covariance shift 라고 합니다.  
데이터의 분포 성질이 달라지면 network의 결과값도 달라지고, 이를 학습하기 위해서는 retraining을 해야한다는 것입니다.  
  
이를 해결하기 위해 가장 좋은 방법은 data distribution을 통일시키는 것입니다.  
  
사실 저희는 기본적으로 input data distribution을 일정하게 해주는 작업에 대해 배웠습니다.  
바로 Normalization이죠.  
Normalization은 아래 그림과 같이 input data의 분포를 둥글게 만들어 줍니다. 이러면 학습이 더욱 쉽고 빠르다는 것을 배웠죠.  
![IMG_292ABCAA0F6A-1](https://user-images.githubusercontent.com/57203764/99019342-75b8b880-259f-11eb-8cfb-a0da52d3b9c1.jpeg)
  
하지만 이 input data 가 layer 몇개를 통과하고 난다면, 그 분포가 여전히 일정할까요? 아닙니다. Network는 linear 함수와 non-linear 함수가 섞여 있는 것과 같기 때문에 distribution에는 계속 변화가 일어납니다.(함수를 통과하면 당연히 distribution 바뀜)  
  
![IMG_1C351DB70E17-1](https://user-images.githubusercontent.com/57203764/99019737-579f8800-25a0-11eb-8201-5e23aeea9bc2.jpeg)
  
즉 hidden layer들은 여전히 covariance shift 문제가 발생하였고, 이를 해결하기 위해 hidden unit들을 normalization 해 주는 것이 바로 Batch normalization 입니다.

## Batch Normalization

우리는 training을 진행할 때, data를 mini-batch로 묶어서 사용합니다. 빠른 학습을 위해서죠.  
batch normalization 과정 또한 mini-batch 단위로 진행됩니다.  
어렵게 생각할 것 없고, 그냥 forward propagation 되는 모든 data를 normalization 하는 것입니다.  
  
우리는 보통 Gaussian distribution을 가정합니다.(일반적인 data들에 대해서)  
Gaussian distribution에 대한 Normalization 식은 아래와 같습니다.  
  
![IMG_EA80976CF75C-1](https://user-images.githubusercontent.com/57203764/99020057-178cd500-25a1-11eb-919f-6a52e9a3e31c.jpeg)
  
이를 그대로 적용시키기만 하면 됩니다. 참 쉽죠?  
주의해야 할 사항은 Batch Normalization은 보통 activation function 전에 사용한다는 점입니다.  
그 이유는 무엇일까요?  
ReLU의 비대칭성 때문입니다.  
ReLU의 경우 음수값을 전부 0으로 바꿔버립니다. 0이 되어 버린 값들은 아무 정보도 전달할 수 없죠.  
0 값들까지 Normalization 하는 것이 의미가 없다는 것입니다. 즉, activation function 전에 BN layer를 추가하는 것이 더욱 효율적이라는 것이죠.  
그래서 각 layer 별로 Batch normalization을 적용한 식은 아래와 같습니다.  
  
![IMG_EF0D07D5F204-1](https://user-images.githubusercontent.com/57203764/99020312-9a159480-25a1-11eb-97a7-53ec461fd4ba.jpeg)
  
그냥 모든 mini-batch 속 데이터에 대해서 평균값 구하고, 분산 값 구해서 Normalization 해주면 됩니다.  
그리고는 두개의 training param을 통해 linear transform을 해주게 되는데, 이건 왜 하는 걸까요?  
  
이는 두 가지 문제를 해결하기 위함입니다.  
1. layer의 representation power가 약화된다.
- 모든 layer를 똑같게 만들어 버리면 학습은 쉬워질지 몰라도 당연히 뭔가를 표현하는데 있어서 약점이 있겠죠? 이러한 trade-off를 조금이나마 해결하고자 하는 것입니다.  
2. hidden unit이 항상 zero mean과 unit variance를 가지게 된다.
- 위에서 한 얘기랑 비슷합니다만, 어쨋든 mean 값이 zero인 것은 그다지 정보 전달에 유리하지는 않습니다. variance가 일정한 것도 마찬가지입니다.  
  
이 두가지를 해결하기 위해서 linear transform과정이 진행되고 두 개의 parameter, gamma와 beta는 모두 학습되는 것입니다. 요런식으로요.  
![IMG_6EC33A8F795B-1](https://user-images.githubusercontent.com/57203764/99021035-1d83b580-25a3-11eb-8e8e-dafb048fee07.jpeg)
  
## Batch Normalization Effect

위대한 Batch Norm의 효과를 나열해보도록 하겠습니다.  
  
- training 속도가 엄청나게 빨라진다.  
- Input distribution이 바뀌더라도 stable 하게 대응할 수 있다.  
- Layer 간의 dependency가 줄어든다.  
- **Regularization 효과가 있다**  
  
나머지는 됐고, 마지막 특징에 대해 알아보겠습니다.  
우리는 mini-batch의 데이터 전체에 대해서 평균을 취하고 분산을 구했습니다. 그렇다면 이 평균과 분산이 데이터 전체의 평균과 분산과 일치할까요? 당연히 아니겠죠. full-batch가 아닌 이상 절대 불가능한 일입니다.  
이러한 특성은 Normalization 과정에서의 noise로 다가오게 됩니다.  
하지만 Noisy한 data는 Regularization 효과를 가져옵니다.  
즉, 놀랍게도 Batch Normalization layer를 추가하면 Regularization 없이도 network가 잘 학습되는 것을 볼 수 있습니다.(regularization 왜 배우지...)  
  
하지만 역시나 trade-off가 있습니다.  
batch-size를 늘리면 iteration에 걸리는 시간이 줄어드는 대신 BN에 의한 regularization 효과가 줄어들고,  
batch-size를 줄이면 BN에 의한 regularization 효과가 늘어나는 대신 iteration에 걸리는 시간이 늘어납니다.  
하지만 크게 신경쓰지 말고 데이터가 굉장히 많고, class가 많은 task라면 batch-size는 되는대로 크게 설정하는 것이 좋습니다.(리소스가 허용하는 한계치까지)  
  
# Softmax

## Softmax function 

다음은 softmax function입니다. 이걸 이제야 배우다니 놀랍네요. 보통은 엄청 초반에 나오는 함수인데....  
  
Softmax는 hardmax와 반대되는 개념이라고 생각하면 편합니다.  
Hardmax는 그냥 최대값을 뽑아주는 거죠.  
Softmax는 최대값을 뽑는 거긴 한데, one-hot vector로 뽑지는 않고, 좀 더 완화된 값으로 뽑는다고 보면 됩니다.  
Sigmoid는 계단형 함수를 soft하게 만든 거죠. 그거랑 비슷합니다.
  
![IMG_B1369578C619-1](https://user-images.githubusercontent.com/57203764/99021808-d4346580-25a4-11eb-8302-45bea3ef2b56.jpeg)  
  
식도 간단합니다.  
![IMG_C01F9512F5E3-1](https://user-images.githubusercontent.com/57203764/99021891-01811380-25a5-11eb-82ed-10df20a926ba.jpeg)  
  
간단하게 적용해 주면 그것으로 끝입니다. 

## Softmax task

Softmax는 classification task에서 아주 유용하게 사용됩니다. 왜냐하면 위 식에서도 알 수 있듯이, 결과값의 총 합이 1이기 때문입니다.  
총 합이 1이라는 것은 각 출력값이 class일 확률을 나타낼 수 있다는 말입니다.  
그러면 그냥 출력값 중 최대인 class를 골라주면 classification이 되는 것이죠.  
softmax를 사용하면, hidden layer 없이도 linear 한 classification이 가능합니다.  
  
![IMG_8BFE496D380B-1](https://user-images.githubusercontent.com/57203764/99022072-5fadf680-25a5-11eb-8035-82a733368b92.jpeg)
  
물론 non-linear한 classification을 하려면 hidden layer는 필수입니다.  

## Loss function for Softmax

확률로 나오는 결과에서 사용하는 loss function은 뭐였죠?  
Cross-entropy입니다.  
그냥 사용해 주면 됩니다~