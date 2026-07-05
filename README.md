# NovaGrad

[![PyPI version](https://badge.fury.io/py/novagrad.svg)](https://badge.fury.io/py/novagrad)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


\
\
**NovaGrad**는 JAX와 MLX 백엔드를 완벽하게 추상화하여, 클라우드(Colab TPU/GPU)와 로컬(Mac 통합 메모리) 환경을 자유롭게 넘나드는 하이브리드 딥러닝 프레임워크입니다. 



## NovaGrad의 탄생 과정

1. 혼자 딥러닝 공부를 하며, 직접 라이브러리를 만들고 그걸 통제하는게 기존 라이브러리를 쓰는 것보다 더 편할 거 같아서 만들었습니다.
2. 공부는 어떻게?
    - 책과 인터넷을 통해 공부를 하고, 파이썬으로 실습코드 개조하면서 공부 중입니다.
    - 현재 군대에 있느라, 짬내서 조금씩 공부해나가고 있어요.
3. 이 라이브러리는 실전을 위하기보단, 스스로 공부와 연구를 하기 위해 만든 도구입니다.
    - 기존 프레임워크의 틀에 갇혀있지 않고 자유롭게 코드를 만질 수 있어요.
    - 그만큼 무언가 실험하기 용이합니다.
4. 하드웨어 가속은 어떻게?
    - JAX, MLX를 사용합니다.
    - JAX로 구글의 TPU나 일반 GPU에서의 가속을, MLX로 맥에서의 가속을 지원할 수 있게 만들고 있어요.
    - TPU는 가성비가 있다고 생각해서 쓰고 있습니다.
    - MLX로 맥에서 가속하는 이유는 단순히 제가 맥을 갖고 있는데 이걸 가만히 두긴 아깝다고 생각했어요.

## 설치 방법

```bash
pip install novagrad
```
또는
```bash
pip install git+https://github.com/yeunzu/NovaGrad.git
```


### 주의사항
현재 초기 개발도 끝나지 않은 상태입니다. 따라서 수없이 말도 안되는 버그들이 많다는 점 이해해주시면 좋겠습니다

## 퀵 스타트 (Quick Start)
가장 간단한 다층 퍼셉트론(MLP)으로 MNIST 데이터를 학습하는 예제입니다. 모델이 이런 식으로 구성되게 만들었다 정도의 코드입니다.
```python
import novagrad
from novagrad.models import MyModel
from novagrad.layers import Affine, Relu, Softmax
from novagrad.optimizers import Adam
from novagrad.losses import CrossEntropyError

# 1. 백엔드 자동 감지 및 초기화 (JAX or MLX)
be = novagrad.Backend()

# 2. 모델 조립
model = MyModel(backend=be)
model.add_layer(Affine(input_size=784, output_size=256, optimizer=Adam, backend=be))
model.add_layer(Relu(backend=be))
model.add_layer(Affine(input_size=256, output_size=10, optimizer=Adam, backend=be))
model.add_layer(Softmax(backend=be))

model.set_loss_func(CrossEntropyError(backend=be))

# 3. 학습 시작!
# trainer.fit() 등을 이용해 학습을 경험하세요.
```

## 공식 문서 (Documentation)
레이어별 상세 파라미터, 데이터 증강(Albumentations) 사용법, 그리고 커스텀 모델 구축 가이드는 아래 공식 Notion 페이지에서 확인하실 수 있습니다. 아직은 코드도 완전히 완성되지 않아서, 아무 내용이 없습니다! 빠른 시일 내에 뭐라도 적어놓을게요.

-> [NovaGrad 공식 사용 설명서 (Notion 링크)](https://app.notion.com/p/Home-NovaGrad-Documentation-3944bb67167a80bf845af6676ad69893?source=copy_link)

## 기여하기
버그 리포트, 기능 제안, Pull Request를 언제나 환영합니다!

