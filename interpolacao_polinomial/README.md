# Métodos Numéricos - Interpolação Polinomial

Este projeto implementa métodos de interpolação polinomial com foco em **Splines Cúbicos** para aproximação de funções. O projeto foi desenvolvido como parte da disciplina de Métodos Numéricos I.

## 📋 Descrição

O projeto implementa diferentes tipos de interpolação por splines cúbicos:
- **Spline Cúbico Natural** (condições de contorno naturais)
- **Spline Cúbico Fixado** (condições de contorno com derivadas especificadas)

### Função de Teste
A função utilizada para validação é `f(x) = cos(πx)` no intervalo `[0, 1]`, que possui propriedades matemáticas conhecidas:
- `f'(0.5) = -π`
- `f''(0.5) = 0`
- `∫₀¹ cos(πx) dx = 0`

## 🚀 Funcionalidades

- ✅ Interpolação por splines cúbicos naturais
- ✅ Interpolação por splines cúbicos fixados
- ✅ Cálculo de derivadas dos splines
- ✅ Integração numérica dos splines
- ✅ Comparação de erros absolutos e relativos
- ✅ Análise de convergência com diferentes densidades de pontos

## 📂 Estrutura do Projeto

```
interpolacao_polinomial/
├── src/
│   ├── methods/
│   │   └── curve_adjusting.py     # Implementação dos splines cúbicos
│   ├── utils/
│   │   └── parser.py              # Utilitários para parsing e derivação
│   ├── projeto_2.ipynb            # Notebook principal com análises
│   └── README.md
```

## 🛠️ Tecnologias Utilizadas

- **Python 3.x**
- **NumPy** - Computação numérica
- **SciPy** - Integração numérica
- **SymPy** - Computação simbólica
- **Pandas** - Manipulação de dados
- **Jupyter Notebook** - Ambiente de desenvolvimento

## 📊 Experimentos Realizados

### Teste A: 5 Pontos
Pontos de interpolação: `x = [0, 0.25, 0.5, 0.75, 1.0]`

### Teste B: 9 Pontos
Pontos de interpolação: `x = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]`

### Métricas Avaliadas
1. **Erro de Integração**: Comparação com valor exato (0)
2. **Erro da Primeira Derivada**: Comparação de `f'(0.5)` com `-π`
3. **Erro da Segunda Derivada**: Comparação de `f''(0.5)` com `0`

## 🔬 Resultados Principais

### Observações
- **Splines Fixados** geralmente apresentam menor erro que splines naturais
- **Maior densidade de pontos** reduz significativamente os erros
- **Derivadas** são calculadas com alta precisão usando diferenciação simbólica

### Conclusões
- Para funções suaves como `cos(πx)`, splines fixados com conhecimento das derivadas nas bordas oferecem melhor aproximação
- O aumento do número de pontos melhora a precisão de forma consistente

## 🚀 Como Executar

1. **Clone o repositório**
```bash
git clone <url-do-repositorio>
cd interpolacao_polinomial
```

2. **Instale as dependências**
```bash
pip install numpy scipy sympy pandas jupyter
```

3. **Execute o notebook**
```bash
jupyter notebook src/projeto_2.ipynb
```

## 📋 Exemplo de Uso

```python
from methods.curve_adjusting import cubic_splines
from utils.parser import evaluate_one_variable
import numpy as np

# Definir pontos
x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
y = np.array([evaluate_one_variable("cos(pi * x)", xi) for xi in x])

# Spline natural
natural_coef = cubic_splines(x, y)

# Spline fixado
fixed_coef = cubic_splines(x, y, dx_0=0, dx_n=0)
```

## 📈 Análise de Erros

O projeto inclui análise detalhada dos erros:
- **Erro Absoluto**: `|valor_exato - valor_calculado|`
- **Erro Relativo**: `|valor_exato - valor_calculado| / |valor_exato|`

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para:
- Reportar bugs
- Sugerir melhorias
- Adicionar novos métodos de interpolação
- Melhorar a documentação

## 📝 Licença

Este projeto está sob licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 👨‍💻 Autores
**Igor Sousa dos Santos Santana**
- Email: igorssant@hotmail.com | issantos.ppgmc@uesc.br
- GitHub: [@igorSantana](https://github.com/issant)
**Matheus Santos Silva**
- Email: mssilva.ppgmc@uesc.br
- GitHub: [@matheusssilva991](https://github.com/matheusssilva991)

## 📚 Referências

- Burden, R. L., & Faires, J. D. (2010). *Numerical Analysis*
- Press, W. H., et al. (2007). *Numerical Recipes: The Art of Scientific Computing*
- Documentação do SciPy: https://docs.scipy.org/
