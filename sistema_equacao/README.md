# Sistemas de Equações Lineares — Métodos Numéricos

Este projeto explora a resolução de sistemas de equações lineares utilizando métodos numéricos diretos e iterativos, com foco em análise de condicionamento, precisão, convergência e desempenho computacional.

## Estrutura do Projeto

- `src/`: Notebooks e scripts principais para experimentos e análise.
- `output/`: Resultados gerados (csv, gráficos).
- `dev_files/`: Materiais de apoio e referências.
- `environment.yml`: Ambiente Conda para reprodução dos experimentos.

## Principais Funcionalidades

- **Geração de matrizes de Hilbert** para análise de condicionamento.
- **Resolução de sistemas** por métodos diretos:
  - Eliminação de Gauss (simples, pivotamento parcial, escalonado, completo)
  - Fatoração LU
- **Resolução por métodos iterativos**:
  - Jacobi
  - Gauss-Seidel
  - Relaxamento (SOR)
- **Cálculo de determinante, número de condição e inversa** das matrizes.
- **Avaliação de precisão** (erro absoluto e relativo), convergência e tempo de execução.
- **Exportação de resultados** para arquivos CSV e geração de gráficos.

## Como Executar

1. **Crie o ambiente Conda:**

   ```sh
   conda env create -f environment.yml
   conda activate mn
   ```

2. **Abra os notebooks em `src/` no VS Code ou Jupyter.**
   - Execute as células para reproduzir os experimentos e gerar os resultados.

## Dependências

Veja [environment.yml](environment.yml) para detalhes. Principais pacotes:

- Python 3.13
- numpy, pandas, sympy
- matplotlib, seaborn
- ipykernel, nbformat

## Resultados

Os resultados dos experimentos (soluções, erros, gráficos de convergência) são salvos em [output/](output/).

## Organização dos Arquivos

- `src/main.ipynb`: Notebook principal com todos os experimentos e análises.
- `src/methods/`: Implementações dos métodos diretos e iterativos.
- `src/utils/`: Funções auxiliares, como geração de matrizes.
- `output/`: Resultados em CSV e imagens dos gráficos.
- `dev_files/`: Materiais de apoio, referências e anotações.

## Exemplos de Saída

- Tabelas de erros absolutos e relativos para cada método e tamanho de matriz.
- Gráficos de convergência dos métodos iterativos.
- Arquivos CSV com os resultados das execuções.

## Referências

- Métodos Numéricos para Engenharia — CAP 257
- Algebra Linear e suas Aplicações — Petronio Pulino

---

> Projeto acadêmico para estudo de métodos numéricos aplicados à resolução de sistemas lineares.
