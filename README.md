# Treinamento_TF-IDF-SVM — Baseline Clássico para Classificação de Golpes no Facebook

**Artigo relacionado:** *VeritaPlugin: Uma Extensão de Navegador para Detecção Semântica de Fraudes no Facebook* — Universidade Federal Fluminense (UFF)

**Resumo do artigo.** A Engenharia Social em redes sociais explora vulnerabilidades para
iludir usuários, tornando defesas técnicas tradicionais insuficientes. Este trabalho
apresenta o VeritaPlugin, uma extensão de navegador que detecta fraudes no Facebook por
meio de um pipeline híbrido BERTimbau, RAG determinístico e GPT-4o. A arquitetura opera em
conformidade com a LGPD e o resultado apresenta ao usuário a categoria do golpe,
enquadramento legal e ações recomendadas. Para calibração, foi construído e disponibilizado
o dataset BrScamsFacebook, com 450 instâncias de golpes reais do contexto brasileiro. Na
avaliação técnica, o classificador obteve F1-macro de 0,763 ± 0,034 na validação cruzada
k=5, superando o baseline.

**Resumo do artefato.** Este repositório implementa o **baseline clássico** contra o qual
o classificador BERTimbau do **VeritaPlugin** é comparado no artigo: um pipeline
**TF-IDF + LinearSVC** treinado sobre o mesmo dataset **BrScamsFacebook** (450
publicações, 75 por categoria). O ponto central do artefato é o **rigor da comparação**: o
baseline é avaliado com exatamente o mesmo protocolo do modelo neural — validação cruzada
estratificada com k = 5 e `random_state=42` —, de modo que os números são diretamente
comparáveis e a diferença observada não pode ser atribuída a diferenças de partição dos
dados. O script também calcula um classificador por maioria (`DummyClassifier`), que
estabelece o limite inferior de referência, e imprime ao final o comparativo entre os três
modelos. Por usar apenas algoritmos determinísticos com semente fixa, este experimento é
**integralmente reproduzível**: produz valores idênticos a cada execução, em CPU, em menos
de dois minutos.

**Artefato principal:** [VeritaPlugin](https://github.com/brunanoroes/VeritaPlugin)
**Autora:** Bruna Norões — brunanoroes@id.uff.br

---

# Estrutura do readme.md

Este README segue os requisitos mínimos do Comitê Técnico de Artefatos do SBSeg 2026:

| Seção | Conteúdo |
|---|---|
| **Título projeto** | Identificação do artefato, vínculo com o artigo e resumo |
| **Estrutura do readme.md** | Esta seção — mapa do documento e organização do repositório |
| **Selos Considerados** | Selos pleiteados na avaliação |
| **Informações básicas** | Ambiente de execução, requisitos e metodologia |
| **Dependências** | Versões de linguagem, bibliotecas e dataset |
| **Preocupações com segurança** | Riscos para o avaliador e medidas de mitigação |
| **Instalação** | Passo a passo para preparar o ambiente |
| **Teste mínimo** | Execução mínima que demonstra o funcionamento |
| **Experimentos** | Reprodução da reivindicação do artigo |
| **LICENSE** | Licença do artefato |

## Organização do repositório

```
Treinamento_TF-IDF-SVM/
├── BrScamsFacebook.xlsx    # Dataset de entrada (450 instâncias, colunas Mensagem e Categoria)
├── treinamento.py          # Pipeline TF-IDF + LinearSVC, k-fold e comparativo final
├── LICENSE
└── README.md
```

O repositório é intencionalmente minimalista: um único script autocontido, com as etapas
numeradas em comentários (1. Carregamento, 2. Pipeline, 3. k-fold, 4. Relatório
consolidado, 5. Classificador por maioria, 6. Comparativo final).

---

# Selos Considerados

Os selos considerados são: **Artefatos Disponíveis (SeloD)** e **Artefatos Funcionais
(SeloF)**.

| Selo | Onde é atendido neste README |
|---|---|
| **Disponíveis (D)** | Repositório público e estável no GitHub, com este README e licença MIT |
| **Funcionais (F)** | Seções *Dependências* (com versões), *Informações básicas* (ambiente), *Instalação* e *Teste mínimo* |

> Este repositório sustenta a **Reivindicação #2** do artefato principal
> [VeritaPlugin](https://github.com/brunanoroes/VeritaPlugin), que pleiteia os quatro
> selos. Por ser determinístico e rodar em CPU em ≈ 2 minutos, é o experimento de
> reprodução mais barata de todo o trabalho.

---

# Informações básicas

## Ambiente de execução

| Item | Especificação |
|---|---|
| Sistema operacional | Windows 11 (também validado em Ubuntu 22.04 LTS) |
| Python | 3.10.x |
| GPU | **Não necessária** |
| Rede | Necessária apenas para instalar as dependências |

## Requisitos de hardware

| Recurso | Mínimo |
|---|---|
| CPU | 2 núcleos x86-64 |
| RAM | 1 GB |
| Disco | 200 MB (dependências) + 1 MB (dataset) |
| GPU | Não necessária |
| Tempo de execução | ≈ 2 minutos |

## Metodologia

### Pipeline

```python
Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000,
                              ngram_range=(1, 2),
                              sublinear_tf=True)),
    ("svm",   LinearSVC(C=1.0, max_iter=2000, random_state=42))
])
```

| Parâmetro | Valor | Justificativa |
|---|---|---|
| `max_features` | `10000` | Limita o vocabulário, evitando esparsidade excessiva em um dataset de 450 instâncias |
| `ngram_range` | `(1, 2)` | Unigramas e bigramas — bigramas capturam expressões típicas de golpe ("dados bancários", "clique aqui") |
| `sublinear_tf` | `True` | Log-normalização da frequência de termos, prática padrão para SVM em texto |
| `C` | `1.0` | Regularização padrão do `LinearSVC` |
| `max_iter` | `2000` | Suficiente para convergência neste volume de dados |
| `random_state` | `42` | **A mesma seed usada no BERTimbau**, garantindo partições idênticas |

### Protocolo de avaliação

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` — **idêntico ao adotado no
repositório [treinamento-BERTimbau](https://github.com/brunanoroes/treinamento-BERTimbau)**.
Cada fold serve uma vez como conjunto de teste; as predições de todos os folds são
acumuladas, produzindo 450 avaliações (75 por categoria) para o relatório consolidado.

A métrica principal é o **F1-macro**, que atribui peso igual a cada categoria — adequado
porque o interesse do trabalho está no desempenho em todas as classes de golpe, e não
apenas nas mais frequentes.

### Limite inferior de referência

Um `DummyClassifier(strategy="most_frequent")` é avaliado sob o mesmo k-fold. Como o
dataset é perfeitamente balanceado (75 por categoria), sua F1-macro fica em **0,048** — o
valor analiticamente esperado para um classificador que sempre responde a mesma classe
entre seis: a classe predita obtém F1 = 2·(1/6)/(1+1/6) ≈ 0,286, as outras cinco obtêm 0,
e a média macro resulta em ≈ 0,048. Esse número contextualiza os demais resultados.

## Saída do script

O script imprime, em sequência:

1. Tamanho e balanceamento do dataset
2. F1-macro de cada um dos 5 folds, seguido da média ± desvio padrão
3. Relatório de classificação consolidado (precisão, recall e F1 por categoria) e acurácia geral
4. Bloco `COMPARATIVO FINAL`, com as três linhas de comparação e o ganho do BERTimbau

---

# Dependências

## Linguagem e runtime

| Dependência | Versão |
|---|---|
| Python | 3.10+ |
| pip | 22+ |

## Bibliotecas Python

| Biblioteca | Versão testada | Finalidade |
|---|---|---|
| `scikit-learn` | 1.5.x | `TfidfVectorizer`, `LinearSVC`, `StratifiedKFold`, `DummyClassifier` e métricas |
| `pandas` | 2.x | Leitura do dataset |
| `numpy` | 1.26.x | Média e desvio padrão dos folds |
| `openpyxl` | 3.1.x | Engine de leitura do arquivo `.xlsx` |

Nenhuma dependência de deep learning é necessária — este repositório **não usa PyTorch nem
transformers**.

## Recursos de terceiros

**Nenhum.** Este repositório não faz chamadas de rede em tempo de execução, não requer
chaves de API e não incorre em custo financeiro.

## Dataset

| Dataset | Instâncias | Colunas usadas | Origem |
|---|---|---|---|
| **BrScamsFacebook** | 450 (75 por categoria) | `Mensagem`, `Categoria` | Incluído como `BrScamsFacebook.xlsx`; também no [Kaggle](https://www.kaggle.com/datasets/brunaassisgt/brscamsfacebook) |

---

# Preocupações com segurança

A execução deste artefato **não oferece risco à máquina do avaliador**. O script apenas lê
uma planilha local e imprime resultados no terminal — não escreve arquivos, não acessa a
rede, não requer privilégios administrativos e não executa código de terceiros. É o
artefato de menor superfície de risco de todo o trabalho.

## 1. Conteúdo do dataset

`BrScamsFacebook.xlsx` contém **textos reais de golpes** coletados de publicações do
Facebook e de fóruns públicos, incluindo links maliciosos e números de telefone. **Não
acesse os links nem entre em contato com os números** presentes nos dados. O conteúdo é
fornecido exclusivamente como entrada para treinamento e classificação.

## 2. Dados pessoais

Os textos vêm de fontes públicas e passaram por revisão manual durante a rotulação, mas
podem conter identificadores residuais de terceiros. O dataset destina-se apenas a
pesquisa acadêmica e não deve ser usado para identificar ou contatar indivíduos.

## 3. Escopo do resultado

O baseline é um ponto de comparação metodológico, não um classificador destinado a uso
real. Com F1-macro em torno de 0,67, ele erra aproximadamente uma em cada três predições.

---

# Instalação

Tempo total: aproximadamente **2 minutos**.

## Passo 1 — Obter o repositório

```bash
git clone https://github.com/brunanoroes/Treinamento_TF-IDF-SVM.git
cd Treinamento_TF-IDF-SVM
```

## Passo 2 — Criar o ambiente virtual

Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## Passo 3 — Instalar as dependências

```bash
pip install scikit-learn pandas numpy openpyxl
```

> Tempo esperado: 1–2 minutos. Espaço em disco: ≈ 200 MB.

## Passo 4 — Verificar a instalação

```bash
python -c "import sklearn, pandas, numpy, openpyxl; print('sklearn', sklearn.__version__)"
```

Saída esperada (a versão pode variar):
```
sklearn 1.5.2
```

Ao final deste passo o ambiente está pronto para executar o experimento.

---

# Teste mínimo

Neste repositório, o teste mínimo **é a própria execução completa** — ela leva cerca de
dois minutos e não requer configuração adicional.

```bash
python treinamento.py
```

**Resultado esperado** — a saída no terminal segue esta estrutura:

```
Dataset: 450 instancias | 6 categorias
Balanceamento: {...75 por categoria...}

============================================================
VALIDACAO CRUZADA ESTRATIFICADA (k=5)
============================================================
  Fold 1..5: F1-macro por fold

  Media +/- DP : 0.711 +/- 0.027  <- resultado principal

============================================================
RELATORIO CONSOLIDADO (450 avaliacoes, 75 por categoria)
============================================================
                                      precision    recall  f1-score   support

Ataques de Phishing e Roubo de Dados       0.78      0.80      0.79        75
    Fraudes em Lojas Virtuais Falsas       0.70      0.75      0.72        75
   Golpes Baseados Em Relacionamento       0.73      0.92      0.82        75
      Golpes de Desinformacao Digital       0.62      0.64      0.63        75
  Golpes de Ganho Financeiro Ilusorio       0.84      0.76      0.80        75
                              Seguro       0.61      0.44      0.51        75

                            accuracy                           0.72       450
                           macro avg       0.71      0.72      0.71       450

Acuracia geral: 71.8%

============================================================
COMPARATIVO FINAL
============================================================
  Classificador por maioria  : F1-macro ~0.048
  TF-IDF + LinearSVC         : F1-macro  0.711 +/- 0.027
  BERTimbau fine-tuned       : F1-macro  0.763 +/- 0.034

  Ganho BERTimbau vs baseline: +0.052 pp de F1-macro
```

**Critérios de verificação:**

1. O dataset carrega com **450 instâncias e 6 categorias**, balanceado em 75 por categoria
2. Os 5 folds executam sem erro e a média é **0,711 ± 0,027**
3. A acurácia geral é **71,8%**
4. O bloco `COMPARATIVO FINAL` é impresso com as três linhas e o ganho do BERTimbau sobre
   o baseline é **+0,052**

> Os valores acima foram obtidos com `scikit-learn 1.5.x`. Por serem determinísticos, devem
> reproduzir-se dígito a dígito na mesma versão da biblioteca.

> Recursos: < 1 GB de RAM, sem escrita em disco, sem rede. Tempo: ≈ 2 minutos. Custo: zero.

## Solução de problemas

| Problema | Causa provável | Solução |
|---|---|---|
| `FileNotFoundError: BrScamsFacebook.xlsx` | Diretório errado | Execute a partir da raiz do repositório |
| `ImportError: openpyxl` | Engine de Excel ausente | `pip install openpyxl` |
| `KeyError: 'Mensagem'` ou `'Categoria'` | Planilha alterada | Restaure o arquivo original com `git checkout BrScamsFacebook.xlsx` |
| `ConvergenceWarning` do `LinearSVC` | Convergência lenta | Aviso, não erro; os resultados permanecem válidos. Para suprimi-lo, aumente `max_iter` |
| Valores diferentes dos esperados | Versão muito distinta de scikit-learn | Fixe a versão: `pip install "scikit-learn==1.5.2"` |

---

# Experimentos

Este repositório sustenta a **Reivindicação #2** do artefato principal.

**Tempo total estimado:** ≈ 2 minutos. **Recursos:** CPU apenas, < 1 GB de RAM. **Custo:**
zero.

## Reivindicações #2 — O BERTimbau supera o baseline TF-IDF + SVM sob protocolo idêntico de avaliação

**Arquivos de configuração:** nenhuma alteração é necessária. Todos os hiperparâmetros
estão fixados no corpo de `treinamento.py` (seção 2 do script) e reproduzem exatamente a
configuração reportada no artigo.

**Comando:**
```bash
python treinamento.py
```

**Recursos esperados:** CPU apenas, < 1 GB de RAM, sem GPU, sem rede, sem escrita em
disco. Tempo de execução: ≈ 2 minutos.

**Resultado esperado** — o bloco `COMPARATIVO FINAL`, ao fim da saída:

| Modelo | F1-macro | Observação |
|---|---|---|
| Classificador por maioria | **0,048** | Limite inferior de referência |
| TF-IDF + LinearSVC | **0,711 ± 0,027** | Baseline reproduzido por este script |
| **BERTimbau fine-tuned** | **0,763 ± 0,034** | Reproduzido em [treinamento-BERTimbau](https://github.com/brunanoroes/treinamento-BERTimbau) |

**O que a reivindicação afirma.** O ganho do BERTimbau sobre o baseline é de **+0,052 de
F1-macro (5,2 pontos percentuais)**, e é atribuível ao modelo, não à partição dos dados:
ambos os experimentos usam `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
sobre o mesmo dataset, produzindo **os mesmos folds**.

> **Leitura honesta da margem.** O ganho de 0,052 é da ordem de 1,5 desvio padrão do
> baseline (0,027) e cerca de 1,5 do próprio BERTimbau (0,034), de modo que os intervalos
> de ±1 DP dos dois modelos (0,684–0,738 e 0,729–0,797) **se sobrepõem marginalmente**. Com
> k = 5 e 450 instâncias, a evidência aponta consistentemente a favor do modelo neural, mas
> não sustenta a afirmação de superioridade categórica. O TF-IDF + SVM é um baseline forte
> nesta tarefa — resultado que merece registro, e não omissão.

**Análise complementar por categoria.** O relatório consolidado permite comparar o
desempenho categoria a categoria com a tabela equivalente do BERTimbau:

| Categoria | TF-IDF + SVM | BERTimbau | Diferença |
|---|---|---|---|
| Golpes Baseados em Relacionamento | 0,82 | 0,87 | +0,05 |
| Golpes de Ganho Financeiro Ilusório | 0,80 | 0,78 | −0,02 |
| Ataques de Phishing e Roubo de Dados | 0,79 | 0,78 | −0,01 |
| Fraudes em Lojas Virtuais Falsas | 0,72 | 0,71 | −0,01 |
| Golpes de Desinformação Digital | 0,63 | 0,78 | **+0,15** |
| Seguro | 0,51 | 0,64 | **+0,13** |

O ganho do modelo neural **não é uniforme**: ele se concentra em *Desinformação Digital* e
*Seguro*, justamente as duas categorias que dependem de compreensão contextual — distinguir
uma notícia falsa de uma verdadeira, ou conteúdo legítimo de golpe, exige interpretar o
sentido do texto, não apenas contar termos. Nas categorias marcadas por vocabulário
característico e recorrente (prêmios, dados bancários, ofertas), o TF-IDF alcança
desempenho equivalente ou ligeiramente superior. Ambos os modelos têm na categoria *Seguro*
seu pior resultado.

**Sobre o determinismo.** Diferentemente do fine-tuning do BERTimbau, este experimento é
**totalmente determinístico**: `TfidfVectorizer`, `LinearSVC` e `StratifiedKFold` operam
com semente fixa e sem paralelismo não determinístico. Repetidas execuções na mesma versão
de scikit-learn produzem **valores idênticos**, dígito a dígito. Variações só devem
ocorrer entre versões distintas da biblioteca.

---

# LICENSE

Este artefato é distribuído sob a **Licença MIT**. O texto completo está no arquivo
[LICENSE](LICENSE) deste repositório.

```
MIT License

Copyright (c) 2026 Bruna Norões

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions: [...]
```

O dataset **BrScamsFacebook** é disponibilizado para fins de pesquisa acadêmica.
