# Perguntas prováveis e respostas defensáveis

Respostas curtas, apoiadas em artefatos commitados. Quando a evidência não existe, a
resposta correta é dizer **"não foi medido"** — isso não é uma falha da apresentação,
é o padrão do projeto.

---

### 1. Por que YOLO11n, e não um modelo maior?

O YOLO11n foi a **linha de base** declarada antes de qualquer execução. Depois disso
rodei duas variações controladas, cada uma mudando **um** campo: capacidade
(YOLO11s) e resolução de entrada (768).

A variação de capacidade ficou **abaixo** da linha de base na métrica de seleção
(0,560017 contra 0,570142); a variação de resolução ficou **acima**, por 0,023876,
além da margem de 0,005 fixada antes. Por isso o modelo final é nano a 768.

Isso diz que, **nesta configuração e com uma execução por variação**, mais capacidade
não ajudou e mais resolução ajudou. Não é uma afirmação de que o nano é ótimo em
geral. O modelo maior nunca foi retreinado nem ajustado para tentar recuperar.

### 2. Por que treinar detecção *e* segmentação, em vez de só uma?

Duas razões. A primeira é o enunciado, que pede as duas tarefas. A segunda é a
pergunta do trabalho: a anotação canônica do conjunto **é** a segmentação, e as
caixas são derivadas matematicamente dos polígonos. Isso significa que as duas
tarefas descrevem exatamente os mesmos objetos, nas mesmas imagens, nos mesmos
conjuntos — e é justamente essa igualdade que permite perguntar o que a máscara
acrescenta, sem confundir o efeito da representação com o efeito dos dados.

### 3. Se o S1 também detecta, por que ele não substituiu o D2?

Porque a evidência não mostra superioridade. No teste, a diferença de AP de caixas é
de **+0,006733** a favor do S1, mas **três das cinco classes pioraram**; na validação,
retirando a classe de suporte mínimo, a diferença é **−0,008040**, ou seja, o
segmentador fica **abaixo**. Não houve teste de significância predeclarado e cada
configuração rodou uma vez.

Além disso, o S1 custa cerca de **30% mais** de latência média de saída completa e
cerca de **2,4×** de pico reservado de memória de inferência.

A recomendação registrada é **condicional ao uso**: o detector quando classe,
confiança e caixa bastam, ao menor custo; o segmentador quando a aplicação precisa de
suporte de primeiro plano, geometria não retangular, área, centroide, preenchimento
ou sobreposição espacial. Nenhum dos dois é declarado vencedor.

### 4. Por que o conjunto de teste foi acessado uma única vez?

Porque qualquer decisão tomada olhando o teste destrói o que ele mede. Assim que se
escolhe um modelo, um limiar ou uma limpeza de dados com base nele, ele deixa de ser
uma estimativa de generalização e vira parte do treino.

O protocolo completo da avaliação final — quais checkpoints, quais confianças, quais
avaliadores, como contar TP/FP/FN, como ordenar exemplos qualitativos — foi congelado
num arquivo de configuração com impressão digital **antes de ler um byte** do teste.
O acesso exigia duas autorizações independentes, em código e no ambiente. Foi uma
tentativa, com três passagens de inferência predeclaradas, e as predições foram
persistidas e hasheadas **antes** de qualquer métrica ser calculada.

Hoje o teste está observado e bloqueado: nenhum número dele pode motivar mudança de
modelo, de limiar, de agrupamento de classes ou do conjunto de dados.

### 5. Por que a classe `vest_loose` é tão fraca?

Principalmente por suporte. Na população inteira são 45 instâncias em 8 imagens; no
split congelado, 5 imagens de treino, **1** de validação e **2** de teste. No teste
são **7 instâncias em 2 imagens** — o D2 não recuperou nenhuma e o S1 marcou AP de
máscara 0,003850.

Por que ela falha, em termos visuais ou de aprendizado, é **desconhecido**: nenhum
experimento isolou a causa. O que o projeto fez foi declarar, **antes** dos
resultados, uma regra de suporte (≥ 5 imagens positivas **e** ≥ 20 instâncias) que
exclui essa classe da métrica de decisão — sem nunca escondê-la do relatório. Ela é
reportada por inteiro e não decide nada.

### 6. Por que a inferência no vídeo não é em tempo real?

Os ~6 quadros por segundo da demonstração medem o **pipeline inteiro**: decodificação
de um intermediário sem perdas, hashing dos pixels, os **dois** modelos rodando em
sequência em cada quadro, o desenho das sobreposições e a codificação de uma saída de
3840 × 1080. Não é latência de inferência.

A latência de inferência medida em benchmark controlado é de cerca de 9,2 ms para o
D2 e 11,9 ms para o S1, por imagem, em batch 1.

E nenhuma engenharia de tempo real foi feita: sem batching, sem meia precisão, sem
TensorRT, sem pular quadros, sem rodar um modelo só. Tempo real não é reivindicado em
lugar nenhum do projeto.

### 7. Na prática, o que a segmentação de instâncias acrescentou?

Acrescentou **medidas que o retângulo não expressa**: suporte de primeiro plano,
contorno não retangular, área, centroide, razão de preenchimento e sobreposição
espacialmente específica. A mediana da razão entre a área da máscara prevista e a
área da sua própria caixa é **0,664433** — cerca de um terço do retângulo mediano não
é o objeto. E onde existe um substituto retangular, ele é sistematicamente inflado:
área média de máscara 144.563 px contra 237.206 px de proxy de caixa; em **11 de 349**
pares candidatos, as caixas se sobrepunham e as máscaras não compartilhavam um único
pixel.

O que ela **não** acrescentou, na regra testada: associação pessoa-EPI. Fixando o
modelo e a regra de contenção em 0,50, houve 103 concordâncias, 3 casos só-caixa,
**zero** casos só-máscara e 66 sem associação, em 173 relações. Isso vale para essa
regra e essa população; não generaliza.

### 8. Por que o Colab não reexecuta todo o experimento por padrão?

Por três motivos, nessa ordem de importância.

Primeiro, cada modelo foi treinado **uma vez** sob um protocolo congelado antes da
execução. Retreinar produziria bytes diferentes com o mesmo nome e invalidaria os
resultados publicados, que são identificados por hash.

Segundo, o teste final está gasto. Não existe caminho executável de avaliação final
no notebook — de propósito.

Terceiro, na prática: treinar exige GPU CUDA e o conjunto materializado, que o
notebook não distribui.

O que a seção de treino faz é mostrar a receita registrada por inteiro — arquitetura,
resolução, épocas, batch, seed, o otimizador que o `auto` resolveu, todos os
argumentos de aumento, a regra de seleção de checkpoint, a época escolhida e a
identidade do checkpoint — com as curvas gravadas na época, e imprime o comando
reprodutível. A seção de avaliação lê os resultados commitados campo a campo. A
inferência é opcional, com pesos verificados por SHA-256.

### 9. O que você melhoraria com mais tempo?

Em ordem de retorno esperado:

1. **Mais dados, sobretudo de `vest_loose`.** É a limitação que mais restringe as
   conclusões atuais.
2. **Repetições das mesmas configurações.** A variância entre execuções é hoje
   **desconhecida**, porque nada foi repetido. Com repetições, diferenças como os
   0,006733 entre D2 e S1 poderiam ser testadas em vez de comparadas a uma margem de
   engenharia.
3. **Reprodução a partir de um clone limpo**, que ainda não foi demonstrada.
4. **Rastreamento** (o bônus), para consistência temporal — o erro de fundo visível no
   vídeo é exatamente o tipo de coisa que um rastreador ajudaria a caracterizar.
5. **Publicação dos checkpoints**, após concluir a revisão de licenciamento.
6. **Associação pessoa-EPI em mais de um limiar de contenção**, já que o resultado
   atual vale só para 0,50.

Tudo isso é plano, não resultado.

---

## Perguntas extras plausíveis

### O que é mAP@0,50:0,95 e por que usar os dois?

`mAP@0,50` resume a curva precisão-revocação exigindo 50% de sobreposição com o
objeto real. `mAP@0,50:0,95` faz a média exigindo sobreposições cada vez maiores, de
0,50 a 0,95, então é bem mais rigoroso com a localização. Reportar os dois separa
"achou o objeto" de "achou e delimitou bem".

### Por que o desempenho no teste é menor que na validação?

Todos os APs canônicos caíram: D2 caixas de 0,485390 para 0,427031, S1 caixas de
0,505682 para 0,433764, S1 máscaras de 0,455034 para 0,410143. A **causa é
desconhecida** e não pode mais ser investigada empiricamente sem reabrir o teste, o
que o protocolo proíbe. É registrado como comparação descritiva de generalização, sem
teste de significância.

### O que é `overlap_mask`, e por que desligá-lo?

É como o framework monta o alvo de máscara quando duas instâncias dividem pixels. Com
ele ligado, a instância **menor** fica com os pixels disputados — o colete fica com os
pixels da pessoa que o veste. Um diagnóstico pós-hoc indicou que isso prejudicava a
máscara de `person`, então desligá-lo virou a **única** variável de um experimento
controlado, congelado antes de rodar. A métrica agregada melhorou 0,074820, com
`person` respondendo pela maior parte e `helmet_loose` **piorando**. A consistência com
a hipótese não a prova.

### Você garante que não houve vazamento entre os conjuntos?

Não. O que posso afirmar é o que foi feito: duas impressões digitais perceptuais mais
revisão humana de **todos** os candidatos levantados, com 11 grupos confirmados
mantidos indivisíveis, de modo que duplicatas e cenas repetidas nunca cruzam a
fronteira do split. Nada estabelece que duas imagens em conjuntos diferentes não
compartilhem local, dia, câmera ou trabalhador. Por isso o split é descrito como
"com restrição por grupo e por classe", e nunca como perfeitamente estratificado.
