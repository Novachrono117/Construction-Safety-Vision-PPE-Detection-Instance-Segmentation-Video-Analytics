# Roteiro completo do vídeo-pitch

**Duração alvo: 6:30** (faixa segura 5:30–7:15; o limite do enunciado é 5–8 min).
Idioma falado: português do Brasil. Apresentador único: **Vinicius Pereira Gomes**.

Todo número falado aqui vem de um artefato commitado e está registrado em
[`pitch_manifest.json`](pitch_manifest.json), que um teste reconfere campo a campo.
Os valores falados são arredondados de propósito; os exatos ficam na tela.

Marcações entre colchetes são instruções de tela, **não são faladas**.

---

## 0:00 – 0:30 · Abertura e problema

*[Tela: topo do README no GitHub — título, subtítulo e tabela de resumo.]*

> Em obras, boa parte dos acidentes graves envolve equipamento de proteção ausente
> ou usado de forma incorreta, e a supervisão manual não escala.
>
> O problema é localizar cada pessoa e cada EPI num quadro de obra e separar o
> equipamento em uso daquele que só está na cena: um capacete na bancada e um
> capacete na cabeça são objetos parecidos com significados opostos.
>
> Eu treinei e congelei dois modelos sobre os mesmos dados — um detector de caixas e
> um segmentador de instâncias — para perguntar o que a máscara informa além da
> caixa, e quanto isso custa.

*[Transição: rolar até a seção de dados e split.]*

---

## 0:30 – 1:10 · Dados e desenho experimental

*[Tela: README, seção "Data and split", com a tabela do split.]*

> Os dados vêm do Roboflow Universe, o conjunto *Construction PPE Compliance
> Detection*, sob CC BY 4.0. Depois da auditoria ficaram 436 imagens-fonte, 433 na
> modelagem, com 2.031 anotações em cinco classes: person, helmet_on_head,
> helmet_loose, vest_on_body e vest_loose.
>
> A anotação canônica é a segmentação: as caixas são derivadas dos polígonos, então
> as duas tarefas descrevem exatamente os mesmos objetos.
>
> O split é 303 de treino, 65 de validação e 65 de teste, e é por grupo: imagens
> duplicadas ou da mesma cena ficam sempre do mesmo lado, para não vazar informação.
> O teste foi congelado antes de qualquer treino.

*[Transição: rolar até "The two frozen models".]*

---

## 1:10 – 2:00 · Os dois modelos congelados

*[Tela: README, tabela D2 × S1.]*

> O detector final é o D2: YOLO11n, entrada 768. O segmentador final é o S1:
> YOLO11n-seg, também em 768, com mask_ratio 4 e overlap_mask desligado.
>
> Os dois foram escolhidos usando só a validação, por um critério declarado antes
> das execuções, e congelados por hash antes de qualquer contato com o teste.
>
> A diferença prática é de representação: o D2 entrega classe, confiança e um
> retângulo; o S1 entrega isso e mais a máscara do objeto, ou seja, quais pixels
> pertencem àquela instância. É essa informação extra que eu queria medir, junto
> com o preço dela.

*[Transição: rolar até "Results on the held-out test set".]*

---

## 2:00 – 3:05 · Resultados finais no teste

*[Tela: README, tabela de resultados; expandir o bloco por classe ao falar de vest_loose.]*

> Este é o resultado no conjunto de teste, lido uma única vez, com os dois modelos
> já congelados: 65 imagens e 305 instâncias.
>
> Nas caixas, o D2 chega a 42,7% de mAP de 0,50 a 0,95, e o S1, a 43,4%. Nas
> máscaras, o S1 fica em 41,0%. As precisões ficam em torno de 78% e as revocações
> entre 62 e 66%, no ponto de operação fixado.
>
> Eu não vou dizer que o S1 venceu. A diferença agregada nas caixas é de sete
> milésimos, e não é uniforme: três das cinco classes pioraram. Não houve teste de
> significância predeclarado, e cada configuração rodou uma vez só. A leitura
> honesta é que os dois têm desempenho de localização amplamente semelhante.
>
> Um diagnóstico separado mede a máscara diretamente: quando o S1 encontra o objeto,
> a IoU média é 0,83; normalizando por todas as instâncias cai para 0,59, porque
> cerca de 30% dos objetos não são encontrados.

*[Transição: rolar até a figura "What a mask adds over a box".]*

---

## 3:05 – 4:05 · O que a máscara acrescenta, e quanto custa

*[Tela: figura caixa × máscara (ground truth, D2, S1); depois a tabela de latência.]*

> Aqui está onde a máscara paga. A caixa inclui fundo; a máscara isola a geometria
> do objeto.
>
> Medindo nas próprias predições do segmentador, a mediana da razão entre a área da
> máscara e a área da sua própria caixa é 0,66 — ou seja, cerca de um terço do
> retângulo mediano não é o objeto. Isso habilita área, forma, centroide e
> sobreposição espacial, que um retângulo não expressa.
>
> *[Tela: tabela "What the masks cost".]*
>
> E isso tem preço. Num benchmark controlado nesta máquina, uma RTX 5070 Laptop, em
> FP32, batch 1 e entrada 768, a latência média de saída completa vai de cerca de
> 9,2 ms para 11,9 ms: aproximadamente 30% a mais. A memória de inferência
> sobe cerca de 2,4 vezes no pico reservado, embora em valor absoluto ainda seja
> baixa.

*[Transição: alternar para o player com o vídeo de saída já aberto e pausado em 00:12.]*

---

## 4:05 – 5:20 · Demonstração em vídeo real

*[Tela: player em tela cheia. Falar as duas primeiras frases com o vídeo ainda pausado.]*

> Agora o sistema rodando em vídeo real. É um clipe de obra do Wikimedia Commons, de
> Frank Vincentz, sob CC BY-SA 3.0. Os dois modelos congelados processaram o clipe
> inteiro: 104 segundos, 2.613 quadros. À esquerda o D2, com caixas; à direita o S1,
> com máscaras.

*[**Dar play em 00:12 e deixar rodar até 00:26** — cerca de 14 segundos. Narrar por cima.]*

> Repare nos trabalhadores no centro: os dois sistemas marcam pessoa e colete, e nas
> regiões densas os rótulos se sobrepõem.
>
> E repare também no erro: ali no palete, atrás do pilar, o S1 desenha uma pessoa,
> com máscara, sobre material embalado, onde não há ninguém. O D2 não desenha nada
> ali. É um falso positivo de fundo que persiste nos quadros revisados, e está
> registrado no relatório do projeto. Eu não editei predição nenhuma nem escolhi um
> trecho sem falha.

*[Pausar o vídeo em 00:26.]*

> Uma ressalva: processar esse vídeo rodou a cerca de 6 quadros por segundo, contra
> 25 da fonte. Isso é o pipeline inteiro de demonstração — decodificação, os dois
> modelos, desenho e codificação — não a velocidade de inferência. Nada aqui é
> tempo real.

*[Transição: voltar ao navegador, seção "Error analysis".]*

---

## 5:20 – 6:00 · Erros e limitações

*[Tela: README, tabela TP/FP/FN e as duas matrizes de confusão do teste.]*

> No teste, o modo de falha dominante é objeto não detectado, não confusão de
> classe: o D2 tem 189 acertos, 51 falsos positivos e 116 falsos negativos; o S1,
> 205, 56 e 100.
>
> A limitação mais séria é a classe vest_loose: só 7 instâncias, em 2 imagens, e o
> D2 não recuperou nenhuma. O número é publicado como está e não decide nada.
>
> *[Tela: rolar até "Limitations".]*
>
> Some-se a isso um conjunto pequeno, queda da validação para o teste, ausência de
> rastreamento, e o teste já gasto: foi lido uma vez e não pode mais ajustar nada.

*[Transição: alternar para a aba do Colab.]*

---

## 6:00 – 6:30 · Reprodutibilidade, Colab e conclusão

*[Tela: Colab, mostrando os cabeçalhos TREINO, AVALIAÇÃO e INFERÊNCIA; depois voltar ao README.]*

> Tudo está num repositório público, com relatório técnico e um Colab executável,
> com seções de treino, avaliação e inferência. Treino e avaliação mostram a receita
> e os resultados já medidos: não retreinam nem reabrem o teste. A inferência é
> opcional, com os pesos congelados verificados por hash.
>
> *[Tela: README, seção "What the comparison actually shows".]*
>
> A conclusão: caixas e máscaras alcançaram evidência de localização amplamente
> semelhante; o que a segmentação acrescentou foi representação espacial mais rica,
> e essa informação tem custo computacional. A representação correta depende do que
> a aplicação precisa medir.

*[Fim. Parar a gravação.]*

---

## Observações para o apresentador

- **Nunca** dizer "o S1 venceu", "a segmentação foi melhor" ou "o D2 perdeu".
  A formulação aprovada é *desempenho de localização amplamente semelhante*.
- Falar os números arredondados; os exatos aparecem na tela.
- O trecho de vídeo é de **00:12 a 00:26** da saída comparativa. A alternativa
  documentada está em [`storyboard.md`](storyboard.md).
- Se o tempo apertar, os cortes seguros são, nesta ordem: a frase sobre memória de
  inferência, a frase sobre matrizes de confusão e a frase sobre o Colab não
  retreinar. **Nunca** cortar o erro visível no vídeo nem a limitação de
  `vest_loose`.
