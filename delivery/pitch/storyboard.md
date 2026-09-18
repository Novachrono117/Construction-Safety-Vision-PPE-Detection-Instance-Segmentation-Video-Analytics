# Storyboard do vídeo-pitch

Cena a cena: quando, o que aparece na tela, o objetivo da fala, quanto dura e como
se passa para a próxima. O texto falado completo está em
[`full_script.md`](full_script.md).

**Duração total planejada: 6:30.** Três janelas apenas — navegador com o README,
uma aba do Colab e o player de vídeo — para reduzir troca de tela ao mínimo.

---

## Cena 1 — 00:00 – 00:30

**Tela:** README do repositório no GitHub, topo: título, subtítulo, badges e a
tabela de resumo (detector, segmentador, dados, protocolo de teste, demonstração).

**Objetivo da fala:** apresentar o problema de EPI em obra e dizer, em uma frase, o
que foi construído e qual é a pergunta científica.

**Duração falada:** ~30 s.

**Transição:** rolar suavemente até a seção *Data and split*.

---

## Cena 2 — 00:30 – 01:10

**Tela:** README, seção *Data and split*, com a tabela `train / validation / test`
visível.

**Objetivo da fala:** origem e licença do conjunto, tamanho da população, as cinco
classes, segmentação como anotação canônica e o split por grupo.

**Duração falada:** ~40 s.

**Transição:** rolar até *The two frozen models*.

---

## Cena 3 — 01:10 – 02:00

**Tela:** README, tabela comparativa D2 × S1 (arquitetura, `imgsz`, `mask_ratio`,
`overlap_mask`, base de seleção, hash do checkpoint).

**Objetivo da fala:** identificar os dois modelos congelados e explicar que a
diferença central entre eles é de **representação**, não de família de arquitetura.

**Duração falada:** ~50 s.

**Transição:** rolar até *Results on the held-out test set*.

---

## Cena 4 — 02:00 – 03:05

**Tela:** README, tabela de resultados do teste. Ao mencionar `vest_loose`,
**expandir** o bloco recolhido *Per-class AP@0.50:0.95 on the holdout*.

**Objetivo da fala:** dar os números finais arredondados, recusar explicitamente a
leitura de "vencedor", e separar IoU de máscara pareada de IoU normalizada.

**Duração falada:** ~65 s.

**Transição:** rolar até *Seeing it work* → figura caixa × máscara.

---

## Cena 5 — 03:05 – 04:05

**Tela:** primeiro a figura `box_vs_mask_hero_candidate.png` (ground truth · D2 ·
S1, lado a lado); depois rolar até a tabela de latência em *What the masks cost*.

**Objetivo da fala:** mostrar visualmente que a caixa inclui fundo e a máscara
isola o objeto; quantificar com a razão de preenchimento; então apresentar o custo
de latência e memória.

**Duração falada:** ~60 s.

**Transição:** alternar (Alt+Tab) para o player, já aberto e **pausado em 00:12**.

---

## Cena 6 — 04:05 – 05:20 · demonstração em vídeo real

**Tela:** player em tela cheia com `final_construction_ppe_compare.mp4`
(3840 × 1080, D2 à esquerda, S1 à direita).

**Roteiro de reprodução:**

| Momento | Ação |
| --- | --- |
| 04:05 | Vídeo **pausado em 00:12**. Falar a introdução (fonte, licença, 104 s, 2613 quadros, layout). |
| ~04:20 | **Play.** Deixar correr de **00:12 até 00:26** (≈14 s), narrando por cima. |
| ~04:34 | **Pausar em 00:26.** |
| 04:34 – 05:20 | Comentar o erro visível e a ressalva de throughput, com o quadro parado. |

**Objetivo da fala:** mostrar o sistema funcionando em imagem real, apontar
detecções de trabalhadores, sobreposição de rótulos em cena densa **e** um erro
visível, e deixar claro que 6 FPS é o pipeline de demonstração, não inferência.

**Duração falada:** ~75 s.

**Transição:** voltar ao navegador, seção *Error analysis*.

---

## Cena 7 — 05:20 – 06:00

**Tela:** README, tabela TP/FP/FN do teste e as duas matrizes de confusão; depois
rolar até *Limitations*.

**Objetivo da fala:** modo de falha dominante, o caso `vest_loose`, e as limitações
mais importantes ditas em voz alta.

**Duração falada:** ~40 s.

**Transição:** alternar para a aba do Colab.

---

## Cena 8 — 06:00 – 06:30

**Tela:** Colab aberto nos cabeçalhos **TREINO**, **AVALIAÇÃO** e **INFERÊNCIA**
(rolar rápido pelos três); depois voltar ao README, seção *What the comparison
actually shows*.

**Objetivo da fala:** reprodutibilidade em uma frase, o que o Colab faz e não faz,
e a conclusão científica como última frase.

**Duração falada:** ~30 s.

**Transição:** nenhuma. Parar a gravação.

---

## Escolha do trecho de vídeo

O clipe processado tem **104,52 s / 2613 quadros / 25 FPS**. Seis quadros foram
congelados **antes** da inferência e são os únicos instantes com observação
registrada; além deles, a saída completa foi revisada por contact sheet a cada
segundo (105 amostras). As janelas abaixo são recortes de apresentação ancorados
nesses instantes documentados — não são medições novas.

### Opção A — **00:12 → 00:26** (recomendada, ≈14 s)

Contém os instantes documentados de **17,44 s e 17,92 s**.

O que o registro descreve nesses dois quadros: o S1 desenha uma caixa e uma máscara
de `person` sobre o material embalado no palete atrás do pilar central, **nos dois
quadros**, onde nenhuma pessoa é visível; o D2 não desenha nada equivalente ali.
Ambos os sistemas marcam `person` e `vest_on_body` em vários trabalhadores reais, e
os rótulos se sobrepõem na região do pilar.

**Por que esta é a recomendada:** entrega os quatro requisitos de uma só vez —
detecções de trabalhadores reais, máscaras, comportamento em cena densa e um erro
visível — e o erro é **persistente** nos quadros revisados, então uma janela de 14 s
o contém de forma confiável durante uma apresentação ao vivo.

### Opção B — **01:22 → 01:34** (alternativa, ≈12 s)

Contém os instantes documentados de **87,12 s e 87,60 s**: trabalhadores cruzando o
primeiro plano, vários rótulos de `person` e `vest_on_body` em ambos os sistemas, e
um capacete visível num trabalhador curvado que o S1 **não** marca em 87,12 s e
marca em 87,60 s, enquanto o D2 não marca em nenhum dos dois.

**Quando preferir:** se a intenção for mostrar um **falso negativo** e instabilidade
temporal em vez de um falso positivo. É menos legível ao vivo, porque uma ausência
de caixa é mais difícil de enxergar em poucos segundos do que uma máscara espúria.

### Por que o terço do meio foi descartado

Em 52,24 s o S1 mostra uma caixa e máscara grandes de `helmet_loose` sobre os
banheiros químicos e o carregador; em 52,72 s essa sobreposição **já não está lá**.
O registro estabelece o aparecimento e o desaparecimento, não o início, o fim ou a
duração do erro — então um recorte curto pode ou não contê-lo. É um exemplo
excelente em imagem parada e uma aposta ruim numa reprodução ao vivo.

### Regras que valem para qualquer opção

- Não editar, filtrar ou re-renderizar predições. O MP4 exibido é a saída gravada.
- Não montar um compilado só com acertos.
- O erro visível **precisa** ser narrado, não ignorado.
- Se preferir mostrar o erro do terço do meio, use o **quadro parado**
  `reports/figures/final_video/frame_001306.jpg`, e diga que é um quadro extraído.
