# Pacote do vídeo-pitch

Tudo o que é preciso para gravar o pitch acadêmico de 5 a 8 minutos.
**Estado: `VIDEO_PITCH_READY_TO_RECORD`.** Nada aqui é a entrega final: o pitch só
está completo quando existir uma gravação humana publicada e acessível.

| Arquivo | Para quê |
| --- | --- |
| [`full_script.md`](full_script.md) | Roteiro completo em português, com marcações de tela |
| [`presenter_cues.md`](presenter_cues.md) | Só os gatilhos, para apresentar sem ler |
| [`storyboard.md`](storyboard.md) | Cena a cena: tela, objetivo, duração, transição; e a escolha do trecho de vídeo |
| [`qa.md`](qa.md) | Perguntas prováveis do avaliador, com respostas apoiadas em artefatos |
| [`recording_checklist.md`](recording_checklist.md) | Preparo da máquina, gravação, conferência e publicação |
| [`pitch_manifest.json`](pitch_manifest.json) | Cada número falado, com o artefato e o campo de onde ele vem |

---

## O que o enunciado exige

Vídeo-pitch de **5 a 8 minutos**, em **YouTube não listado** ou **Google Drive**, com
participação de todos os integrantes, demonstrando **o sistema funcionando, incluindo
o vídeo com inferência**. Peso: **5%** da nota.

Este pacote cobre clareza técnica, domínio do projeto, sistema em funcionamento e o
vídeo de inferência real. A participação é coberta por um apresentador único, porque
**um único integrante está documentado no repositório**: Vinicius Pereira Gomes, autor
registrado no [relatório técnico](../../academic/final_report.md). Autoria de commits
não foi usada para inferir outros integrantes.

## Como usar, na ordem

1. Ler [`storyboard.md`](storyboard.md) uma vez, para entender o fluxo de telas.
2. Ler [`full_script.md`](full_script.md) em voz alta, **com cronômetro**.
3. Preparar a máquina com [`recording_checklist.md`](recording_checklist.md), seções A a C.
4. Gravar apoiado em [`presenter_cues.md`](presenter_cues.md), não no roteiro completo.
5. Conferir e publicar com as seções F a H do checklist.
6. Reler [`qa.md`](qa.md) antes da apresentação ao vivo, se houver arguição.

## Duração

Alvo **6:30**. O roteiro tem **917 palavras faladas**, o que dá cerca de 6:33 a
140 palavras por minuto — ritmo normal de fala técnica em português. A faixa segura
de ensaio é **5:30–7:15**, dentro do limite de 5 a 8 minutos do enunciado.

Se o ensaio passar de 7:15, os cortes seguros estão no fim de
[`presenter_cues.md`](presenter_cues.md).

## Regras que o pitch não pode quebrar

Estas não são preferências de estilo; elas mantêm a apresentação alinhada com o que a
evidência sustenta.

- **Nenhum vencedor.** Nunca dizer "o S1 venceu", "a segmentação foi melhor" ou "o D2
  perdeu". A formulação correta é *desempenho de localização amplamente semelhante*.
  Três das cinco classes pioraram na comparação final de caixas.
- **Nenhum tempo real.** Os ~6 quadros por segundo da demonstração são o pipeline
  inteiro, não a latência de inferência.
- **Nenhuma falha escondida.** O trecho de vídeo escolhido contém um erro visível, e
  ele precisa ser narrado. Nada de compilado só com acertos, nada de editar predições.
- **`vest_loose` aparece.** 7 instâncias em 2 imagens no teste, e o detector não
  recuperou nenhuma. Isso é dito em voz alta.
- **Nenhum número inventado.** Todo valor falado está em
  [`pitch_manifest.json`](pitch_manifest.json), ligado ao artefato e ao campo de
  origem, e conferido por teste automático.

## Materiais que a gravação usa

Já existentes no repositório, nada precisa ser gerado:

- o [README público](../../README.md) — superfície principal da apresentação;
- a figura [caixa × máscara](../../reports/figures/qualitative/box_vs_mask_hero_candidate.png);
- as matrizes de confusão do teste, em `reports/figures/final_test/`;
- o [notebook do Colab](../../notebooks/construction_safety_vision_demo.ipynb);
- o MP4 comparativo D2/S1, que é um artefato **local** — ele não está no repositório,
  e sua distribuição pública continua pendente. Ver [`../VIDEO.md`](../VIDEO.md) e o
  [relatório da demonstração](../../reports/final_real_video_demo.md).

## O que este pacote não faz

- Não grava, não edita e não publica vídeo nenhum.
- Não executa modelo, não treina, não recalcula métrica e não toca no conjunto de
  teste, que está gasto e bloqueado.
- Não marca o pitch como entregue. A transição para concluído depende do link
  publicado e é uma etapa humana separada.
