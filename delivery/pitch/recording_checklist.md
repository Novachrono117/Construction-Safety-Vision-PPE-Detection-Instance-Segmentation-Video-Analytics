# Checklist de gravação e publicação

Do preparo da máquina até o link entregue. Nada aqui exige software pago: o gravador
embutido do sistema, o OBS Studio ou o próprio Google Meet gravando a tela resolvem.

---

## A. Antes de gravar — máquina e tela

- [ ] Resolução de gravação **1920 × 1080 no mínimo**, a 30 FPS.
- [ ] **Notificações desativadas** (Windows: Assistente de Foco / Não Perturbe).
- [ ] Fechar e-mail, mensageiros, calendário e qualquer aba pessoal.
- [ ] **Janela anônima ou perfil limpo** do navegador: sem favoritos pessoais, sem
      histórico na barra de endereço, sem contas logadas visíveis.
- [ ] Zoom do navegador em **125–150%** — o texto do README precisa ser legível em
      tela cheia quando o vídeo for reduzido no player do avaliador.
- [ ] Ocultar a barra de favoritos (`Ctrl+Shift+B`).
- [ ] Cursor grande / destaque de clique, se o gravador oferecer. Opcional, ajuda.
- [ ] **Terminal fechado ou limpo.** Nenhum caminho absoluto local deve aparecer.
- [ ] Nenhum arquivo de checkpoint, `.env`, chave ou token visível em qualquer tela.
- [ ] Explorador de arquivos fechado — ele expõe a estrutura de pastas local.

## B. Antes de gravar — conteúdo aberto

Abrir **nesta ordem**, para que a apresentação seja quase só rolagem:

1. **Aba 1 — README** do repositório público no GitHub, no topo da página.
2. **Aba 2 — Colab**, o notebook já aberto, rolado até o cabeçalho **TREINO**.
3. **Player** com `final_construction_ppe_compare.mp4` aberto, em tela cheia,
   **pausado em 00:12**, com o som mudo (o vídeo não tem áudio).

- [ ] Confirmar que o repositório está público e o README renderiza (ele é a
      superfície principal da apresentação).
- [ ] Confirmar que o vídeo abre e que a busca para 00:12 funciona no player
      escolhido. Um MP4 de 3840 × 1080 pode engasgar em players leves — testar antes.
- [ ] Deixar [`presenter_cues.md`](presenter_cues.md) num segundo monitor, no celular
      ou impresso. **Não** deixá-lo visível na tela gravada.

## C. Áudio

- [ ] Testar o microfone com **30 segundos de gravação** e ouvir o resultado antes.
- [ ] Ambiente silencioso; sem ventilador nem ar-condicionado próximos ao microfone.
- [ ] Falar a uma distância constante. Fone com microfone costuma bastar.
- [ ] Ritmo alvo: **~140 palavras por minuto**. O roteiro tem 917 palavras faladas,
      o que dá cerca de **6:33** nesse ritmo.

## D. Ensaio

- [ ] Ler o roteiro em voz alta **uma vez, com cronômetro**, antes de gravar.
- [ ] Se passar de **7:15**, cortar na ordem indicada no fim de
      [`presenter_cues.md`](presenter_cues.md).
- [ ] Se ficar abaixo de **5:30**, falar mais devagar na seção de resultados e no
      vídeo — não inventar conteúdo novo.
- [ ] Ensaiar especificamente a transição para o player e o play em 00:12: é o único
      momento com risco de atrapalhar o tempo.

## E. Durante a gravação

- [ ] Gravar **em uma tomada só**, se possível. Um tropeço pequeno é aceitável num
      pitch técnico; recomeçar dez vezes custa mais.
- [ ] Nunca dizer "o S1 venceu", "a segmentação foi melhor" ou "o D2 perdeu".
- [ ] Falar o erro visível no vídeo. Não pular.
- [ ] Falar a limitação de `vest_loose`. Não pular.
- [ ] Encerrar com a frase de conclusão científica, não com agradecimento.

## F. Conferência antes de publicar

- [ ] **Duração entre 5:00 e 8:00** — requisito do enunciado. Alvo 6:30.
- [ ] Áudio audível do início ao fim, sem corte.
- [ ] O vídeo de inferência real **aparece e é reconhecível** (é requisito explícito).
- [ ] Nenhum caminho local, token, e-mail pessoal ou aba privada visível em nenhum
      quadro. Passar o vídeo rapidamente conferindo isso.
- [ ] Nenhum número falado que não esteja em [`pitch_manifest.json`](pitch_manifest.json).
- [ ] O apresentador aparece ou é identificado por voz — o enunciado pede participação
      de todos os integrantes (aqui, um único integrante documentado).

## G. Publicação

O enunciado aceita **YouTube não listado** ou **Google Drive**.

**Recomendação: YouTube não listado.** O acesso costuma ser mais simples para o
avaliador — não depende de conta Google autorizada, não exibe pedido de permissão e
reproduz sem download. O Drive é uma alternativa legítima, mas exige acertar o
compartilhamento como "qualquer pessoa com o link".

- [ ] Fazer o upload como **Não listado** (não "Privado" — privado exige convite).
- [ ] Título sugerido: `Construction Safety Vision — PPE Detection & Instance Segmentation`.
- [ ] Na descrição, incluir o link do repositório e o crédito do vídeo-fonte:
      *Frank Vincentz, Wikimedia Commons, CC BY-SA 3.0.*
- [ ] **Verificar em janela anônima**, deslogado, que o link abre e reproduz. Esse
      teste é obrigatório: um vídeo que só abre na conta do autor não foi entregue.
- [ ] Guardar o link e entregá-lo com o restante do material.

> Nenhum upload é feito automaticamente por este repositório. A publicação é uma ação
> humana, e o pitch só passa a contar como entregue depois que o link acessível
> existir.

## H. Depois de publicar

- [ ] Registrar o link no local combinado da entrega acadêmica.
- [ ] Atualizar o rastreador de entrega apenas quando o link existir — o estado
      `VIDEO_PITCH_READY_TO_RECORD` só muda depois da gravação publicada, e essa
      transição é uma etapa humana separada.
