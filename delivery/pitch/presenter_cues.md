# Cue sheet do apresentador

Versão curta, só gatilhos, para apresentar sem ler. O texto completo está em
[`full_script.md`](full_script.md); as telas, em [`storyboard.md`](storyboard.md).

**Regra de ouro:** nunca dizer *"o S1 venceu"*, *"a segmentação foi melhor"* ou
*"o D2 perdeu"*. A formulação correta é **"desempenho de localização amplamente
semelhante"**.

---

### 00:00 · README topo — problema
- Acidentes graves ↔ EPI ausente ou mal usado · supervisão manual não escala
- Localizar pessoa + EPI · separar **em uso** de **solto** (capacete na bancada ≠ na cabeça)
- Dois modelos congelados, mesmos dados: detector + segmentador
- Pergunta: **o que a máscara informa além da caixa, e quanto custa**

### 00:30 · README dados/split
- Roboflow Universe · *Construction PPE Compliance Detection* · CC BY 4.0
- **436** fonte → **433** modelagem · **2031** anotações · **5** classes
- Segmentação é canônica → caixas **derivadas** dos polígonos → mesmos objetos
- Split **303 / 65 / 65** · **por grupo** (duplicatas nunca se separam)
- Teste congelado **antes** do treino

### 01:10 · README modelos
- **D2** = YOLO11n @ 768 · **S1** = YOLO11n-seg @ 768, `mask_ratio 4`, `overlap_mask` off
- Escolhidos **só na validação**, critério declarado antes · congelados por hash
- Diferença = **representação**: caixa vs. quais pixels são o objeto

### 02:00 · README resultados *(expandir o bloco por classe)*
- Teste: **65 imagens / 305 instâncias**, lido **uma vez**
- Caixas: D2 **42,7%** · S1 **43,4%** — máscaras: S1 **41,0%** (mAP 0,50–0,95)
- P ≈ **78%** · R **62–66%** no ponto de operação fixado
- ⚠️ **Sem vencedor**: delta de 7 milésimos, **3 das 5 classes pioraram**, sem teste
  de significância, uma execução por configuração
- IoU de máscara pareada **0,83** → normalizada por todas as instâncias **0,59**
  (≈30% dos objetos não encontrados)

### 03:05 · README figura caixa × máscara → tabela de custo
- Caixa inclui fundo · máscara isola geometria
- Mediana máscara/caixa = **0,66** → ~⅓ do retângulo mediano não é objeto
- Habilita: **área, forma, centroide, sobreposição**
- Custo (RTX 5070 Laptop, FP32, batch 1, 768): **9,2 → 11,9 ms** ≈ **+30%**
- Memória de inferência ≈ **2,4×** no pico reservado (absoluto ainda baixo)

### 04:05 · VÍDEO — pausado em **00:12**
- Wikimedia Commons · **Frank Vincentz** · **CC BY-SA 3.0**
- Clipe inteiro processado: **104 s / 2613 quadros** · D2 esquerda, S1 direita
- ▶️ **PLAY 00:12 → 00:26** (≈14 s), narrar por cima:
  - trabalhadores no centro: pessoa + colete nos dois · rótulos se sobrepõem
  - ❗ **erro**: S1 desenha pessoa + máscara sobre material no palete atrás do pilar
    — não há ninguém ali · D2 não desenha · **falso positivo persistente**
  - "não editei predição nenhuma nem escolhi um trecho sem falha"
- ⏸️ **PAUSE em 00:26**
- Ressalva: **~6 FPS vs 25 FPS da fonte** = pipeline inteiro da demo, **não**
  inferência · **nada aqui é tempo real**

### 05:20 · README erros → limitações
- Falha dominante = **objeto não detectado**, não confusão de classe
- D2 **189 / 51 / 116** · S1 **205 / 56 / 100** (TP/FP/FN)
- ❗ `vest_loose`: **7 instâncias em 2 imagens** · D2 recuperou **zero** · não decide nada
- Limites: conjunto pequeno · queda validação→teste · sem rastreamento · **teste gasto**

### 06:00 · Colab → README conclusão
- Repositório público · relatório técnico · Colab executável
- **TREINO / AVALIAÇÃO** = receita e resultados registrados — **não** retreinam,
  **não** reabrem o teste · **INFERÊNCIA** opcional, pesos verificados por hash
- 🎯 Fechar com: *localização amplamente semelhante · máscara = representação
  espacial mais rica · com custo computacional · a representação certa depende do
  que a aplicação precisa medir*

---

### Se o tempo apertar, corte nesta ordem
1. Memória de inferência (03:05)
2. Matrizes de confusão (05:20)
3. "o Colab não retreina" (06:00)

**Nunca cortar:** o erro visível no vídeo · a limitação de `vest_loose` · a recusa
de declarar vencedor.
