
# Day Trade Scanner V2

Protótipo de laboratório para estudar confluências de análise técnica.

## Incluído

- tendência por EMA20/EMA50 + inclinação;
- aproximação algorítmica de LTA/LTB;
- suportes e resistências por pivôs;
- liquidez e varredura de máxima/mínima;
- falso rompimento;
- pullback aproximado;
- Fibonacci 38,2%, 50% e 61,8%;
- RSI;
- leitura de candles:
  - doji
  - martelo
  - estrela cadente
  - engolfo de alta/baixa
  - candle de força;
- score de confluências;
- veto de operação contra a tendência;
- gatilho somente com alinhamento mínimo;
- gráfico;
- backtest didático.

## Instalação

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Observação

A V2 é um laboratório. LTA/LTB, pivôs, liquidez e padrões de candle são
representados por regras matemáticas aproximadas. Eles precisam ser validados
com dados e observação visual antes de qualquer conclusão.

Não há integração de execução de ordens com corretoras/plataformas.

## Teste de uma semana

Use conta demo e registre, para cada sinal:
- horário;
- ativo;
- direção indicada;
- score;
- confluências presentes;
- veto;
- o que aconteceu depois;
- se o sinal apareceu cedo/tarde;
- se o candle mudou após o alerta.

Depois use essas observações para ajustar a V3.
