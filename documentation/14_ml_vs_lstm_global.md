# ML × LSTM de igual para igual: local, regional, global e LOCO

Comparação das duas famílias com as **mesmas linhas, o mesmo split, as mesmas 49 colunas** (selecionadas, sem
`anor`) e **o mesmo EQM**, nos clusters 1, 2, 3, 4 (RS/SC) e 12. Notebook:
`notebooks/feature_study_grupos/atual/ml_vs_lstm_global.ipynb`.

## Desenho

**Linhas.** Pares (estação, dia) dos clusters 1–4 (grade ERA5 de RS/SC) e 12, na hora do pico de rajada ERA5 daquele
dia. Só entram as linhas com janela de 24 h completa: 411.976 de 411.976.

**Split.** O de calendário do estudo (`station_dataset.calendar_split`), igual para todas as regiões e idêntico ao do
estudo de features (no cluster 3: 22.827 linhas de treino e 4.452 de validação):
- teste: jan/abr/jul/out de 2000–2024;
- validação: blocos (ano, mês);
- 2025: fora de tudo.

**Modelos.**
- **ML:** o campeão do cluster 3 em cada trimestre (MLP DJF, ExtraTrees MAM, CatBoost JJA, HistGB SON). Vê a hora do
  pico.
- **LSTM:** a do estudo (LSTM(96) → Dropout(0,3) → Dense(1), Huber, early stopping). Vê as 24 horas até o pico,
  calculadas com as mesmas regras (`era5_features.peak_windows_year`; a última hora da janela é idêntica à linha do
  pico).
- Uma seed (42) por ajuste.
- EQM ajustado na validação de cada treino.

**Desenhos.**

| desenho | treina com | avaliado em |
|---|---|---|
| A (só ML) | o modelo do cluster 3 do estudo | todos |
| B local | as estações do próprio cluster | todos |
| C regional | RS/SC (1–4) | todos |
| G global | 1–4 + 12 | todos |
| L LOCO global | todos menos o avaliado | o que ficou de fora |
| **GA global anual** | 1–4 + 12, **um modelo para todos os trimestres** (dia do ano como coluna, um EQM só); no ML, CatBoost | todos |

## Resultados (RMSE diário em m/s, meses de teste 2000–2024, sem EQM)

| | 1 | 2 | 3 | 4 | 12 |
|---|---|---|---|---|---|
| ERA5 | 2,62 | 2,54 | 2,66 | 3,44 | 2,27 |
| ML local | **2,28** | 2,33 | **2,40** | 2,74 | 1,95 |
| LSTM local | 2,38 | 2,34 | 2,47 | **2,72** | **1,92** |
| ML global | 2,41 | 2,29 | 2,46 | 2,79 | 2,05 |
| LSTM global | 2,41 | **2,28** | 2,43 | 2,74 | 1,97 |
| ML LOCO | 2,57 | 2,42 | 2,62 | 3,27 | 2,39 |
| LSTM LOCO | 2,71 | 2,39 | 2,68 | 3,36 | 2,54 |

2025, o ano nunca visto, mostra o mesmo quadro.

**POD no P99, com EQM, desenho global:**
- ML: 0,15–0,24;
- LSTM: 0,11–0,23;
- ERA5: 0,08–0,22.

Sem EQM, a LSTM fica em 0,00–0,07.

## Um modelo para o ano todo (GA) × um por trimestre (G)

RMSE sem EQM, meses de teste 2000–2024:

| | 1 | 2 | 3 | 4 | 12 |
|---|---|---|---|---|---|
| ERA5 | 2,62 | 2,54 | 2,66 | 3,44 | 2,27 |
| ML global por trimestre | 2,41 | 2,29 | 2,46 | 2,79 | 2,05 |
| ML global anual | 2,32 | 2,26 | 2,39 | 2,68 | 2,01 |
| LSTM global por trimestre | 2,41 | 2,28 | 2,43 | 2,74 | 1,97 |
| **LSTM global anual** | **2,28** | **2,24** | **2,33** | **2,57** | **1,85** |

- **O GA ganha do G nas duas famílias, em todos os clusters e trimestres, em 2000–2024 e em 2025.**
- **A LSTM anual é a melhor abordagem testada:** empata com o melhor modelo local no cluster 1 e ganha dele nos outros
  quatro.
- **Com EQM, a LSTM anual ganha do ERA5 no RMSE em todos os clusters** (1,95–2,70) e mantém ou supera a detecção de
  extremos (POD P99 de 0,13 a 0,28; ERA5 de 0,08 a 0,22).
- **Causa provável:** cada trimestre aproveita os dados dos outros (222 mil linhas de treino em vez de ~55 mil), e o
  dia do ano dá a época sem cortar os dados.
- **Ressalva:** no ML, o GA usa CatBoost e o G os campeões de cada trimestre, então parte do ganho pode ser do
  CatBoost. Na LSTM a arquitetura é a mesma nos dois.
- **Custo:** o modelo anual da LSTM é uma única unidade, de 6 minutos em T4.

## Conclusões

1. **Com estações da região no treino, ML e LSTM empatam no erro diário.** A diferença fica entre −0,08 e +0,15 m/s.
   A LSTM é um pouco melhor no modelo global.
2. **Em região não vista (LOCO), a LSTM generaliza pior que o ML**, de +0,06 a +0,15 m/s.
3. **O modelo global é o mais robusto** nas duas famílias: fica a ~0,1 m/s do melhor local em todos os clusters.
4. **Extremos:** a LSTM sem EQM praticamente não detecta o P99. Com EQM, as duas famílias chegam perto do ERA5 ou o
   superam, e o ML + EQM detecta um pouco mais.
5. **Custo:** a LSTM exige janelas de 24 h (~2 GB em disco) e GPU. As 49 unidades usaram ~37 min de T4.
6. **Recomendação (atualizada com o GA):** a **LSTM global anual**, sem EQM para o erro do dia a dia e com EQM para
   percentis e extremos. O **ML global anual (CatBoost)** fica a poucos centésimos e não precisa de janelas nem de GPU.

## Limites

- Uma seed.
- Arquiteturas não retriadas por região: o ML usa os campeões do cluster 3, e a LSTM, a arquitetura do estudo.
- O cluster 12 tem grade própria. Os clusters 1–4 compartilham a grade de RS/SC.

## Como rodar

```bash
uv run python scripts/treinar_global.py --stage base
uv run python scripts/treinar_global.py --stage janelas         # ~45 min
uv run python scripts/treinar_global.py --stage ml              # ~21 min (inclui o GA)
uv run python scripts/treinar_global.py --stage ml-anual        # só o GA, sem refazer o resto
MODAL_PROFILE=<perfil> modal run src/modal/global_lstm.py --stage upload        # ~2 GB; não interromper
MODAL_PROFILE=<perfil> modal run --detach src/modal/global_lstm.py --stage fit  # 48 unidades em T4
MODAL_PROFILE=<perfil> modal run src/modal/global_lstm.py --stage download
uv run python scripts/treinar_global.py --stage avaliar
uv run python scripts/build_notebook_global.py
```

Código:
- `src/feature_study/diagnostics/regional.py`: desenhos e avaliação;
- `src/feature_study/diagnostics/regional_lstm.py`;
- `src/feature_study/data/era5_features.py`: `peak_windows_year`;
- `src/feature_study/data/station_dataset.py`: `calendar_split`.
