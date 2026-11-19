# Roadmap

<!--
Letto automaticamente da Projects Overview (https://alessiomartini.github.io/projects-overview/):
ogni titolo "##" è una fase, le checkbox "- [ ]" / "- [x]" danno la percentuale di avanzamento,
"## Log" è il diario (una riga per giorno, la più recente in alto).
-->

## Fatto
- [x] Dati: pageview Wikipedia e volatilità realizzata degli ETF
- [x] Analisi e risultati del prototipo

## Finitura
- [x] Pipeline riproducibile (fix yfinance, requirements UTF-8, cache dati)
- [x] Baseline HAR-RV vs modello con attenzione, walk-forward
- [x] Test Diebold-Mariano HAC + Holm su 6 coppie, ADF
- [x] README in inglese con figure e risultato nullo
- [ ] Sezione matematica nel README (Alessio)

## Decisioni e cose provate
- Il 2026-11-14 una sessione automatica aveva aggiunto (senza dati reali, API in 429) classe con Ridge, ADF/DM a mano, CLI e test sintetici. Il 2026-11-19, con i dati reali scaricabili, ho sostituito tutto con una pipeline unica basata su statsmodels (HAR OLS, DM-HAC, Holm, ADF con p-value); rimossi `main.py`, `src/stats_tests.py`, i vecchi test e `requirements-dev.txt`. Nuovo test: `tests/test_no_lookahead.py` (pytest).
- Risultato: nullo su tutte e 6 le coppie (nessun guadagno OOS significativo dell'attenzione sopra la baseline HAR). Limiti: 6 coppie scelte a mano, modello lineare, volatilità close-to-close.

## Log
- 2026-11-19: progetto finito: baseline HAR, DM-HAC+Holm su 6 coppie 2016-2026, risultato nullo, README con figure; rimossa email dallo User-Agent
- 2026-10-18: aggiunte le idee da IDEE.md (protocollo unico)
- 2026-10-15: creata la roadmap per Projects Overview
