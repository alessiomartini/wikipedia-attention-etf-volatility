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
- [x] Sezione matematica nel README (scritta da Claude su richiesta di Alessio, da rileggere)
- [x] Test Clark-West (modelli annidati) e simulazione della dimensione dei test
- [x] Notebook 01-04 con glossario finanziario, visualizzazioni, matematica
- [x] Estensione ML (gradient boosting, feature di attenzione ricche): peggio di HAR, nessun guadagno
- [x] Revisione critica nel README: assunzioni, alternative, parte più difficile, idee ML/DL
- [x] Estensione intraday (Alpaca IEX 15 min, RV giornaliera, HAR-RV) + notebook 05
- [ ] Dati intraday consolidati 1-5 min (a pagamento), panel su molte coppie, finestre attorno agli earnings

## Decisioni e cose provate
- Il 2026-11-14 una sessione automatica aveva aggiunto (senza dati reali, API in 429) classe con Ridge, ADF/DM a mano, CLI e test sintetici. Il 2026-11-19, con i dati reali scaricabili, ho sostituito tutto con una pipeline unica basata su statsmodels (HAR OLS, DM-HAC, Holm, ADF con p-value); rimossi `main.py`, `src/stats_tests.py`, i vecchi test e `requirements-dev.txt`. Nuovo test: `tests/test_no_lookahead.py` (pytest).
- Risultato: nullo su tutte e 6 le coppie (nessun guadagno OOS significativo dell'attenzione sopra la baseline HAR). Aggiunto Clark-West perché DM è troppo conservativo per modelli annidati (simulazione: 0.7% vs 4.7% di rifiuti sotto H0). Con GBM non si usa CW (non valido per alberi). Intraday: IEX copre solo 2-3% del volume e parte da luglio 2020 (~800 giorni di test, poca potenza); 15 min invece di 5 perché IEX è sottile su XLE/XLF/GLD. Limiti: 6 coppie scelte a mano, modello lineare, volatilità close-to-close.

## Log
- 2026-11-19: estensione intraday: senza controllo del giorno della settimana Clark-West+Holm darebbe QQQ/Nvidia e XLE/Petroleum significativi, con le dummy settimanali nessuna coppia sopravvive (p Holm migliore 0.149): stagionalità settimanale di visite e volatilità = falso segnale
- 2026-11-19: notebook 01-04, Clark-West + simulazione, estensione GBM, glossario finanziario e revisione critica nel README; XLK/Apple CW p=0.019 grezzo ma 0.11 dopo Holm
- 2026-11-19: progetto finito: baseline HAR, DM-HAC+Holm su 6 coppie 2016-2026, risultato nullo, README con figure; rimossa email dallo User-Agent
- 2026-10-18: aggiunte le idee da IDEE.md (protocollo unico)
- 2026-10-15: creata la roadmap per Projects Overview
