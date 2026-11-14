# Roadmap

<!--
Letto automaticamente da Projects Overview (https://alessiomartini.github.io/projects-overview/):
ogni titolo "##" è una fase, le checkbox "- [ ]" / "- [x]" danno la percentuale di avanzamento,
"## Decisioni e cose provate
- L'API di Wikimedia e Yahoo risponde HTTP 429 dall'ambiente di lavoro automatico: non ho scaricato dati reali, quindi i test usano solo dati sintetici e nessun risultato reale è stato prodotto o riportato.
- Il motore accetta DataFrame già caricati (`prices`, `views`) per poter testare offline.
- `requirements.txt` è in UTF-16: ho messo pytest in `requirements-dev.txt` invece di modificarlo.

## Log
- 2026-11-14: 🤖 aggiunti baseline (persistenza e solo lag di volatilità), CLI, percorso offline, test pytest su dati sintetici e risposte alle domande nel README
" è il diario (una riga per giorno, la più recente in alto).
-->

## Fatto
- [x] Dati: pageview Wikipedia e volatilità realizzata degli ETF
- [x] Analisi e risultati del prototipo

## Da fare
- [ ] Finire il progetto
  - [x] Test su dati sintetici (trasformazioni, lag, ordine dei fold, colonne baseline)
  - [x] Baseline di persistenza e Ridge solo con lag di volatilità da battere
  - [x] CLI con ticker/articolo/date (`main.py`)
  - [x] Salvataggio delle metriche per fold in CSV (`--out`)
  - [x] Risposte alle domande del README (sezione "Notes on the questions in the text")
  - [ ] Eseguire su dati reali (appena le API non rispondono più 429) e riportare solo i risultati veri
  - [ ] Test di stazionarietà (ADF) sulle serie usate
  - [ ] Più coppie articolo–ETF
  - [ ] Test di significatività della differenza modello vs baseline
  - [ ] Nota finale sui limiti con i risultati reali

## Log
- 2026-10-18: aggiunte le idee da IDEE.md (protocollo unico)
- 2026-10-15: creata la roadmap per Projects Overview
