# Portfolio-Construction-and-Backtesting

Quantitative finance coursework notebooks (University of Iowa):

- `Demo Portfolio Construction and Backtesting (1).ipynb` — portfolio
  construction and backtesting.
- `Demo Quantitative Stock Selection Model_ Analyst Forecast Dispersion (1).ipynb`
  — stock-selection model based on analyst forecast dispersion (IBES data).
- `Demo Quantitative Stock Selection Model_ Asset Growth (2).ipynb` —
  stock-selection model based on asset growth.

All three notebooks are fully executed, with outputs saved in the files.

## Data requirement (not included)

The notebooks read CSV files from a `classdata/` folder (course datasets,
including licensed IBES data). Those files are **not in this repo** and are
not redistributable, so the notebooks cannot be re-run from a fresh clone —
they are presented as executed coursework with their outputs intact. To
re-run one, place the course CSVs in `classdata/` next to the notebook and
execute top to bottom.

## Dependencies

```bash
pip install -r requirements.txt
```

## Cleanup note

This repo previously held seven notebooks: four copies of the portfolio
notebook and two of the forecast-dispersion notebook, with byte-identical
code. The duplicates were removed; one copy of each distinct notebook is
kept above.
