# Data

This repository intentionally does **not** redistribute the source CSV.

The analysis expects a cross-sectional Portsmouth domestic water-consumption file containing at least:

- `LSOA_CODE`: spatial identifier
- `TOTAL_CONSUMPTION`: numeric consumption outcome

The reference source file used during development also contained `DATA_SOURCE`, `Year`, `NUMBER_OF_METERS`, and an object identifier. In the analysed copy, the first three of those fields were structurally empty and were excluded from modelling.

## Expected location

Place the source file locally, for example:

```text
data/Portsmouth_Water_Domestic_Consumption_25-26.csv
```

Then run:

```bash
python src/analysis.py --data data/Portsmouth_Water_Domestic_Consumption_25-26.csv
```

## Licensing and provenance

Before redistributing any source data, verify the dataset's original licence and attribution requirements. The software licence in this repository applies to the code only; it does not grant rights to third-party data.

## Data-integrity principle

The script does not create synthetic fallback observations when the source file is missing. It fails explicitly instead, so published results cannot silently switch from real to simulated data.
