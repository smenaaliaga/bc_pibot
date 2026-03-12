# Master Plan Validation Report (2026-03-12)

- Generated at: 2026-03-12T15:25:50.254614+00:00
- Baseline QA file: `qa/reports/response_baseline_master_20260312.txt`
- New features QA file: `qa/reports/response_new_features_master_20260312_v2.txt`
- Regression QA file: `qa/reports/response_regression_master_20260312_v3.txt`

## Overall Status

- Checks passed: **16/16**
- Master-plan implementation status: **VALID**

## Detailed Checks

- [OK] NF-1 PIB corrientes: Cuál es el PIB en pesos?
- [OK] NF-2 PIB corrientes: Cuánto es el PIB de Chile?
- [OK] NF-3 PIB corrientes: A cuánto asciende el PIB en Chile?
- [OK] NF-4 Particion gasto guardrail: Cuánto pesa el consumo en el PIB?
- [OK] NF-5 Particion gasto guardrail: Cuánto pesa el consumo en el PIB?
- [OK] NF-8 Particion gasto guardrail: Cuál es la participación de la inversión en el PIB?
- [OK] NF-9 PIB per capita guardrail: Cuál es el PIB PER capita
- [OK] NF-10 PIB per capita guardrail: ii. Cual es el PIB per capita del 2025
- [OK] NF-11 PIB per capita guardrail: iii. Cual es el PIB per capita 2024
- [OK] NF-12 PIB per capita guardrail: Iv. Cual es el PIB per capita entre 2022 y 2025
- [OK] RG-53 PIB corrientes serie nominal: Cual es el valor del PIB a precios corrientes del 2020
- [OK] RG-61 PIB corrientes serie nominal: Cual es el pib a precios corrientes
- [OK] RG-11 Particion mineria share: ¿Cuánto pesa minería en el PIB?
- [OK] RG-12 PIB per capita guardrail: Cual es el PIB per capita
- [OK] RG-22 PIB per capita guardrail: ¿Cuál fue el último producto interno bruto o PIB per cápita o por persona?
- [OK] RG-56 PIB per capita guardrail: cual es el pib per capita del 2029

## Notes

- `qa_batch.py` is homologated to Streamlit-like streaming and marker handling.
- PIB corrientes now uses nominal series `F032.PIB.FLU.N.CLP.EP18.Z.Z.0.T` in validated cases.
- PIB per capita currently has no enabled catalog series; guardrail message is returned by design.
- Gasto-share queries (consumo/inversion/exportaciones/importaciones) currently return a guardrail because the catalog share family for gasto has no series configured.
- Residual risk: some broad wording can still route to RAG/fallback instead of DATA (classification/routing variability).
