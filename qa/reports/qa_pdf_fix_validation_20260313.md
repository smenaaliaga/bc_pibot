# QA PDF Fix Validation Report (2026-03-13)

- Generated at: 2026-03-13T03:52:34.800859+00:00
- Source QA file: `qa/reports/response_qa_pdf_20260313.txt`
- Target series for Cuanto/Cuánto parity: `F032.PIB.FLU.N.CLP.EP18.Z.Z.0.T`

## Overall Status

- Checks passed: **3/3**
- Status: **VALID**

## Checks

- [OK] Primera consulta IMACEC evita fallback generico
- [OK] Marker FOLLOWUP presente en respuesta PIB (bridge stream/state)
- [OK] Paridad Cuanto/Cuánto/combining-accent en sesion compartida

## Shared Session Diagnostics

- Q: cual es el valor del imacec | route=data | intent=value | context=standalone | series=F032.IMC.IND.Z.Z.EP18.Z.Z.0.M | price=enc | elapsed_s=4.204
- Q: cual es el valor del pib | route=data | intent=value | context=standalone | series=F032.PIB.FLU.R.CLP.EP18.Z.Z.0.T | price=enc | elapsed_s=3.883
- Q: Cuanto es el PIB de Chile? | route=data | intent=value | context=standalone | series=F032.PIB.FLU.N.CLP.EP18.Z.Z.0.T | price=co | elapsed_s=4.631
- Q: ¿Cuánto es el PIB de Chile? | route=data | intent=value | context=standalone | series=F032.PIB.FLU.N.CLP.EP18.Z.Z.0.T | price=co | elapsed_s=2.580
- Q: ¿Cuánto es el PIB de Chile? | route=data | intent=value | context=standalone | series=F032.PIB.FLU.N.CLP.EP18.Z.Z.0.T | price=co | elapsed_s=2.221

## Followup Marker Diagnostics

- PIB query route=data | has_followup_marker=True | elapsed_s=41.512
