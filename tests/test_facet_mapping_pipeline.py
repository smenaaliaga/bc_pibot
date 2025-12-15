"""
Test unitario del mapeo de facets (sin necesitar el modelo JointBERT)
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from orchestrator.utils.facet_mapping import (
    normalize_sector_to_facet,
    normalize_indicator_to_facet
)
from orchestrator.utils.component_normalizer import normalize_component
from orchestrator.utils.indicator_normalizer import standardize_indicator
from orchestrator.data.get_series import detect_series_code

def test_facet_mapping():
    """Test del pipeline de normalización → facet mapping → detección"""
    print("\n" + "="*70)
    print("TEST DE MAPEO DE FACETS")
    print("="*70)
    
    test_cases = [
        {
            "raw_indicator": "imacec",
            "raw_sector": "minería",
            "expected_series": "F032.IMC.IND.Z.Z.EP18.03.Z.0.M",
            "expected_title": "Imacec minero"
        },
        {
            "raw_indicator": "IMACEC",
            "raw_sector": "producción de bienes",
            "expected_series": "F032.IMC.IND.Z.Z.EP18.PB.Z.0.M",
            "expected_title": "Imacec producción de bienes"
        },
        {
            "raw_indicator": "imacec",
            "raw_sector": "servicios",
            "expected_series": "F032.IMC.IND.Z.Z.EP18.SV.Z.0.M",
            "expected_title": "Imacec no minero"
        },
        {
            "raw_indicator": "imacec",
            "raw_sector": None,
            "expected_series": "F032.IMC.IND.Z.Z.EP18.Z.Z.0.M",
            "expected_title": "Imacec empalmado"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'-'*70}")
        print(f"CASO {i}: {test['raw_indicator']} + {test['raw_sector']}")
        print(f"{'-'*70}")
        
        # Paso 1: Normalizar indicador
        indicator_norm = standardize_indicator(test['raw_indicator'])
        indicator_value = indicator_norm['indicator'].lower()
        print(f"\n1️⃣  Indicador normalizado: '{indicator_value}'")
        
        # Paso 2: Mapear a facet del catálogo
        indicator_facet = normalize_indicator_to_facet(indicator_value)
        print(f"2️⃣  Indicador facet: '{indicator_facet}'")
        
        # Paso 3: Normalizar sector (si existe)
        if test['raw_sector']:
            sector_norm = normalize_component(test['raw_sector'])
            print(f"\n3️⃣  Sector normalizado: '{sector_norm}'")
            
            # Paso 4: Mapear sector a facet
            sector_facet = normalize_sector_to_facet(sector_norm)
            print(f"4️⃣  Sector facet: '{sector_facet}'")
        else:
            sector_facet = None
            print(f"\n3️⃣  Sin sector especificado")
        
        # Paso 5: Detectar serie
        print(f"\n5️⃣  Detectando serie con:")
        print(f"     indicator='{indicator_facet}'")
        print(f"     sector='{sector_facet}'")
        
        result = detect_series_code(
            indicator=indicator_facet,
            sector=sector_facet
        )
        
        # Mostrar resultado
        print(f"\n✅ Resultado:")
        print(f"   Serie: {result['series_code']}")
        print(f"   Título: {result['metadata']['title']}")
        print(f"   Método: {result['matched_by']}")
        
        # Validar
        expected = test['expected_series']
        actual = result['series_code']
        expected_title = test['expected_title']
        actual_title = result['metadata']['title']
        
        if actual == expected and expected_title in actual_title:
            print(f"\n   ✅ PASS")
        else:
            print(f"\n   ❌ FAIL")
            print(f"      Esperado: {expected}")
            print(f"      Obtenido: {actual}")
    
    print("\n" + "="*70)
    print("RESUMEN DEL PIPELINE:")
    print("="*70)
    print("1. Raw Input → Normalizer (standardize_indicator, normalize_component)")
    print("2. Normalized → Facet Mapper (normalize_indicator_to_facet, normalize_sector_to_facet)")
    print("3. Facets → Series Detector (detect_series_code)")
    print("4. Series Detector → Catalog Lookup (series_catalog.json)")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_facet_mapping()
