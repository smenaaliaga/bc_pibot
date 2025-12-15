"""
Test de integración: Verificar que los normalizadores producen valores 
que coinciden con standard_names del catálogo.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from orchestrator.utils.indicator_normalizer import standardize_indicator
from orchestrator.utils.component_normalizer import normalize_component
from orchestrator.data.get_series import detect_series_code


def test_normalizers_match_catalog():
    """
    Verifica que los normalizadores producen valores que coinciden 
    con standard_names del catálogo de series.
    """
    print("\n" + "="*70)
    print("TEST: Verificar normalizadores vs catalog.standard_names")
    print("="*70)
    
    test_cases = [
        {
            "indicator_text": "imacec",
            "component_text": "minero",
            "expected_indicator": "imacec",
            "expected_component": "minero",
            "expected_series": "F032.IMC.IND.Z.Z.EP18.03.Z.0.M"
        },
        {
            "indicator_text": "IMACEC",
            "component_text": "mineria",
            "expected_indicator": "imacec",
            "expected_component": "minero",
            "expected_series": "F032.IMC.IND.Z.Z.EP18.03.Z.0.M"
        },
        {
            "indicator_text": "imacec",
            "component_text": "produccion de bienes",
            "expected_indicator": "imacec",
            "expected_component": "produccion de bienes",
            "expected_series": "F032.IMC.IND.Z.Z.EP18.PB.Z.0.M"
        },
        {
            "indicator_text": "imacec",
            "component_text": "productos",
            "expected_indicator": "imacec",
            "expected_component": "impuestos sobre los productos",
            "expected_series": None  # No hay serie de IMACEC para impuestos
        },
        {
            "indicator_text": "imacec",
            "component_text": "no minero",
            "expected_indicator": "imacec",
            "expected_component": "no minero",
            "expected_series": "F032.IMC.IND.Z.Z.EP18.N03.Z.0.M"
        },
        {
            "indicator_text": "imacec",
            "component_text": "imacec",
            "expected_indicator": "imacec",
            "expected_component": "imacec",
            "expected_series": "F032.IMC.IND.Z.Z.EP18.Z.Z.0.M"
        },
    ]
    
    all_passed = True
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'-'*70}")
        print(f"CASO {i}: {test['indicator_text']} + {test['component_text']}")
        print(f"{'-'*70}")
        
        # Normalizar indicador
        indicator_result = standardize_indicator(test['indicator_text'])
        indicator_norm = indicator_result.get('indicator')
        
        print(f"📊 Indicador:")
        print(f"   Input:    {test['indicator_text']}")
        print(f"   Output:   {indicator_norm}")
        print(f"   Expected: {test['expected_indicator']}")
        
        if indicator_norm != test['expected_indicator']:
            print(f"   ❌ FAIL: Indicador no coincide")
            all_passed = False
        else:
            print(f"   ✅ PASS")
        
        # Normalizar componente
        component_norm = normalize_component(test['component_text'])
        
        print(f"\n📊 Componente:")
        print(f"   Input:    {test['component_text']}")
        print(f"   Output:   {component_norm}")
        print(f"   Expected: {test['expected_component']}")
        
        if component_norm != test['expected_component']:
            print(f"   ❌ FAIL: Componente no coincide")
            all_passed = False
        else:
            print(f"   ✅ PASS")
        
        # Detectar serie
        series_result = detect_series_code(
            indicator=indicator_norm,
            component=component_norm
        )
        
        series_code = series_result.get('series_code')
        
        print(f"\n📊 Serie detectada:")
        print(f"   Código:   {series_code}")
        print(f"   Expected: {test['expected_series']}")
        
        if test['expected_series']:
            if series_code == test['expected_series']:
                print(f"   ✅ PASS")
                if series_result.get('metadata'):
                    print(f"   Título: {series_result['metadata'].get('title')}")
            else:
                print(f"   ❌ FAIL: Serie no coincide")
                all_passed = False
        else:
            print(f"   ℹ️  No se espera serie específica para este caso")
            if series_code:
                print(f"   Código retornado: {series_code}")
                if series_result.get('metadata'):
                    print(f"   Título: {series_result['metadata'].get('title')}")
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ TODOS LOS TESTS PASARON")
    else:
        print("❌ ALGUNOS TESTS FALLARON")
    print("="*70)
    
    return all_passed


def test_catalog_standard_names_coverage():
    """
    Verifica que los normalizadores cubren todos los valores de 
    standard_names que existen en el catálogo.
    """
    print("\n" + "="*70)
    print("TEST: Cobertura de standard_names del catálogo")
    print("="*70)
    
    import json
    from pathlib import Path
    
    catalog_path = Path(__file__).parent.parent / "catalog" / "series_catalog.json"
    
    with open(catalog_path, 'r', encoding='utf-8') as f:
        catalog = json.load(f)
    
    # Extraer todos los valores únicos de standard_names
    indicators = set()
    components = set()
    
    for series_code, metadata in catalog.items():
        standard_names = metadata.get('standard_names', {})
        if standard_names:
            ind = standard_names.get('indicator')
            comp = standard_names.get('component')
            if ind:
                indicators.add(ind)
            if comp:
                components.add(comp)
    
    print(f"\n📊 Indicadores en catálogo: {sorted(indicators)}")
    print(f"📊 Componentes en catálogo: {sorted(components)}")
    
    # Verificar que los normalizadores pueden producir estos valores
    print(f"\n{'-'*70}")
    print("Verificando cobertura de componentes:")
    print(f"{'-'*70}")
    
    from orchestrator.utils.component_normalizer import ComponentNormalizer
    normalizer = ComponentNormalizer()
    valid_components = normalizer.get_valid_components()
    
    print(f"Componentes normalizables: {sorted(valid_components)}")
    
    missing = components - set(valid_components)
    if missing:
        print(f"\n⚠️  Componentes en catálogo sin patrón de normalización:")
        for comp in sorted(missing):
            print(f"   - {comp}")
    else:
        print(f"\n✅ Todos los componentes del catálogo tienen patrón de normalización")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    print("\n")
    print("="*70)
    print("SUITE DE TESTS: standard_names + normalizadores + series_detector")
    print("="*70)
    
    test_normalizers_match_catalog()
    test_catalog_standard_names_coverage()
    
    print("\n")
