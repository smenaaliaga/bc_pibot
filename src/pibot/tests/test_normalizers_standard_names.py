"""
Test de los normalizadores para validar que produzcan valores coincidentes con standard_names del catálogo
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from orchestrator.utils.indicator_normalizer import standardize_indicator
from orchestrator.utils.component_normalizer import normalize_component

def test_indicator_normalizer():
    """Test del indicador normalizer"""
    print("\n" + "="*70)
    print("TEST INDICATOR_NORMALIZER: Valores coinciden con standard_names")
    print("="*70)
    
    test_cases = [
        ("imacec", "imacec"),
        ("IMACEC", "imacec"),
        ("Imacec", "imacec"),
        ("indice mensual de actividad economica", "imacec"),
        ("pib", "pib"),
        ("PIB", "pib"),
        ("producto interno bruto", "pib"),
    ]
    
    print("\nIndicador Normalizer debe retornar: 'imacec' o 'pib' (lowercase)")
    print("-" * 70)
    
    for input_text, expected in test_cases:
        result = standardize_indicator(input_text)
        indicator = result.get('indicator')
        status = "✅ PASS" if indicator == expected else "❌ FAIL"
        print(f"{status} | Input: '{input_text:40s}' → '{indicator}' (esperado: '{expected}')")
    
    print()

def test_component_normalizer():
    """Test del componente normalizer"""
    print("\n" + "="*70)
    print("TEST COMPONENT_NORMALIZER: Valores coinciden con standard_names")
    print("="*70)
    
    # Valores esperados del catálogo:
    # "minero", "no minero", "produccion de bienes", "imacec", "industria",
    # "resto de bienes", "comercio", "servicios", "a costo de factores",
    # "impuestos sobre los productos"
    
    test_cases = [
        ("minería", "minero"),
        ("mineria", "minero"),
        ("sector minero", "minero"),
        ("no minero", "no minero"),
        ("producción de bienes", "produccion de bienes"),
        ("produccion de bienes", "produccion de bienes"),
        ("imacec", "imacec"),
        ("total", "imacec"),
        ("general", "imacec"),
        ("industria", "industria"),
        ("resto de bienes", "resto de bienes"),
        ("comercio", "comercio"),
        ("servicios", "servicios"),
        ("sector servicios", "servicios"),
        ("a costo de factores", "a costo de factores"),
        ("impuestos sobre los productos", "impuestos sobre los productos"),
    ]
    
    print("\nComponente Normalizer debe retornar valores de standard_names.component:")
    print("['minero', 'no minero', 'produccion de bienes', 'imacec', 'industria',")
    print(" 'resto de bienes', 'comercio', 'servicios', 'a costo de factores',")
    print(" 'impuestos sobre los productos']")
    print("-" * 70)
    
    for input_text, expected in test_cases:
        result = normalize_component(input_text)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        print(f"{status} | Input: '{input_text:35s}' → '{result:30s}' (esperado: '{expected}')")
    
    print()

if __name__ == "__main__":
    test_indicator_normalizer()
    test_component_normalizer()
    
    print("="*70)
    print("TESTS COMPLETADOS")
    print("="*70)
