"""
Script de ejemplo y prueba para el detector de series.

Muestra cómo usar la función detect_series_code con diferentes casos de uso.
"""

import sys
import os

# Agregar el path del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from orchestrator.data.get_series import (
    detect_series_code,
    add_series_mapping,
    get_available_mappings,
    get_series_info
)


def test_basic_detection():
    """Prueba detección básica con valores por defecto"""
    print("\n" + "="*70)
    print("TEST 1: Detección con valores por defecto (sin parámetros)")
    print("="*70)
    
    result = detect_series_code()
    
    print(f"Serie: {result['series_code']}")
    print(f"Indicador: {result['indicator']}")
    print(f"Componente: {result['component']}")
    print(f"Título: {result['metadata']['title']}")
    print(f"Método: {result['matched_by']}")


def test_explicit_params():
    """Prueba con parámetros explícitos"""
    print("\n" + "="*70)
    print("TEST 2: Detección con parámetros explícitos")
    print("="*70)
    
    # Caso 1: IMACEC - Producción de bienes
    print("\n--- IMACEC - Producción de Bienes ---")
    result = detect_series_code(
        indicator="imacec",
        component="produccion de bienes"
    )
    print(f"Serie: {result['series_code']}")
    print(f"Esperado: F032.IMC.IND.Z.Z.EP18.PB.Z.0.M")
    print(f"Match: {'OK' if result['series_code'] == 'F032.IMC.IND.Z.Z.EP18.PB.Z.0.M' else 'FAIL'}")
    
    # Caso 2: IMACEC - Minería
    print("\n--- IMACEC - Minería ---")
    result = detect_series_code(
        indicator="imacec",
        component="mineria"
    )
    print(f"Serie: {result['series_code']}")
    print(f"Esperado: F032.IMC.IND.Z.Z.EP18.03.Z.0.M")
    print(f"Match: {'OK' if result['series_code'] == 'F032.IMC.IND.Z.Z.EP18.03.Z.0.M' else 'FAIL'}")
    
    # Caso 3: IMACEC - IMACEC (default)
    print("\n--- IMACEC - IMACEC (default) ---")
    result = detect_series_code(
        indicator="imacec",
        component="imacec"
    )
    print(f"Serie: {result['series_code']}")
    print(f"Esperado: F032.IMC.IND.Z.Z.EP18.Z.Z.0.M")
    print(f"Match: {'OK' if result['series_code'] == 'F032.IMC.IND.Z.Z.EP18.Z.Z.0.M' else 'FAIL'}")


def test_normalized_dict():
    """Prueba con diccionario normalized (como viene de JointBERT)"""
    print("\n" + "="*70)
    print("TEST 3: Detección desde diccionario normalized")
    print("="*70)
    
    # Simulando estructura del normalizer
    normalized = {
        "indicator": {
            "standard_name": "IMACEC",
            "original": "imacec"
        },
        "sector": {
            "standard_name": "mineria",
            "original": "sector minero"
        },
        "period": {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "granularity": "monthly"
        }
    }
    
    result = detect_series_code(normalized=normalized)
    
    print(f"Serie: {result['series_code']}")
    print(f"Indicador detectado: {result['indicator']}")
    print(f"Componente detectado: {result['component']}")
    print(f"Título: {result['metadata']['title']}")
    print(f"Método: {result['matched_by']}")


def test_fuzzy_matching():
    """Prueba de matching parcial (fuzzy)"""
    print("\n" + "="*70)
    print("TEST 4: Matching parcial (fuzzy)")
    print("="*70)
    
    # Probar con variaciones de escritura
    tests = [
        ("imacec", "minero"),
        ("imacec", "minería"),
        ("IMACEC", "MINERIA"),
        ("imacec", "producción de bienes"),
        ("imacec", "produccion de bienes"),
    ]
    
    for ind, comp in tests:
        result = detect_series_code(indicator=ind, component=comp)
        print(f"{ind} + {comp:25s} → {result['series_code']}")


def test_add_new_mapping():
    """Prueba agregar mapeos dinámicamente"""
    print("\n" + "="*70)
    print("TEST 5: Agregar nuevos mapeos dinámicamente")
    print("="*70)
    
    # Agregar un nuevo componente de IMACEC
    print("\nAgregando: IMACEC - Servicios")
    add_series_mapping(
        indicator="imacec",
        component="servicios",
        series_code="F032.IMC.IND.Z.Z.EP18.SV.Z.0.M"  # Ejemplo
    )
    
    # Probar que funciona
    result = detect_series_code(indicator="imacec", component="servicios")
    print(f"Serie detectada: {result['series_code']}")
    
    # Agregar un nuevo indicador completo
    print("\nAgregando nuevo indicador: PIB")
    add_series_mapping(
        indicator="pib",
        component="total",
        series_code="F032.PIB.FLU.N.CLP.EP18.Z.Z.0.T"
    )
    
    result = detect_series_code(indicator="pib", component="total")
    print(f"Serie detectada: {result['series_code']}")


def test_reverse_lookup():
    """Prueba búsqueda inversa por código de serie"""
    print("\n" + "="*70)
    print("TEST 6: Búsqueda inversa (serie → info)")
    print("="*70)
    
    codes = [
        "F032.IMC.IND.Z.Z.EP18.Z.Z.0.M",
        "F032.IMC.IND.Z.Z.EP18.PB.Z.0.M",
        "F032.IMC.IND.Z.Z.EP18.03.Z.0.M",
    ]
    
    for code in codes:
        info = get_series_info(code)
        if info:
            print(f"{code}")
            print(f"  → Título: {info.get('title', 'N/A')}")
            print(f"  → Indicador: {info.get('facets', {}).get('indicator', 'N/A')}")
            print(f"  → Sector: {info.get('facets', {}).get('sector', 'N/A')}")
            print(f"  → Clasificación: {info.get('classification', 'N/A')}")
            print()


def test_available_mappings():
    """Muestra todos los mapeos disponibles"""
    print("\n" + "="*70)
    print("TEST 7: Ver todos los mapeos disponibles")
    print("="*70)
    
    mappings = get_available_mappings()
    
    for indicator, components in mappings.items():
        print(f"\n{indicator.upper()}:")
        for component in components:
            # Buscar el código de serie para este componente
            result = detect_series_code(indicator=indicator, component=component)
            print(f"  • {component:30s} → {result['series_code']}")


def main():
    """Ejecutar todas las pruebas"""
    print("\n" + "="*70)
    print("PRUEBAS DEL DETECTOR DE SERIES")
    print("="*70)
    
    test_basic_detection()
    test_explicit_params()
    test_normalized_dict()
    test_fuzzy_matching()
    test_add_new_mapping()
    test_reverse_lookup()
    test_available_mappings()
    
    print("\n" + "="*70)
    print("PRUEBAS COMPLETADAS")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
