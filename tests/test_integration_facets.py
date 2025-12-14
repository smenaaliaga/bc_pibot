"""
Test de integración: Pregunta → JointBERT → Normalización → Detección de Serie
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from orchestrator.intents.joint_bert_classifier import PIBotPredictor
from orchestrator.data.get_series import detect_series_code

def test_integration():
    """Test del flujo completo"""
    print("\n" + "="*70)
    print("TEST DE INTEGRACIÓN: Pregunta → Entidades → Serie")
    print("="*70)
    
    # Inicializar clasificador
    print("\n1. Inicializando JointBERT...")
    model_dir = "model/in"  # Path al modelo entrenado
    classifier = PIBotPredictor(model_dir=model_dir)
    
    # Casos de prueba
    test_cases = [
        {
            "question": "¿Cuál fue el Imacec minero en marzo 2024?",
            "expected_indicator": "imacec",
            "expected_sector": "minero",
            "expected_series": "F032.IMC.IND.Z.Z.EP18.03.Z.0.M"
        },
        {
            "question": "Dame el imacec de producción de bienes del último mes",
            "expected_indicator": "imacec",
            "expected_sector": "producción de bienes",
            "expected_series": "F032.IMC.IND.Z.Z.EP18.PB.Z.0.M"
        },
        {
            "question": "Muéstrame el imacec general",
            "expected_indicator": "imacec",
            "expected_sector": None,
            "expected_series": "F032.IMC.IND.Z.Z.EP18.Z.Z.0.M"
        },
        {
            "question": "¿Cómo está el sector minería este año?",
            "expected_indicator": "imacec",
            "expected_sector": "minero",
            "expected_series": "F032.IMC.IND.Z.Z.EP18.03.Z.0.M"
        }
    ]
    
    print("\n2. Ejecutando casos de prueba...")
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'-'*70}")
        print(f"CASO {i}: {test['question']}")
        print(f"{'-'*70}")
        
        # Clasificar pregunta
        result = classifier.classify(test['question'])
        
        print(f"\n📊 Resultado JointBERT:")
        print(f"  Intent: {result.get('intent')}")
        print(f"  Entidades detectadas: {result.get('entities', {})}")
        print(f"  Entidades normalizadas:")
        
        normalized = result.get('normalized', {})
        for key, value in normalized.items():
            if isinstance(value, dict):
                print(f"    {key}:")
                for k, v in value.items():
                    print(f"      {k}: {v}")
            else:
                print(f"    {key}: {value}")
        
        # Extraer valores para detección
        indicator_norm = normalized.get('indicator', {})
        sector_norm = normalized.get('sector', {})
        
        # Usar 'standard_name' que ya tiene el formato de facets del catálogo
        indicator_value = indicator_norm.get('standard_name') if indicator_norm else None
        sector_value = sector_norm.get('standard_name') if sector_norm else None
        
        print(f"\n🔍 Valores para detección:")
        print(f"  Indicator: {indicator_value}")
        print(f"  Sector: {sector_value}")
        
        # Detectar serie
        series_result = detect_series_code(
            indicator=indicator_value,
            sector=sector_value
        )
        
        print(f"\n✅ Serie detectada:")
        print(f"  Código: {series_result['series_code']}")
        print(f"  Título: {series_result['metadata']['title']}")
        print(f"  Método: {series_result['matched_by']}")
        
        # Validar resultado
        expected = test['expected_series']
        actual = series_result['series_code']
        
        if actual == expected:
            print(f"\n✅ PASS: Serie correcta detectada")
        else:
            print(f"\n❌ FAIL: Esperado {expected}, obtenido {actual}")
    
    print("\n" + "="*70)
    print("TEST DE INTEGRACIÓN COMPLETADO")
    print("="*70)

if __name__ == "__main__":
    test_integration()
