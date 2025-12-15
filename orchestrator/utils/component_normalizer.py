"""
Normalizador de componentes económicos.
Convierte diferentes variaciones de escritura a formas estándar.
"""

import re
from difflib import SequenceMatcher


class ComponentNormalizer:
    """
    Normaliza componentes económicos a su forma estándar del catálogo.
    
    Los valores de salida coinciden con catalog/series_catalog.json -> standard_names.component:
    - "minero", "no minero", "produccion de bienes", "imacec", "industria",
      "resto de bienes", "comercio", "servicios", "a costo de factores",
      "impuestos sobre los productos"
    """
    
    def __init__(self):
        # Diccionario de patrones de normalización
        # Clave: forma normalizada (coincide con standard_names.component), Valor: lista de patrones regex
        # IMPORTANTE: "no minero" debe ir antes que "minero" para evitar falsos positivos
        self.component_patterns = {
            "no minero": [
                r"no\s+miner[ao]",
                r"no\s+miner[ií]a",
                r"sin\s+miner[ií]a",
                r"excluye?\s+miner[ií]a",
                r"excepto\s+miner[ií]a",
                r"menos\s+miner[ií]a",
            ],
            "produccion de bienes": [
                r"producci[oó]n\s+de\s+bienes?",
                r"producion\s+de\s+bienes?",
                r"produc[cs]ion\s+de\s+bienes?",
                r"pr[od]uc+ion\s+de\s+bienes?",
                r"pr[od]+[ou]c+ion\s+(?:de\s+)?bienes?",
                r"produccion\s+bienes?",
                r"prod\s+de\s+bienes?",
                r"prod\.?\s+bienes?",
                r"pib\s+bienes?",
            ],
            "minero": [
                r"\bminer[ií]a\b",
                r"\bminera\b",
                r"\bmieneria\b",
                r"\bmineri[ao]\b",
                r"\bminr[ií]a\b",
                r"\bminria\b",
                r"\bmineri?a\b",
                r"sector\s+miner[ií]a",
                r"sector\s+minero",
                r"\bminero\b",
            ],
            "industria": [
                r"industri[ao]",
                r"industr[ií]a",
                r"sector\s+industri[ao]",
                r"sector\s+industrial",
                r"manufacturera?",
            ],
            "resto de bienes": [
                r"resto\s+de\s+bienes?",
                r"resto\s+bienes?",
                r"otros?\s+bienes?",
                r"demas\s+bienes?",
                r"dem[aá]s\s+bienes?",
            ],
            "comercio": [
                r"comercio",
                r"comersio",
                r"comers?io",
                r"comerci?o",
                r"sector\s+comercio",
                r"comerci[ao]l",
            ],
            "servicios": [
                r"servicios?",
                r"servisios?",
                r"serv[ií]cios?",
                r"serbicios?",
                r"sector\s+servicios?",
                r"sector\s+terciario",
            ],
            "a costo de factores": [
                r"a\s+costo\s+de\s+factores?",
                r"costo\s+de\s+factores?",
                r"costos?\s+factores?",
                r"precio\s+de\s+factores?",
                r"precios?\s+factores?",
            ],
            "impuestos sobre los productos": [
                r"impuestos?\s+sobre\s+(?:los\s+)?productos?",
                r"impuestos?\s+(?:sobre\s+)?producci[oó]n",
                r"impuestos?\s+productos?",
                r"impuesto\s+sobre\s+producci[oó]n",
                r"impuestos?\s+prod\b",
                r"inpuestos?\s+(?:sobre\s+)?productos?",
                r"inpuestos?\s+sobre",
                r"tax\s+productos?",
                r"(?:los\s+)?productos$",  # Solo "productos" al final
                r"^productos$",            # Solo "productos"
            ],
            "imacec": [
                r"^imacec$",
                r"^total$",
                r"^general$",
            ],
        }
    
    def _similarity(self, text1, text2):
        """
        Calcula la similitud entre dos textos.
        
        Args:
            text1 (str): Primer texto
            text2 (str): Segundo texto
            
        Returns:
            float: Similitud entre 0 y 1
        """
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def normalize(self, component_text, similarity_threshold=0.7):
        """
        Normaliza un texto de componente a su forma estándar.
        
        Args:
            component_text (str): Texto del componente a normalizar
            similarity_threshold (float): Umbral de similitud para fuzzy matching (0-1)
            
        Returns:
            str: Componente normalizado o el texto original si no se encuentra coincidencia
        """
        if not component_text:
            return component_text
        
        # Convertir a minúsculas y limpiar espacios extras
        text = component_text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Paso 1: Buscar coincidencias exactas con los patrones regex
        for normalized_form, patterns in self.component_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return normalized_form
        
        # Paso 2: Si no hay coincidencia exacta, buscar por similitud (fuzzy matching)
        best_match = None
        best_similarity = 0
        
        for normalized_form in self.component_patterns.keys():
            # Comparar con el nombre normalizado
            similarity = self._similarity(text, normalized_form)
            
            if similarity > best_similarity and similarity >= similarity_threshold:
                best_similarity = similarity
                best_match = normalized_form
        
        # Si se encuentra una coincidencia por similitud, devolverla
        if best_match:
            return best_match
        
        # Si no se encuentra coincidencia, devolver el texto limpio
        return text
    
    def normalize_batch(self, component_list):
        """
        Normaliza una lista de componentes.
        
        Args:
            component_list (list): Lista de textos de componentes
            
        Returns:
            list: Lista de componentes normalizados
        """
        return [self.normalize(component) for component in component_list]
    
    def get_valid_components(self):
        """
        Retorna la lista de componentes válidos (normalizados).
        
        Returns:
            list: Lista de componentes válidos
        """
        return list(self.component_patterns.keys())


# Función de utilidad para uso directo
def normalize_component(component_text):
    """
    Función de utilidad para normalizar un componente sin instanciar la clase.
    
    Args:
        component_text (str): Texto del componente a normalizar
        
    Returns:
        str: Componente normalizado
    """
    normalizer = ComponentNormalizer()
    return normalizer.normalize(component_text)
