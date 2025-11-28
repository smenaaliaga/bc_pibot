"""
state_manager.py
----------------
Módulo para la gestión del estado de la conversación y memoria a corto plazo.
Soporta Redis para persistencia y manejo de concurrencia, con fallback a memoria local.
"""

import json
import logging
from typing import Optional, Dict, Any, Tuple

try:
    import redis
except ImportError:
    redis = None

logger = logging.getLogger(__name__)

class StateManager:
    """Gestor de estado de conversación.
    
    Permite almacenar y recuperar contexto de conversación (series activas, años, filtros)
    y sugerencias pendientes (acciones ofrecidas al usuario).
    """
    
    def __init__(self, redis_url: Optional[str] = None, ttl: int = 3600):
        self.enabled = False
        self.ttl = ttl
        self._local_storage = {}  # Fallback local
        
        if redis and redis_url:
            try:
                self.client = redis.Redis.from_url(redis_url, decode_responses=True)
                self.client.ping()
                self.enabled = True
                # Mask password in logs
                safe_url = redis_url
                if "@" in safe_url:
                    try:
                        prefix, rest = safe_url.split("@", 1)
                        if ":" in prefix:
                            scheme_user, _pass = prefix.rsplit(":", 1)
                            safe_url = f"{scheme_user}:***@{rest}"
                    except Exception:
                        pass
                logger.info(f"[STATE_MANAGER] Conectado a Redis: {safe_url}")
            except Exception as e:
                logger.warning(f"[STATE_MANAGER] No se pudo conectar a Redis: {e}. Usando memoria local.")
        else:
            logger.info("[STATE_MANAGER] Redis no configurado. Usando memoria local.")

    def _key(self, session_id: str) -> str:
        return f"pibot:sess:{session_id}"

    def get_context(self, session_id: str) -> Dict[str, Any]:
        """Recupera el contexto completo de la sesión."""
        if not self.enabled:
            return self._local_storage.get(session_id, {})
        
        try:
            data = self.client.get(self._key(session_id))
            return json.loads(data) if data else {}
        except Exception as e:
            logger.error(f"[STATE_MANAGER] Error leyendo contexto: {e}")
            return {}

    def update_context(self, session_id: str, updates: Dict[str, Any]):
        """Actualiza campos específicos del contexto."""
        if not self.enabled:
            if session_id not in self._local_storage:
                self._local_storage[session_id] = {}
            self._local_storage[session_id].update(updates)
            return

        try:
            ctx = self.get_context(session_id)
            ctx.update(updates)
            self.client.setex(self._key(session_id), self.ttl, json.dumps(ctx))
        except Exception as e:
            logger.error(f"[STATE_MANAGER] Error actualizando contexto: {e}")

    def set_pending_suggestion(self, session_id: str, action_type: str, meta: Dict[str, Any] = None):
        """Registra explícitamente qué acción se le ofreció al usuario (Short Term Memory)."""
        logger.info(f"[STATE_MANAGER] Setting pending suggestion: {action_type} for {session_id}")
        self.update_context(session_id, {
            "pending_suggestion": action_type,
            "suggestion_meta": meta or {}
        })

    def get_pending_suggestion(self, session_id: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """Recupera la última sugerencia pendiente y sus metadatos."""
        ctx = self.get_context(session_id)
        return ctx.get("pending_suggestion"), ctx.get("suggestion_meta") or {}

    def clear_pending_suggestion(self, session_id: str):
        """Limpia la sugerencia pendiente una vez consumida."""
        self.update_context(session_id, {"pending_suggestion": None, "suggestion_meta": None})

    def clear_session(self, session_id: str):
        """Borra toda la sesión."""
        if not self.enabled:
            self._local_storage.pop(session_id, None)
            return
        try:
            self.client.delete(self._key(session_id))
        except Exception as e:
            logger.error(f"[STATE_MANAGER] Error borrando sesión: {e}")
