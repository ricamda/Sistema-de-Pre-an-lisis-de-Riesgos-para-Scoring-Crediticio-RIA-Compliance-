# rules.py
from typing import Dict, Any, List

def classify_risk_from_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entrada: campos del formulario (keys: sector, tipo_decision, datos_sensibles, impacto, finalidad, ...).
    Salida: dict {nivel_riesgo, score, justificacion, applicable_articles}
    """
    sector = str(inputs.get("sector", "")).strip().lower()
    tipo = str(inputs.get("tipo_decision", "")).strip().lower()
    datos_sens = str(inputs.get("datos_sensibles", "no")).strip().lower()
    impacto = str(inputs.get("impacto", "medio")).strip().lower()
    finalidad = str(inputs.get("finalidad", "")).strip().lower()

    score = 0
    reasons = []

    # reglas ejemplo
    if sector in ["financiero", "salud"]:
        score += 3
        reasons.append(f"Sector {sector} (alto impacto regulatorio).")
    if datos_sens in ["si", "sí", "true", "1"]:
        score += 3
        reasons.append("Procesa datos sensibles.")
    if "scoring" in finalidad or "credito" in finalidad or "prestamo" in finalidad:
        score += 2
        reasons.append("Finalidad: scoring/decisión crítica.")
    if tipo in ["totalmente automatizada", "automatizada"]:
        score += 2
        reasons.append("Decisión totalmente automatizada (sin supervisión humana efectiva).")
    if impacto in ["alto", "alto riesgo"]:
        score += 2
        reasons.append("Impacto de errores alto.")

    # umbrales
    if score >= 6:
        nivel = "Alto Riesgo"
    elif score >= 3:
        nivel = "Riesgo Limitado"
    else:
        nivel = "Riesgo Mínimo"

    justificante = " ".join(reasons) if reasons else "Basado en entradas: sin factores de riesgo relevantes detectados automáticamente."
    # articles placeholder: idealmente extraer de docs/normativa
    applicable_articles: List[str] = []

    # heurística: si alto, añadir referencia a RIA (usuario debe revisar)
    if nivel == "Alto Riesgo":
        applicable_articles.append("RIA - Artículo X (evaluación de alto riesgo) — revisar pdf fuente.")
    return {
        "nivel_riesgo": nivel,
        "score": score,
        "justificacion": justificante,
        "applicable_articles": applicable_articles
    }