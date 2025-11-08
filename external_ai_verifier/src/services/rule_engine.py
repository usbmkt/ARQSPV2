#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV18 Enhanced v18.0 - External Rule Engine
Motor de regras para o módulo externo de verificação por IA
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class ExternalRuleEngine:
    """Motor de regras externo independente"""
    
    def __init__(self, config: Dict[str, Any]):
        """Inicializa o motor de regras"""
        self.rules = config.get('rules', [])
        
        # Ensure we have default rules if none provided
        if not self.rules:
            self.rules = self._get_default_rules()
        
        logger.info(f"✅ External Rule Engine inicializado com {len(self.rules)} regras")
        self._log_rules()
    
    def _get_default_rules(self) -> List[Dict[str, Any]]:
        """
        ✅ CORRIGIDO: Regras padrão que RESPEITAM recomendações do LLM
        
        MUDANÇA CRÍTICA: Adicionadas regras prioritárias para REVISÃO_MANUAL
        conforme identificado no documento de melhorias
        """
        return [
            # ✅ PRIORIDADE MÁXIMA: Recomendações do LLM (sempre respeitadas)
            {
                "name": "llm_rejection_high_priority",
                "condition": "llm_recommendation == 'REJEITAR'",
                "action": {
                    "status": "rejected",
                    "reason": "Rejeitado por recomendação LLM de alta confiança",
                    "confidence_adjustment": -0.2
                },
                "priority": 1
            },
            {
                "name": "llm_manual_review_required",
                "condition": "llm_recommendation == 'REVISÃO_MANUAL'",
                "action": {
                    "status": "manual_review_required",
                    "reason": "Revisão manual requerida por recomendação LLM",
                    "confidence_adjustment": 0.0
                },
                "priority": 1
            },
            {
                "name": "llm_approval_high_confidence",
                "condition": "llm_recommendation == 'APROVAR'",
                "action": {
                    "status": "approved",
                    "reason": "Aprovado por recomendação LLM de alta confiança",
                    "confidence_adjustment": 0.1
                },
                "priority": 2
            },
            # ✅ PRIORIDADE ALTA: Regras de segurança críticas
            {
                "name": "high_risk_bias_rejection",
                "condition": "overall_risk >= 0.7",
                "action": {
                    "status": "rejected",
                    "reason": "Alto risco de viés/desinformação detectado",
                    "confidence_adjustment": -0.3
                },
                "priority": 3
            },
            {
                "name": "very_low_confidence_rejection",
                "condition": "overall_confidence <= 0.25",
                "action": {
                    "status": "rejected",
                    "reason": "Confiança extremamente baixa",
                    "confidence_adjustment": -0.2
                },
                "priority": 4
            },
            # ✅ PRIORIDADE MÉDIA: Regras de aprovação/rejeição padrão
            {
                "name": "high_confidence_approval",
                "condition": "overall_confidence >= 0.85",
                "action": {
                    "status": "approved",
                    "reason": "Alta confiança no conteúdo",
                    "confidence_adjustment": 0.1
                },
                "priority": 5
            },
            {
                "name": "low_confidence_manual_review",
                "condition": "overall_confidence <= 0.45",
                "action": {
                    "status": "manual_review_required",
                    "reason": "Confiança baixa - revisão manual recomendada",
                    "confidence_adjustment": 0.0
                },
                "priority": 6
            },
            # ✅ PRIORIDADE BAIXA: Regras de fallback
            {
                "name": "medium_risk_manual_review",
                "condition": "overall_risk >= 0.4",
                "action": {
                    "status": "manual_review_required",
                    "reason": "Risco médio detectado - revisão manual recomendada",
                    "confidence_adjustment": -0.1
                },
                "priority": 7
            }
        ]
    
    def apply_rules(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        ✅ CORRIGIDO: Aplica regras aos dados do item com PRIORIDADE para recomendações LLM
        
        MUDANÇA CRÍTICA: Agora ordena regras por prioridade e SEMPRE respeita
        recomendações de REVISÃO_MANUAL conforme documento de melhorias
        
        Args:
            item_data (Dict[str, Any]): Dados do item para análise
            
        Returns:
            Dict[str, Any]: Resultado da aplicação das regras
        """
        try:
            # Initialize decision result
            decision = {
                "status": "approved",  # Default to approved if no rules trigger
                "reason": "Nenhuma regra específica ativada",
                "confidence_adjustment": 0.0,
                "triggered_rules": []
            }
            
            # Extract relevant scores from item_data
            validation_scores = item_data.get("validation_scores", {})
            sentiment_analysis = item_data.get("sentiment_analysis", {})
            bias_analysis = item_data.get("bias_disinformation_analysis", {})
            llm_analysis = item_data.get("llm_reasoning_analysis", {})
            
            # Calculate overall confidence and risk
            overall_confidence = self._calculate_overall_confidence(validation_scores, sentiment_analysis, bias_analysis, llm_analysis)
            overall_risk = bias_analysis.get("overall_risk", 0.0)
            llm_recommendation = llm_analysis.get("llm_recommendation", "REVISÃO_MANUAL")
            llm_confidence = llm_analysis.get("llm_confidence", 0.5)
            
            # ✅ CORREÇÃO CRÍTICA: Ordena regras por prioridade (menor número = maior prioridade)
            sorted_rules = sorted(self.rules, key=lambda r: r.get("priority", 999))
            
            logger.debug(f"🔍 Aplicando regras: confidence={overall_confidence:.3f}, risk={overall_risk:.3f}, llm_rec='{llm_recommendation}', llm_conf={llm_confidence:.3f}")
            
            # ✅ PRIORIDADE ABSOLUTA: Verifica recomendações LLM primeiro
            if llm_recommendation == 'REJEITAR' and llm_confidence >= 0.6:
                return {
                    "status": "rejected",
                    "reason": f"Rejeitado por análise LLM (confiança: {llm_confidence:.3f})",
                    "confidence_adjustment": -0.2,
                    "triggered_rules": ["llm_rejection_priority"],
                    "llm_override": True
                }
            elif llm_recommendation == 'REVISÃO_MANUAL':
                return {
                    "status": "manual_review_required",
                    "reason": f"Revisão manual requerida por LLM (confiança: {llm_confidence:.3f})",
                    "confidence_adjustment": 0.0,
                    "triggered_rules": ["llm_manual_review_priority"],
                    "llm_override": True
                }
            elif llm_recommendation == 'APROVAR' and llm_confidence >= 0.8:
                return {
                    "status": "approved",
                    "reason": f"Aprovado por análise LLM (confiança: {llm_confidence:.3f})",
                    "confidence_adjustment": 0.1,
                    "triggered_rules": ["llm_approval_priority"],
                    "llm_override": True
                }
            
            # Apply rules in priority order
            for rule in sorted_rules:
                if self._evaluate_condition(rule, overall_confidence, overall_risk, llm_recommendation, item_data):
                    rule_name = rule.get("name", "unknown_rule")
                    action = rule.get("action", {})
                    priority = rule.get("priority", 999)
                    
                    # Update decision
                    decision["status"] = action.get("status", "approved")
                    decision["reason"] = action.get("reason", f"Regra '{rule_name}' ativada")
                    decision["confidence_adjustment"] = action.get("confidence_adjustment", 0.0)
                    decision["triggered_rules"].append(rule_name)
                    decision["rule_priority"] = priority
                    
                    logger.debug(f"✅ Regra '{rule_name}' (prioridade {priority}) ativada: {decision['status']} - {decision['reason']}")
                    
                    # Stop at first matching rule (rules are ordered by priority)
                    break
            
            # ✅ Log da decisão final para debugging
            logger.debug(f"🎯 Decisão final: {decision['status']} - {decision['reason']}")
            
            return decision
            
        except Exception as e:
            logger.error(f"❌ Erro ao aplicar regras: {e}")
            return {
                "status": "rejected",  # Fail safe - reject on error
                "reason": f"Erro no processamento de regras: {str(e)}",
                "confidence_adjustment": -0.3,
                "triggered_rules": ["error_fallback"],
                "error": True
            }
    
    def _evaluate_condition(self, rule: Dict[str, Any], overall_confidence: float, overall_risk: float, llm_recommendation: str, item_data: Dict[str, Any]) -> bool:
        """
        Avalia se a condição de uma regra é atendida
        
        Args:
            rule: Regra para avaliar
            overall_confidence: Confiança geral calculada
            overall_risk: Risco geral calculado
            llm_recommendation: Recomendação do LLM
            item_data: Dados completos do item
            
        Returns:
            bool: True se a condição for atendida
        """
        try:
            condition = rule.get("condition", "")
            
            if not condition:
                return False
            
            # Simple condition evaluation
            # Replace variables in condition string
            condition = condition.replace("overall_confidence", str(overall_confidence))
            condition = condition.replace("overall_risk", str(overall_risk))
            condition = condition.replace("llm_recommendation", f"'{llm_recommendation}'")
            
            # Evaluate mathematical expressions
            if any(op in condition for op in [">=", "<=", "==", ">", "<", "!="]):
                try:
                    # Safe evaluation of simple mathematical conditions
                    return self._safe_eval_condition(condition)
                except:
                    logger.warning(f"Erro ao avaliar condição: {condition}")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"Erro na avaliação da condição: {e}")
            return False
    
    def _safe_eval_condition(self, condition: str) -> bool:
        """
        Avalia condições matemáticas simples de forma segura
        
        Args:
            condition (str): Condição para avaliar
            
        Returns:
            bool: Resultado da avaliação
        """
        try:
            # Only allow safe mathematical operations and comparisons
            allowed_chars = set("0123456789.><=! '\"ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz")
            
            if not all(c in allowed_chars for c in condition):
                logger.warning(f"Caracteres não permitidos na condição: {condition}")
                return False
            
            # Simple string replacements for evaluation
            if ">=" in condition:
                parts = condition.split(">=")
                if len(parts) == 2:
                    try:
                        left = float(parts[0].strip())
                        right = float(parts[1].strip())
                        return left >= right
                    except ValueError:
                        # Handle string comparisons
                        return parts[0].strip() == parts[1].strip()
                        
            elif "<=" in condition:
                parts = condition.split("<=")
                if len(parts) == 2:
                    left = float(parts[0].strip())
                    right = float(parts[1].strip())
                    return left <= right
                    
            elif "==" in condition:
                parts = condition.split("==")
                if len(parts) == 2:
                    left = parts[0].strip().strip("'\"")
                    right = parts[1].strip().strip("'\"")
                    return left == right
                    
            elif ">" in condition:
                parts = condition.split(">")
                if len(parts) == 2:
                    left = float(parts[0].strip())
                    right = float(parts[1].strip())
                    return left > right
                    
            elif "<" in condition:
                parts = condition.split("<")
                if len(parts) == 2:
                    left = float(parts[0].strip())
                    right = float(parts[1].strip())
                    return left < right
            
            return False
            
        except Exception as e:
            logger.error(f"Erro na avaliação segura da condição: {e}")
            return False
    
    def _calculate_overall_confidence(self, validation_scores: Dict[str, Any], sentiment_analysis: Dict[str, Any], bias_analysis: Dict[str, Any], llm_analysis: Dict[str, Any]) -> float:
        """Calcula confiança geral baseada em todas as análises"""
        try:
            # Start with base validation confidence
            base_confidence = validation_scores.get("overall_confidence", 0.5)
            
            # Adjust based on sentiment analysis
            sentiment_confidence = sentiment_analysis.get("confidence", 0.5)
            sentiment_weight = 0.2
            
            # Adjust based on bias analysis (lower bias risk = higher confidence)
            bias_confidence = 1.0 - bias_analysis.get("overall_risk", 0.5)  # Invert risk to confidence
            bias_weight = 0.3
            
            # Adjust based on LLM analysis
            llm_confidence = llm_analysis.get("llm_confidence", 0.5)
            llm_weight = 0.4
            
            # Weighted combination
            overall_confidence = (
                base_confidence * (1.0 - sentiment_weight - bias_weight - llm_weight) +
                sentiment_confidence * sentiment_weight +
                bias_confidence * bias_weight +
                llm_confidence * llm_weight
            )
            
            return min(max(overall_confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"Erro no cálculo de confiança geral: {e}")
            return 0.5
    
    def _log_rules(self):
        """Log das regras configuradas"""
        logger.debug("Regras configuradas:")
        for i, rule in enumerate(self.rules):
            logger.debug(f"  {i+1}. {rule.get('name', 'sem_nome')}: {rule.get('condition', 'sem_condição')}")
    
    def add_rule(self, rule: Dict[str, Any]):
        """
        Adiciona uma nova regra
        
        Args:
            rule (Dict[str, Any]): Nova regra para adicionar
        """
        if self._validate_rule(rule):
            self.rules.append(rule)
            logger.info(f"Nova regra adicionada: {rule.get('name', 'sem_nome')}")
        else:
            logger.warning(f"Regra inválida rejeitada: {rule}")
    
    def _validate_rule(self, rule: Dict[str, Any]) -> bool:
        """Valida se uma regra está bem formada"""
        return (
            isinstance(rule, dict) and
            "condition" in rule and
            "action" in rule and
            isinstance(rule["action"], dict)
        )