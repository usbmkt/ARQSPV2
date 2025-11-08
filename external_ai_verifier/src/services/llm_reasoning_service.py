#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV18 Enhanced v18.0 - External LLM Reasoning Service
Serviço de raciocínio com LLMs para análise aprofundada
COM ROTAÇÃO DE API KEYS E PROVIDERS (Gemini, OpenAI, Fireworks, Groq)
"""

import logging
import os
import time
import random
from typing import Dict, Any, Optional, List
from pathlib import Path

# ✅ CORREÇÃO: Caminho correto para importar local_model_manager
try:
    import sys
    # Obtém o diretório do arquivo atual
    current_dir = Path(__file__).resolve().parent
    # Caminho correto: external_ai_verifier/src/services -> ../../src/services
    services_dir = current_dir.parent.parent.parent / "src" / "services"
    
    # Adiciona ao path se não estiver lá
    services_path = str(services_dir)
    if services_path not in sys.path:
        sys.path.insert(0, services_path)
    
    from local_model_manager import LocalModelManager, local_model_manager
    LOCAL_MODEL_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info(f"✅ Local Model Manager importado de: {services_path}")
except ImportError as e:
    LOCAL_MODEL_AVAILABLE = False
    local_model_manager = None
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Local Model Manager não disponível: {e}")

# ✅ NOVO: Importa sistema de tratamento de erros
try:
    from error_handler import error_handler, ErrorType, ErrorSeverity
    ERROR_HANDLER_AVAILABLE = True
except ImportError:
    error_handler = None
    ErrorType = None
    ErrorSeverity = None
    ERROR_HANDLER_AVAILABLE = False

# Try to import LLM clients, fallback gracefully
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class LLMProviderManager:
    """Gerenciador inteligente de múltiplos providers de LLM integrado"""
    
    def __init__(self, config: Dict[str, Any]):
        """Inicializa o gerenciador de providers"""
        self.config = config.get('llm_reasoning', {})
        self.provider_priority = self.config.get('provider_priority', ['gemini', 'fireworks', 'groq', 'openai'])
        self.current_provider = self.config.get('provider', 'gemini')
        self.current_provider_index = 0
        
        self.models = self.config.get('models', {
            'gemini': 'gemini-2.0-flash-exp',
            'fireworks': 'fireworks/deepseek-v3',
            'groq': 'qwen/qwen3-32b',
            'openai': 'gpt-3.5-turbo'
        })
        
        self.provider_status = {}
        self.provider_cooldown_until = {}
        self.provider_failure_count = {}
        self.provider_success_count = {}
        self.auto_switch = self.config.get('auto_switch_provider', True)
        self.cooldown_minutes = self.config.get('provider_cooldown_minutes', 30)
        self.fallback_behavior = self.config.get('fallback_behavior', {})
        
        for provider in self.provider_priority:
            self.provider_status[provider] = 'available'
            self.provider_failure_count[provider] = 0
            self.provider_success_count[provider] = 0
            self.provider_cooldown_until[provider] = 0
        
        logger.info(f"🔄 Provider Manager: {' → '.join(self.provider_priority)}")
    
    def get_current_provider(self) -> str:
        return self.current_provider
    
    def get_current_model(self) -> str:
        return self.models.get(self.current_provider, 'gemini-2.0-flash-exp')
    
    def should_switch_provider(self, error_type: str) -> bool:
        if not self.auto_switch:
            return False
        behavior = self.fallback_behavior.get(f'on_{error_type}', 'switch_provider')
        return behavior == 'switch_provider'
    
    def switch_to_next_provider(self) -> Optional[str]:
        current_time = time.time()
        attempts = 0
        max_attempts = len(self.provider_priority)
        
        while attempts < max_attempts:
            self.current_provider_index = (self.current_provider_index + 1) % len(self.provider_priority)
            next_provider = self.provider_priority[self.current_provider_index]
            
            if self._is_provider_available(next_provider, current_time):
                old_provider = self.current_provider
                self.current_provider = next_provider
                logger.info(f"🔄 Provider: {old_provider} → {next_provider}")
                return next_provider
            attempts += 1
        
        logger.error("❌ Nenhum provider disponível")
        return None
    
    def _is_provider_available(self, provider: str, current_time: float) -> bool:
        cooldown_until = self.provider_cooldown_until.get(provider, 0)
        if current_time < cooldown_until:
            return False
        status = self.provider_status.get(provider, 'available')
        return status != 'disabled'
    
    def mark_provider_failure(self, provider: str, error_type: str = 'quota'):
        self.provider_failure_count[provider] = self.provider_failure_count.get(provider, 0) + 1
        failures = self.provider_failure_count[provider]
        
        if error_type in ['quota', 'forbidden', 'auth']:
            cooldown_seconds = self.cooldown_minutes * 60
            self.provider_cooldown_until[provider] = time.time() + cooldown_seconds
            logger.warning(f"🕒 {provider} cooldown {cooldown_seconds/60:.0f}min (erro: {error_type})")
        elif error_type == 'rate_limit':
            self.provider_cooldown_until[provider] = time.time() + 300
        else:
            self.provider_cooldown_until[provider] = time.time() + 60
        
        if failures >= 5:
            self.provider_status[provider] = 'disabled'
            logger.error(f"❌ Provider {provider} desabilitado")
    
    def mark_provider_success(self, provider: str):
        self.provider_success_count[provider] = self.provider_success_count.get(provider, 0) + 1
        if provider in self.provider_cooldown_until:
            del self.provider_cooldown_until[provider]
        
        successes = self.provider_success_count[provider]
        failures = self.provider_failure_count.get(provider, 0)
        if failures > 0 and successes >= 3:
            self.provider_failure_count[provider] = 0
            self.provider_status[provider] = 'available'


class ExternalLLMReasoningService:
    """Serviço de raciocínio com LLMs externo independente com rotação de API keys e providers"""
    
    def __init__(self, config: Dict[str, Any]):
        """Inicializa o serviço de LLM"""
        self.config = config.get('llm_reasoning', {})
        self.enabled = self.config.get('enabled', True)
        self.provider = self.config.get('provider', 'gemini').lower()
        self.model = self.config.get('model', 'gemini-2.0-flash-exp')
        self.max_tokens = self.config.get('max_tokens', 1000)
        self.temperature = self.config.get('temperature', 0.3)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        
        # ✅ Pausa entre requisições
        self.request_delay = self.config.get('request_delay', 10.0)
        self.last_request_time = 0
        
        # Parâmetros para retry
        self.max_retries_on_quota_error = self.config.get('max_retries_on_quota_error', 3)
        self.base_retry_delay = self.config.get('base_retry_delay', 1.0)
        self.max_retry_delay = self.config.get('max_retry_delay', 60.0)
        
        # ✅ Sistema de rotação de API keys
        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        self.key_failure_count = {}
        self.key_success_count = {}
        self.key_last_used = {}
        self.key_cooldown_until = {}
        self.max_key_failures = 3
        self.key_cooldown_seconds = 60
        self.key_recovery_threshold = 5
        
        # ✅ Configurações por provider
        self.provider_configs = {
            'fireworks': {
                'base_url': 'https://api.fireworks.ai/inference/v1/chat/completions',
                'default_model': 'fireworks/deepseek-v3'
            },
            'groq': {
                'base_url': 'https://api.groq.com/openai/v1/chat/completions',
                'default_model': 'qwen/qwen3-32b'
            }
        }
        
        # ✅ Gerenciador de providers
        self.provider_manager = None
        try:
            self.provider_manager = LLMProviderManager(config)
            self.provider = self.provider_manager.get_current_provider()
            self.model = self.provider_manager.get_current_model()
            logger.info(f"✅ Provider Manager ativo: {self.provider}")
        except Exception as e:
            logger.warning(f"⚠️ Provider Manager: {e}")
        
        self.client = None
        self._initialize_llm_client()
        
        logger.info(f"✅ LLM Service (Provider: {self.provider}, Keys: {len(self.api_keys)}, Delay: {self.request_delay}s)")
    
    def _load_api_keys(self) -> List[str]:
        """Carrega todas as API keys disponíveis do ambiente"""
        keys = []
        
        if self.provider == 'gemini':
            base_key = os.getenv('GEMINI_API_KEY')
            if base_key:
                keys.append(base_key)
            index = 1
            while True:
                key = os.getenv(f'GEMINI_API_KEY_{index}')
                if key:
                    keys.append(key)
                    index += 1
                else:
                    break
                    
        elif self.provider == 'openai':
            base_key = os.getenv('OPENAI_API_KEY')
            if base_key:
                keys.append(base_key)
            index = 1
            while True:
                key = os.getenv(f'OPENAI_API_KEY_{index}')
                if key:
                    keys.append(key)
                    index += 1
                else:
                    break
        
        elif self.provider == 'fireworks':
            base_key = os.getenv('FIREWORKS_API_KEY')
            if base_key:
                keys.append(base_key)
            index = 1
            while True:
                key = os.getenv(f'FIREWORKS_API_KEY_{index}')
                if key:
                    keys.append(key)
                    index += 1
                else:
                    break
        
        elif self.provider == 'groq':
            base_key = os.getenv('GROQ_API_KEY')
            if base_key:
                keys.append(base_key)
            index = 1
            while True:
                key = os.getenv(f'GROQ_API_KEY_{index}')
                if key:
                    keys.append(key)
                    index += 1
                else:
                    break
        
        keys = list(dict.fromkeys(keys))
        
        if keys:
            logger.info(f"🔑 {len(keys)} key(s) para {self.provider}")
        else:
            logger.warning(f"⚠️ Nenhuma key para {self.provider}")
        
        return keys
    
    def _get_next_api_key(self) -> Optional[str]:
        """Obtém a próxima API key disponível"""
        if not self.api_keys:
            return None
        
        current_time = time.time()
        available_keys = []
        
        for i, key in enumerate(self.api_keys):
            failures = self.key_failure_count.get(i, 0)
            successes = self.key_success_count.get(i, 0)
            cooldown_until = self.key_cooldown_until.get(i, 0)
            
            # Verifica cooldown
            if current_time < cooldown_until:
                continue
            
            # Lógica especial para chaves com muitos erros 403 (forbidden)
            if failures >= 3:
                # Se teve muitos erros 403, evita usar por mais tempo
                last_failure_time = self.key_last_used.get(i, 0)
                if current_time - last_failure_time < 3600:  # 1 hora
                    logger.debug(f"🔑 Key #{i+1} evitada por histórico de erros 403")
                    continue
            
            # Lógica geral de falhas
            if failures >= self.max_key_failures:
                if successes >= self.key_recovery_threshold:
                    self.key_failure_count[i] = 0
                    self.key_success_count[i] = 0
                    logger.info(f"🔄 Key #{i+1} recuperada (sucessos: {successes})")
                else:
                    continue
            
            # Calcula score penalizando falhas
            score = successes - failures
            if failures > 0:
                score -= failures * 0.5  # Penalidade adicional por falhas
            
            available_keys.append((i, key, score))
        
        if not available_keys:
            # Se não há chaves disponíveis, pode ser problema do provider
            total_keys = len(self.api_keys)
            forbidden_keys = sum(1 for i in range(total_keys) 
                               if self.key_failure_count.get(i, 0) >= 3)
            
            if forbidden_keys >= total_keys * 0.8:  # 80% das chaves com problema
                logger.error(f"🚨 {forbidden_keys}/{total_keys} chaves com problemas - provider {self.provider} pode estar inválido")
                if self.provider_manager:
                    self.provider_manager.mark_provider_failure(self.provider, 'forbidden')
            
            return None
        
        available_keys.sort(key=lambda x: x[2], reverse=True)
        best_key_index, selected_key, score = available_keys[0]
        
        self.current_key_index = best_key_index
        self.key_last_used[best_key_index] = current_time
        
        return selected_key
    
    def _mark_key_failure(self, is_quota_error: bool = False, is_forbidden: bool = False):
        """Marca falha na chave atual"""
        current_time = time.time()
        self.key_failure_count[self.current_key_index] = \
            self.key_failure_count.get(self.current_key_index, 0) + 1
        
        failures = self.key_failure_count[self.current_key_index]
        
        if is_forbidden:
            # Erro 403: API key inválida ou sem permissão - desabilita permanentemente
            cooldown_duration = self.key_cooldown_seconds * 60  # 1 hora
            self.key_cooldown_until[self.current_key_index] = current_time + cooldown_duration
            logger.error(f"🔑 Key #{self.current_key_index + 1} sem permissão - desabilitando por 1h")
            
            # Se muitas chaves falharam com 403, pode ser problema do provider
            forbidden_keys = sum(1 for i, count in self.key_failure_count.items() 
                                if count > 0 and i in self.key_cooldown_until 
                                and self.key_cooldown_until[i] > current_time)
            
            if forbidden_keys >= len(self.api_keys) * 0.7:  # 70% das chaves com problema
                logger.warning(f"⚠️ {forbidden_keys}/{len(self.api_keys)} chaves com erro 403 - problema no provider {self.provider}")
                if self.provider_manager:
                    self.provider_manager.mark_provider_failure(self.provider, 'forbidden')
        elif is_quota_error:
            cooldown_duration = self.key_cooldown_seconds * (2 ** min(failures - 1, 3))
            self.key_cooldown_until[self.current_key_index] = current_time + cooldown_duration
            logger.warning(f"⏸️ Key #{self.current_key_index + 1} quota (cooldown {cooldown_duration/60:.0f}min)")
        else:
            cooldown_duration = self.key_cooldown_seconds // 2
            self.key_cooldown_until[self.current_key_index] = current_time + cooldown_duration
        
        if failures >= self.max_key_failures:
            logger.error(f"❌ Key #{self.current_key_index + 1} desabilitada")
    
    def _mark_key_success(self):
        """Marca sucesso na chave atual"""
        self.key_success_count[self.current_key_index] = \
            self.key_success_count.get(self.current_key_index, 0) + 1
        
        if self.current_key_index in self.key_cooldown_until:
            del self.key_cooldown_until[self.current_key_index]
    
    def _reset_key_failure(self):
        """Reseta contador de falhas"""
        if self.current_key_index in self.key_failure_count:
            self.key_failure_count[self.current_key_index] = 0
        
        self._mark_key_success()
        
        # ✅ Marca sucesso do provider
        if self.provider_manager:
            self.provider_manager.mark_provider_success(self.provider)
    
    def _wait_for_request_delay(self):
        """Implementa pausa entre requisições"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.request_delay:
            wait_time = self.request_delay - time_since_last_request
            logger.info(f"⏳ Aguardando {wait_time:.1f}s...")
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def _initialize_llm_client(self):
        """Inicializa o cliente LLM"""
        try:
            if not self.api_keys:
                logger.warning(f"⚠️ Nenhuma key para {self.provider}")
                return
            
            if self.provider == 'gemini' and GEMINI_AVAILABLE:
                api_key = self._get_next_api_key()
                if api_key:
                    genai.configure(api_key=api_key)
                    self.client = genai.GenerativeModel(self.model)
                    logger.info(f"✅ Gemini: {self.model} (Key #{self.current_key_index + 1})")
                    
            elif self.provider == 'openai' and OPENAI_AVAILABLE:
                api_key = self._get_next_api_key()
                if api_key:
                    openai.api_key = api_key
                    self.client = openai
                    logger.info(f"✅ OpenAI: {self.model} (Key #{self.current_key_index + 1})")
            
            elif self.provider == 'fireworks' and REQUESTS_AVAILABLE:
                api_key = self._get_next_api_key()
                if api_key:
                    self.client = 'fireworks'
                    if not self.model or self.model == 'gemini-2.0-flash-exp':
                        self.model = self.provider_configs['fireworks']['default_model']
                    logger.info(f"✅ Fireworks: {self.model} (Key #{self.current_key_index + 1})")
            
            elif self.provider == 'groq' and REQUESTS_AVAILABLE:
                api_key = self._get_next_api_key()
                if api_key:
                    self.client = 'groq'
                    if not self.model or self.model == 'gemini-2.0-flash-exp':
                        self.model = self.provider_configs['groq']['default_model']
                    logger.info(f"✅ Groq: {self.model} (Key #{self.current_key_index + 1})")
            
            else:
                logger.warning(f"⚠️ Provider '{self.provider}' não disponível")
                
        except Exception as e:
            logger.error(f"Erro ao inicializar: {e}")
            self.client = None
    
    def _rotate_to_next_key(self) -> bool:
        """Rotaciona para próxima chave"""
        try:
            next_key = self._get_next_api_key()
            if not next_key:
                return False
            
            if self.provider == 'gemini' and GEMINI_AVAILABLE:
                genai.configure(api_key=next_key)
                self.client = genai.GenerativeModel(self.model)
                logger.info(f"🔄 Key #{self.current_key_index + 1}")
                return True
                
            elif self.provider == 'openai' and OPENAI_AVAILABLE:
                openai.api_key = next_key
                self.client = openai
                logger.info(f"🔄 Key #{self.current_key_index + 1}")
                return True
            
            elif self.provider in ['fireworks', 'groq'] and REQUESTS_AVAILABLE:
                self.client = self.provider
                logger.info(f"🔄 Key #{self.current_key_index + 1}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erro ao rotacionar: {e}")
            return False
    
    def _classify_error(self, error_message: str, status_code: int = None) -> Dict[str, Any]:
        """Classifica o tipo de erro para tratamento apropriado"""
        error_lower = error_message.lower()
        
        error_info = {
            'type': 'unknown',
            'should_retry_key': False,
            'should_switch_provider': False,
            'is_fatal': False
        }
        
        # Erro 403 - Forbidden (API key inválida ou sem permissão)
        if status_code == 403 or '403' in error_message:
            error_info.update({
                'type': 'forbidden',
                'should_retry_key': True,
                'should_switch_provider': True,
                'is_fatal': False
            })
        
        # Erro 429 - Rate Limit / Quota
        elif status_code == 429 or any(kw in error_lower for kw in ["429", "quota", "rate limit", "resource exhausted"]):
            error_info.update({
                'type': 'quota',
                'should_retry_key': True,
                'should_switch_provider': True,
                'is_fatal': False
            })
        
        # Erro 401 - Authentication
        elif status_code == 401 or '401' in error_message or 'unauthorized' in error_lower:
            error_info.update({
                'type': 'auth',
                'should_retry_key': True,
                'should_switch_provider': True,
                'is_fatal': False
            })
        
        # Erro 400 - Bad Request (geralmente não vale retry)
        elif status_code == 400 or '400' in error_message:
            error_info.update({
                'type': 'bad_request',
                'should_retry_key': False,
                'should_switch_provider': False,
                'is_fatal': True
            })
        
        # Erros de rede/timeout
        elif any(kw in error_lower for kw in ["timeout", "connection", "network"]):
            error_info.update({
                'type': 'network',
                'should_retry_key': False,
                'should_switch_provider': False,
                'is_fatal': False
            })
        
        return error_info
    
    def analyze_with_llm(self, text: str, context: str = "") -> Dict[str, Any]:
        """Analisa texto com LLM"""
        if not self.enabled or not self.client or not text or not text.strip():
            return self._get_default_result()
        
        try:
            self._wait_for_request_delay()
            prompt = self._create_analysis_prompt(text, context)
            
            # ✅ Usa método unificado com retry e rotação automática
            response = self._analyze_with_retry(prompt)
            
            analysis_result = self._parse_llm_response(response, text)
            self._reset_key_failure()
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Erro LLM final: {e}")
            
            # ✅ Tentar fallback para modelo local
            local_result = self._try_local_model_fallback(text, context)
            if local_result:
                logger.info("✅ Análise realizada via modelo local (fallback)")
                return local_result
            
            return self._get_default_result()
    
    def _analyze_with_retry(self, prompt: str) -> str:
        """Análise unificada com retry e rotação automática de provider"""
        last_exception = None
        base_delay = self.base_retry_delay
        provider_switched = False

        for attempt in range(self.max_retries_on_quota_error + 1):
            try:
                # ✅ Chama método específico do provider ativo
                if self.provider == 'gemini':
                    response = self.client.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            max_output_tokens=self.max_tokens,
                            temperature=self.temperature
                        )
                    )
                    self._mark_key_success()
                    return response.text
                    
                elif self.provider == 'fireworks':
                    result = self._analyze_with_fireworks(prompt)
                    self._mark_key_success()
                    return result
                    
                elif self.provider == 'groq':
                    result = self._analyze_with_groq(prompt)
                    self._mark_key_success()
                    return result
                    
                elif self.provider == 'openai':
                    result = self._analyze_with_openai(prompt)
                    self._mark_key_success()
                    return result
                    
                else:
                    raise Exception(f"Provider {self.provider} não suportado")
                
            except Exception as e:
                last_exception = e
                error_message = str(e).lower()
                
                # ✅ Extrai status code se for erro HTTP
                status_code = None
                if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                    status_code = e.response.status_code
                elif '403' in str(e):
                    status_code = 403
                elif '429' in str(e):
                    status_code = 429
                elif '401' in str(e):
                    status_code = 401
                
                # ✅ Classifica o erro
                error_info = self._classify_error(error_message, status_code)
                
                logger.warning(f"⚠️ Tentativa {attempt + 1}/{self.max_retries_on_quota_error + 1} falhou: {error_info['type']} - {str(e)[:100]}")
                
                # Se for erro fatal, não tenta mais
                if error_info['is_fatal']:
                    logger.error(f"❌ Erro fatal: {str(e)[:200]}")
                    raise
                
                # Marca falha apropriada
                is_forbidden = error_info['type'] == 'forbidden'
                is_quota = error_info['type'] == 'quota'
                self._mark_key_failure(is_quota_error=is_quota, is_forbidden=is_forbidden)
                
                # ✅ Tenta trocar provider se apropriado
                if error_info['should_switch_provider'] and not provider_switched:
                    if self.provider_manager and self.provider_manager.should_switch_provider(error_info['type']):
                        self.provider_manager.mark_provider_failure(self.provider, error_info['type'])
                        new_provider = self.provider_manager.switch_to_next_provider()
                        
                        if new_provider:
                            self.provider = new_provider
                            self.model = self.provider_manager.get_current_model()
                            self.api_keys = self._load_api_keys()
                            self._initialize_llm_client()
                            logger.info(f"✅ Novo provider ativo: {new_provider}")
                            provider_switched = True
                            time.sleep(2.0)
                            continue
                
                # ✅ Tenta próxima key do mesmo provider
                if error_info['should_retry_key'] and attempt < self.max_retries_on_quota_error:
                    if self._rotate_to_next_key():
                        delay = min(base_delay * (1.5 ** attempt), 5.0)
                        logger.info(f"⏳ Aguardando {delay:.1f}s antes de tentar próxima key...")
                        time.sleep(delay)
                        continue
                    else:
                        # Nenhuma key disponível, aguarda e reinicializa
                        delay = min(base_delay * (2 ** attempt), self.max_retry_delay)
                        logger.warning(f"⏸️ Nenhuma key disponível, aguardando {delay:.1f}s...")
                        time.sleep(delay)
                        self.current_key_index = 0
                        self._initialize_llm_client()
                        continue

        if last_exception:
            raise last_exception
        raise Exception("Falha após todas tentativas")
    
    def _analyze_with_openai(self, prompt: str) -> str:
        """Análise com OpenAI"""
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return response.choices[0].message.content
    
    def _analyze_with_fireworks(self, prompt: str) -> str:
        """Análise com Fireworks AI"""
        api_key = self.api_keys[self.current_key_index]
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': self.max_tokens,
            'temperature': self.temperature
        }
        response = requests.post(self.provider_configs['fireworks']['base_url'], 
                               json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    
    def _analyze_with_groq(self, prompt: str) -> str:
        """Análise com Groq"""
        api_key = self.api_keys[self.current_key_index]
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': self.max_tokens,
            'temperature': self.temperature
        }
        response = requests.post(self.provider_configs['groq']['base_url'], 
                               json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

    def _create_analysis_prompt(self, text: str, context: str = "") -> str:
        """Cria prompt para análise"""
        return f"""Analise o seguinte texto de forma crítica e objetiva:

TEXTO: "{text}"
{f'CONTEXTO: {context}' if context else ''}

Forneça análise estruturada:

1. QUALIDADE (0-10): Clareza, coerência, fundamentação
2. CONFIABILIDADE (0-10): Fontes, objetividade, credibilidade
3. VIÉS (0-10): Linguagem emotiva, unilateralidade
4. DESINFORMAÇÃO (0-10): Afirmações sem evidências, inconsistências

Formato da resposta:
QUALIDADE: [nota]/10 - [justificativa]
CONFIABILIDADE: [nota]/10 - [justificativa]
VIÉS: [nota]/10 - [justificativa]
DESINFORMAÇÃO: [nota]/10 - [justificativa]
RECOMENDAÇÃO: [APROVAR/REJEITAR/REVISÃO_MANUAL] - [razão]
CONFIANÇA_ANÁLISE: [0-100]% - [justificativa]"""
    
    def _parse_llm_response(self, response: str, original_text: str) -> Dict[str, Any]:
        """Parse da resposta LLM"""
        import re
        
        result = {
            'llm_response': response,
            'quality_score': 0.5,
            'reliability_score': 0.5,
            'bias_score': 0.5,
            'disinformation_score': 0.5,
            'llm_recommendation': 'REVISÃO_MANUAL',
            'llm_confidence': 0.5,
            'provider': self.provider,
            'model': self.model
        }
        
        patterns = {
            'quality_score': r'QUALIDADE:\s*([0-9]+(?:\.[0-9]+)?)',
            'reliability_score': r'CONFIABILIDADE:\s*([0-9]+(?:\.[0-9]+)?)',
            'bias_score': r'VIÉS:\s*([0-9]+(?:\.[0-9]+)?)',
            'disinformation_score': r'DESINFORMAÇÃO:\s*([0-9]+(?:\.[0-9]+)?)',
            'llm_recommendation': r'RECOMENDAÇÃO:\s*(APROVAR|REJEITAR|REVISÃO_MANUAL)',
            'llm_confidence': r'CONFIANÇA_ANÁLISE:\s*([0-9]+)%?'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                if key == 'llm_confidence':
                    result[key] = min(float(match.group(1)) / 100.0, 1.0)
                elif key == 'llm_recommendation':
                    result[key] = match.group(1).upper()
                else:
                    result[key] = min(float(match.group(1)) / 10.0, 1.0)
        
        return result
    
    def _get_default_result(self) -> Dict[str, Any]:
        """Resultado padrão"""
        return {
            'llm_response': 'LLM não disponível',
            'quality_score': 0.5,
            'reliability_score': 0.5,
            'bias_score': 0.5,
            'disinformation_score': 0.5,
            'llm_recommendation': 'REVISÃO_MANUAL',
            'llm_confidence': 0.1,
            'provider': self.provider,
            'model': self.model
        }
    
    def get_provider_status(self) -> Dict[str, Any]:
        """✅ Status de todos os providers"""
        if self.provider_manager:
            current_time = time.time()
            providers_info = []
            
            for provider in self.provider_manager.provider_priority:
                cooldown = self.provider_manager.provider_cooldown_until.get(provider, 0)
                status = 'cooldown' if current_time < cooldown else \
                        self.provider_manager.provider_status.get(provider, 'available')
                
                providers_info.append({
                    'provider': provider,
                    'status': status,
                    'failures': self.provider_manager.provider_failure_count.get(provider, 0),
                    'successes': self.provider_manager.provider_success_count.get(provider, 0),
                    'is_current': provider == self.provider
                })
            
            return {
                'current_provider': self.provider,
                'current_model': self.model,
                'providers': providers_info
            }
        return {"message": "Provider Manager não disponível"}
    
    def get_keys_status(self) -> Dict[str, Any]:
        """✅ Status de todas as API keys do provider atual"""
        current_time = time.time()
        keys_info = []
        
        for i in range(len(self.api_keys)):
            cooldown = self.key_cooldown_until.get(i, 0)
            status = 'cooldown' if current_time < cooldown else 'available'
            
            keys_info.append({
                'key_index': i + 1,
                'status': status,
                'failures': self.key_failure_count.get(i, 0),
                'successes': self.key_success_count.get(i, 0),
                'is_current': i == self.current_key_index,
                'cooldown_remaining': max(0, int(cooldown - current_time))
            })
        
        return {
            'provider': self.provider,
            'total_keys': len(self.api_keys),
            'current_key': self.current_key_index + 1,
            'keys': keys_info
        }
    
    def _try_local_model_fallback(self, text: str, context: str = "") -> Optional[Dict[str, Any]]:
        """Tenta usar modelo local como fallback quando APIs falham"""
        
        # Verifica se fallback está habilitado na configuração
        fallback_config = self.config.get('local_model_fallback', {})
        if not fallback_config.get('enabled', True):  # Habilitado por padrão
            logger.debug("Fallback para modelo local desabilitado na configuração")
            return self._get_emergency_fallback_result(text, "fallback_disabled")
        
        # Verifica se modelo local está disponível
        if not LOCAL_MODEL_AVAILABLE:
            logger.warning("⚠️ Local Model Manager não disponível - usando fallback de emergência")
            return self._get_emergency_fallback_result(text, "local_manager_unavailable")
        
        try:
            # Verificar se modelo local está carregado
            if not local_model_manager.is_model_loaded():
                logger.warning("⚠️ Modelo local não carregado para fallback do AI Verifier - tentando carregar...")
                
                # Tentar carregar modelo automaticamente
                try:
                    local_model_manager._auto_load_model()
                    
                    # Verificar novamente se carregou
                    if not local_model_manager.is_model_loaded():
                        logger.error("❌ Falha ao carregar modelo local para AI Verifier - usando fallback de emergência")
                        return self._get_emergency_fallback_result(text, "model_load_failed")
                        
                except Exception as load_error:
                    logger.error(f"❌ Erro ao carregar modelo local: {load_error} - usando fallback de emergência")
                    return self._get_emergency_fallback_result(text, f"model_load_error: {str(load_error)}")

            # Verificar se CUDA está sendo usado
            model_info = getattr(local_model_manager, 'model_info', {})
            gpu_layers = model_info.get('gpu_layers', 0)
            if gpu_layers > 0:
                logger.info(f"🚀 Usando modelo local com CUDA ({gpu_layers} GPU layers) para AI Verifier")
            else:
                logger.info("💻 Usando modelo local com CPU para AI Verifier")

            # Criar prompt específico para verificação
            prompt = self._create_local_verification_prompt(text, context)

            # Gerar análise com modelo local usando configurações otimizadas para CUDA
            logger.info("🔄 Usando modelo local CUDA como fallback...")
            
            try:
                response = local_model_manager.generate_text(
                    prompt=prompt,
                    max_tokens=4096,
                    temperature=0.6,  # Temperatura otimizada para consistência e qualidade
                    top_p=0.85,       # Top_p otimizado
                    top_k=50,         # Top_k para melhor qualidade
                    repeat_penalty=1.1  # Evitar repetições
                )

                if not response or len(response.strip()) < 50:
                    logger.warning("⚠️ Resposta do modelo local muito curta para análise - usando fallback de emergência")
                    return self._get_emergency_fallback_result(text, "response_too_short")

                # Processar resposta do modelo local
                return self._parse_local_model_response(response, text, fallback_config)
                
            except Exception as generation_error:
                logger.error(f"❌ Erro na geração com modelo local: {generation_error} - usando fallback de emergência")
                return self._get_emergency_fallback_result(text, f"generation_error: {str(generation_error)}")

        except Exception as e:
            logger.error(f"❌ Erro geral no fallback do modelo local: {e} - usando fallback de emergência")
            return self._get_emergency_fallback_result(text, f"general_error: {str(e)}")

    def _get_emergency_fallback_result(self, text: str, reason: str) -> Dict[str, Any]:
        """Fallback de emergência quando nem o modelo local funciona"""
        
        logger.warning(f"🚨 Usando fallback de emergência - razão: {reason}")
        
        # Análise básica baseada em regras simples
        text_lower = text.lower()
        
        # Palavras que indicam conteúdo problemático
        problematic_words = [
            'scam', 'fraude', 'golpe', 'enganar', 'mentira', 'falso',
            'garantido', '100%', 'milagre', 'impossível', 'secreto',
            'urgente', 'limitado', 'exclusivo', 'grátis', 'dinheiro fácil'
        ]
        
        # Contar palavras problemáticas
        problematic_count = sum(1 for word in problematic_words if word in text_lower)
        
        # Análise de comprimento e estrutura
        text_length = len(text)
        has_caps = any(c.isupper() for c in text)
        exclamation_count = text.count('!')
        
        # Calcular score de confiança baseado em regras
        confidence = 0.7  # Base
        
        # Penalizar por palavras problemáticas
        confidence -= (problematic_count * 0.1)
        
        # Penalizar por excesso de exclamações
        if exclamation_count > 3:
            confidence -= 0.1
            
        # Penalizar por texto muito curto ou muito longo
        if text_length < 50 or text_length > 5000:
            confidence -= 0.1
            
        # Garantir que confidence esteja entre 0.3 e 0.8
        confidence = max(0.3, min(0.8, confidence))
        
        # Determinar recomendação
        recommendation = "APROVADO" if confidence > 0.5 else "REJEITADO"
        
        # Criar análise baseada na pontuação
        if confidence > 0.6:
            analise = "Conteúdo aprovado por análise de emergência - sem indicadores críticos de problema"
            motivo = "Análise de emergência: conteúdo dentro dos padrões básicos"
        elif confidence > 0.4:
            analise = "Conteúdo com qualidade moderada - alguns indicadores de atenção detectados"
            motivo = "Análise de emergência: conteúdo requer revisão manual"
        else:
            analise = "Conteúdo rejeitado por análise de emergência - múltiplos indicadores problemáticos"
            motivo = "Análise de emergência: conteúdo com alto risco"
        
        return {
            'llm_confidence': confidence,
            'llm_recommendation': recommendation,
            'llm_response': f"{motivo}. {analise}",
            'quality_score': confidence,
            'reliability_score': 0.5,  # Baixa confiabilidade por ser fallback de emergência
            'bias_score': 0.5,
            'disinformation_score': 1.0 - confidence,
            'provider': 'emergency_fallback',
            'model': 'rule_based_analysis',
            'is_fallback': True,
            'is_emergency_fallback': True,
            'fallback_reason': reason,
            'fallback_reasoning': f"{motivo}. {analise}",
            'problematic_words_found': problematic_count,
            'analysis_method': 'rule_based_emergency'
        }
    
    def _create_local_verification_prompt(self, text: str, context: str = "") -> str:
        """Cria prompt aprimorado para verificação com modelo local CUDA"""
        
        prompt = f"""Você é um especialista em verificação de conteúdo para marketing digital. Analise o texto abaixo com rigor profissional e determine se deve ser APROVADO ou REJEITADO.

TEXTO PARA ANÁLISE:
{text[:2000]}

CONTEXTO ADICIONAL:
{context[:800] if context else 'Análise sem contexto específico - aplicar critérios gerais'}

CRITÉRIOS DE AVALIAÇÃO RIGOROSOS:

✅ APROVAR se o conteúdo atende TODOS os critérios:
   - Factual e baseado em evidências verificáveis
   - Relevante e útil para estratégias de marketing
   - Livre de desinformação, fake news ou dados fabricados
   - Linguagem profissional e equilibrada
   - Informações que agregam valor real ao negócio
   - Dados que podem ser utilizados para tomada de decisão

❌ REJEITAR se o conteúdo apresenta QUALQUER um dos problemas:
   - Informações claramente falsas ou não verificáveis
   - Linguagem extremamente tendenciosa ou sensacionalista
   - Dados fabricados, simulados ou exemplos genéricos
   - Conteúdo irrelevante para marketing ou negócios
   - Informações que podem induzir decisões equivocadas
   - Texto com viés excessivo ou propaganda enganosa

INSTRUÇÕES DE RESPOSTA:
Forneça sua análise no seguinte formato EXATO:

DECISÃO: [APROVADO ou REJEITADO]
CONFIANÇA: [0.0 a 1.0]
MOTIVO: [Explicação clara e objetiva da decisão]
ANÁLISE: [Análise detalhada dos pontos principais do conteúdo]

IMPORTANTE: 
- Seja rigoroso mas justo na avaliação
- Considere o valor prático do conteúdo para marketing
- Priorize a qualidade e veracidade das informações
- Em caso de dúvida, prefira APROVAR conteúdo útil
- Seja MODERADAMENTE PERMISSIVO (tendência para aprovar conteúdo útil)
- Foque em rejeitar apenas conteúdo claramente problemático

RESPONDA EXATAMENTE NO FORMATO:
DECISÃO: [APROVADO/REJEITADO]
CONFIANÇA: [0.0-1.0]
MOTIVO: [Explicação breve em uma linha]
ANÁLISE: [Análise detalhada em 2-3 frases]

Sua resposta:"""

        return prompt
    
    def _parse_local_model_response(self, response: str, original_text: str, fallback_config: Dict[str, Any]) -> Dict[str, Any]:
        """Processa resposta do modelo local e converte para formato esperado"""
        
        try:
            # Extrair informações da resposta
            lines = response.strip().split('\n')
            decisao = "APROVADO"  # Default para aprovação
            confianca = fallback_config.get('default_approval_bias', 0.6)
            motivo = "Análise via modelo local"
            analise = "Conteúdo analisado pelo modelo local de fallback"
            
            for line in lines:
                line = line.strip()
                if line.startswith('DECISÃO:') or line.startswith('DECISAO:'):
                    decisao = line.split(':', 1)[1].strip().upper()
                elif line.startswith('CONFIANÇA:') or line.startswith('CONFIANCA:'):
                    try:
                        confianca = float(line.split(':', 1)[1].strip())
                    except:
                        pass
                elif line.startswith('MOTIVO:'):
                    motivo = line.split(':', 1)[1].strip()
                elif line.startswith('ANÁLISE:') or line.startswith('ANALISE:'):
                    analise = line.split(':', 1)[1].strip()
            
            # Aplicar boost de confiança se configurado
            confidence_boost = fallback_config.get('confidence_boost', 0.1)
            if decisao == "APROVADO":
                confianca = min(1.0, confianca + confidence_boost)
            
            # Determinar recomendação final
            if decisao == "APROVADO":
                recommendation = "APROVADO"
            elif decisao == "REJEITADO":
                recommendation = "REJEITADO"
            else:
                # Default para aprovação moderada se ambíguo
                recommendation = "APROVADO"
                confianca = 0.5
                motivo = "Decisão ambígua - aprovado por política de fallback"
            
            logger.info(f"✅ Modelo local: {recommendation} (confiança: {confianca:.2f})")
            
            return {
                'llm_confidence': confianca,
                'llm_recommendation': recommendation,
                'llm_response': f"{motivo}. {analise}",
                'quality_score': 0.7 if recommendation == "APROVADO" else 0.3,
                'reliability_score': 0.7 if recommendation == "APROVADO" else 0.3,
                'bias_score': 0.5,
                'disinformation_score': 0.3 if recommendation == "APROVADO" else 0.7,
                'provider': 'local_model_fallback',
                'model': 'local_gguf',
                'is_fallback': True,
                'fallback_reasoning': f"{motivo}. {analise}"
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar resposta do modelo local: {e}")
            # Retornar resultado de aprovação padrão em caso de erro
            return {
                'llm_confidence': 0.6,
                'llm_recommendation': 'APROVADO',
                'llm_response': 'Erro no processamento - aprovado por política de fallback',
                'quality_score': 0.6,
                'reliability_score': 0.6,
                'bias_score': 0.5,
                'disinformation_score': 0.4,
                'provider': 'local_model_fallback_error',
                'model': 'local_gguf',
                'is_fallback': True,
                'error': str(e)
            }
