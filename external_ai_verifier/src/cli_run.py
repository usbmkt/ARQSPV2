#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV18 Enhanced v18.0 - External AI Verifier CLI
Command Line Interface para execução do módulo
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import time # Adicionado para time.time()

# Importar ExternalReviewAgent (assumindo que está em external_review_agent.py)
# É necessário que a classe ExternalReviewAgent esteja disponível neste escopo.
# Se estiver em outro arquivo, ajuste o import.
# Exemplo: from my_module.external_review_agent import ExternalReviewAgent
# Para este exemplo, vamos simular a classe se ela não for importada.

try:
    from external_review_agent import ExternalReviewAgent
except ImportError:
    print("WARN: external_review_agent.ExternalReviewAgent not found. Using a mock class.")
    class ExternalReviewAgent:
        async def analyze_content_batch(self, input_data):
            print("MOCK: analyze_content_batch called")
            return {
                'success': True,
                'total_items': len(input_data.get('items', [])),
                'statistics': {'approved_count': 0, 'rejected_count': 0, 'flagged_count': 0, 'average_confidence': 0.0},
                'items': [],
                'rejected_items': [],
                'metadata': {}
            }
        async def analyze_session_consolidacao(self, session_id):
            print(f"MOCK: analyze_session_consolidacao called with session_id: {session_id}")
            # Simula a busca do arquivo e a análise
            base_path = Path(__file__).parent.parent / "src" / "relatorios_intermediarios" / "workflow"
            consolidacao_file = base_path / f"session_{session_id}" / "consolidacao.json" # Assumindo nome e estrutura

            if not consolidacao_file.exists():
                return {'success': False, 'error': f"Arquivo de consolidação não encontrado para a sessão {session_id} em {consolidacao_file}", 'session_id': session_id}

            try:
                with open(consolidacao_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
            except Exception as e:
                return {'success': False, 'error': f"Erro ao ler arquivo de consolidação: {e}", 'session_id': session_id}

            # Simula a análise dos dados da sessão
            # Aqui você integraria a lógica real de análise dos dados de consolidacao.json
            # Por enquanto, retornamos uma estrutura genérica
            mock_result = {
                'success': True,
                'session_analysis': {
                    'session_id': session_id,
                    'consolidacao_file': str(consolidacao_file),
                    'items_processed_from_session': len(session_data.get('consolidacao', [])), # Exemplo
                },
                'total_items': len(session_data.get('consolidacao', [])), # Exemplo
                'statistics': {
                    'approved_count': len([item for item in session_data.get('consolidacao', []) if item.get('status') == 'APROVADO']), # Exemplo
                    'rejected_count': len([item for item in session_data.get('consolidacao', []) if item.get('status') == 'REJEITADO']), # Exemplo
                    'flagged_count': 0,
                    'average_confidence': 0.85 # Exemplo
                },
                'items': [],
                'rejected_items': [],
                'metadata': {'source': 'session_analysis'}
            }
            return mock_result


async def main():
    """Função principal do CLI"""
    try:
        print("🤖 External AI Verifier v18.0 - CLI")
        print("=" * 50)

        # Verifica argumentos da linha de comando
        import sys

        # Inicializa o agente
        agent = ExternalReviewAgent()

        # Verifica se foi passado um session_id como argumento
        if len(sys.argv) > 1 and sys.argv[1].startswith('session_'):
            session_id = sys.argv[1]
            print(f"📋 Modo: Análise de Sessão")
            print(f"🔍 Session ID: {session_id}")
            print("-" * 50)

            # Executa análise da sessão
            start_time = time.time()
            result = await agent.analyze_session_consolidacao(session_id)
            end_time = time.time()

        else:
            # Modo padrão: arquivo example_input.json
            print(f"📋 Modo: Arquivo de Entrada")

            # Verifica se arquivo de entrada existe
            input_file = "example_input.json"
            if not os.path.exists(input_file):
                print(f"❌ Arquivo {input_file} não encontrado!")
                print(f"💡 Uso alternativo: python cli_run.py session_XXXXXX")
                return

            # Carrega dados de entrada
            with open(input_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)

            print(f"📄 Carregado: {len(input_data.get('items', []))} itens para análise")
            print(f"🎯 Tópico: {input_data.get('context', {}).get('topic', 'N/A')}")
            print("🔍 Iniciando análise...")
            print("-" * 50)

            # Executa análise
            start_time = time.time()
            result = agent.analyze_content_batch(input_data)
            end_time = time.time()

        # Mostra resultados
        if result.get('success'):
            print(f"✅ Análise concluída em {end_time - start_time:.2f}s")
            print(f"📊 Total analisado: {result.get('total_items', 0)} itens")

            stats = result.get('statistics', {})
            print(f"✅ Aprovados: {stats.get('approved_count', 0)}")
            print(f"❌ Rejeitados: {stats.get('rejected_count', 0)}")
            print(f"⚠️ Flagged: {stats.get('flagged_count', 0)}")
            print(f"📈 Confiança média: {stats.get('average_confidence', 0):.2f}")

            # Nome do arquivo de saída baseado no tipo de análise
            if 'session_analysis' in result:
                session_id_result = result.get('session_analysis', {}).get('session_id', None)
                if session_id_result:
                    output_file = f"analysis_result_{session_id_result}_{int(time.time())}.json"
                else:
                    output_file = f"analysis_result_{int(time.time())}.json" # Fallback
            else:
                output_file = f"analysis_result_{int(time.time())}.json"

            # Salva resultado
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"💾 Resultado salvo: {output_file}")

        else:
            print(f"❌ Erro na análise: {result.get('error', 'Erro desconhecido')}")
            if 'session_id' in result:
                print(f"🔍 Session ID: {result['session_id']}")

    except Exception as e:
        print(f"❌ Erro crítico: {e}")
        import traceback
        traceback.print_exc()

# O código original do main() foi substituído pelo novo.
# Para que este script funcione como um CLI, ele precisa ser executado com um loop de evento assíncrono.
# Exemplo:
if __name__ == '__main__':
    import asyncio
    # Se o código original estava em um contexto síncrono,
    # a adaptação para async/await pode exigir mais ajustes.
    # Assumindo que 'run_external_review' (ou a nova lógica) é 'async'.
    asyncio.run(main())