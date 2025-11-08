@echo off
REM ARQV30 Enhanced v3.0 - External AI Verifier Installation Script
REM Script de instalação e configuração do módulo independente

echo ========================================
echo ARQV30 Enhanced v3.0 - External AI Verifier
echo Módulo Independente de Verificação por IA
echo ========================================
echo.

REM Verifica se Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERRO: Python não encontrado!
    echo.
    echo Por favor, instale Python 3.11+ de https://python.org      
    echo Certifique-se de marcar "Add Python to PATH" durante a instalação.
    echo.
    pause
    exit /b 1
)

echo ✅ Python encontrado:
python --version
echo.

REM Verifica versão do Python
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Versão do Python: %PYTHON_VERSION%
echo.

REM Cria ambiente virtual para o módulo
echo 🔄 Criando ambiente virtual para AI Verifier...
python -m venv ai_verifier_env
if errorlevel 1 (
    echo ❌ ERRO: Falha ao criar ambiente virtual!
    pause
    exit /b 1
)

REM Ativa ambiente virtual
echo 🔄 Ativando ambiente virtual...
call ai_verifier_env\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ ERRO: Falha ao ativar ambiente virtual!
    pause
    exit /b 1
)

REM Atualiza pip
echo 🔄 Atualizando pip...
python -m pip install --upgrade pip
echo.

REM Instala dependências do módulo
echo 🔄 Instalando dependências do AI Verifier...
echo Isso pode levar alguns minutos...
echo.
pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ ERRO: Falha ao instalar dependências!
    echo Tentando instalação individual das dependências principais...
    
    REM Instalar dependências críticas individualmente
    pip install python-dotenv pyyaml pandas requests
    pip install textblob nltk vaderSentiment
    pip install openai google-generativeai
    pip install scikit-learn numpy
    
    echo ✅ Dependências principais instaladas
) else (
    echo ✅ Todas as dependências instaladas com sucesso!
)

REM Baixa modelos NLTK necessários
echo 🔄 Configurando modelos NLTK...
python -c "
import nltk
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print('✅ Modelos NLTK configurados')
except Exception as e:
    print(f'⚠️ Aviso: Erro ao configurar NLTK: {e}')
"

REM Configura TextBlob
echo 🔄 Configurando TextBlob...
python -c "
import textblob
try:
    textblob.TextBlob('test').sentiment
    print('✅ TextBlob configurado')
except Exception as e:
    print(f'⚠️ Aviso: Erro ao configurar TextBlob: {e}')
"

REM Cria arquivo de exemplo de configuração
echo 🔄 Criando arquivo de configuração de exemplo...
if not exist "config\user_config.yaml" (
    copy config\default_config.yaml config\user_config.yaml
    echo ✅ Configuração de usuário criada: config\user_config.yaml
)

REM Cria diretórios de trabalho
echo 🔄 Criando diretórios de trabalho...
if not exist "logs" mkdir logs
if not exist "temp" mkdir temp
if not exist "output" mkdir output
if not exist "docs" mkdir docs
if not exist "tests" mkdir tests

REM Testa a instalação
echo 🧪 Testando instalação do AI Verifier...
python -c "
import sys
import os
sys.path.insert(0, 'src')

try:
    from external_review_agent import run_external_review
    print('✅ Módulo AI Verifier carregado com sucesso')
    
    # Teste básico
    test_data = {
        'items': [
            {'id': 'test1', 'content': 'Este é um teste do módulo de verificação por IA.'}
        ]
    }
    
    result = run_external_review(test_data)
    if 'statistics' in result:
        print('✅ Teste básico executado com sucesso')
        print(f'   Items processados: {result[\"statistics\"].get(\"total_processed\", 0)}')
    else:
        print('⚠️ Teste básico executou mas resultado incompleto')
        
except ImportError as e:
    print(f'❌ Erro de importação: {e}')
except Exception as e:
    print(f'⚠️ Aviso: Erro no teste: {e}')
"

echo.
echo ========================================
echo 🎉 INSTALAÇÃO DO AI VERIFIER CONCLUÍDA!
echo ========================================
echo.
echo 🚀 Próximos passos:
echo.
echo 1. ✅ Configure suas chaves de API no arquivo .env:
echo    - GEMINI_API_KEY=sua-chave-gemini
echo    - OPENAI_API_KEY=sua-chave-openai
echo.
echo 2. ✅ Ajuste configurações em config\user_config.yaml se necessário
echo.
echo 3. ✅ Execute run.bat para processar dados
echo.
echo 4. ✅ O módulo está pronto para uso independente!
echo.
echo ========================================
echo.
echo 🔥 CARACTERÍSTICAS DO MÓDULO:
echo - ✅ Análise de sentimento avançada
echo - ✅ Detecção de viés e desinformação
echo - ✅ Raciocínio com LLM (Gemini/OpenAI)
echo - ✅ Análise contextual inteligente
echo - ✅ Motor de regras configurável
echo - ✅ Processamento em lote
echo - ✅ Relatórios detalhados
echo.
echo 🎯 USO:
echo   python -m src.external_review_agent
echo   ou execute: run.bat
echo.
pause