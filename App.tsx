
import React, { useState, useCallback, useEffect } from 'react';
import { getMatchPrediction } from './services/geminiService';
import { MatchAnalysis, Player } from './types';
import AnalysisCard from './components/AnalysisCard';
import { ShieldIcon, BarChartIcon, UsersIcon, TacticIcon, AlertTriangleIcon, CheckCircleIcon, InfoIcon } from './components/icons';

const CorinthiansLogo = () => (
    <img src="https://logodownload.org/wp-content/uploads/2015/05/corinthians-logo-4.png" alt="Corinthians Logo" className="h-12 w-12 object-contain" />
);

const loadingMessages = [
    "Inicializando motor WebSailor V2...",
    "Consultando fontes de dados esportivos (SofaScore, FBref)...",
    "Buscando histórico de confrontos diretos...",
    "Analisando momento da equipe e estatísticas...",
    "Verificando status de jogadores (lesões e suspensões)...",
    "Aplicando Raciocínio Super-Humano...",
    "Executando simulação RL em ambiente duplo...",
    "Gerando análise de confronto tático...",
    "Consolidando previsão...",
];

const StatBar: React.FC<{ label: string; value1: number; value2: number; isPercentage?: boolean }> = ({ label, value1, value2, isPercentage }) => {
    const total = value1 + value2;
    const percentage1 = total > 0 ? (value1 / total) * 100 : 50;
    
    return (
        <div>
            <div className="flex justify-between items-center text-sm mb-1 font-medium">
                <span className="text-gray-300">{isPercentage ? `${value1.toFixed(1)}%` : value1.toFixed(1)}</span>
                <span className="text-gray-400 uppercase tracking-wider">{label}</span>
                <span className="text-gray-300">{isPercentage ? `${value2.toFixed(1)}%` : value2.toFixed(1)}</span>
            </div>
            <div className="flex h-2.5 rounded-full bg-gray-600">
                <div className="bg-white rounded-l-full" style={{ width: `${percentage1}%` }}></div>
                <div className="bg-gray-400 rounded-r-full" style={{ width: `${100 - percentage1}%` }}></div>
            </div>
        </div>
    );
};

const PlayerStatusList: React.FC<{ players: Player[] }> = ({ players }) => {
    if (!players || players.length === 0) {
        return <p className="text-gray-400 italic">Sem alterações significativas de jogadores.</p>;
    }

    const statusColorMap: { [key: string]: string } = {
        'Lesionado': 'text-red-400',
        'Suspenso': 'text-yellow-400',
        'Dúvida': 'text-orange-400',
    };

    const filteredPlayers = players.filter(p => p.status !== 'Disponível');

    if (filteredPlayers.length === 0) {
        return <p className="text-gray-400 italic">Todos os jogadores importantes estão disponíveis.</p>;
    }

    return (
        <ul className="space-y-2">
            {filteredPlayers.map((player, index) => (
                <li key={index} className="flex items-center justify-between text-sm">
                    <span className="font-medium">{player.name} ({player.position})</span>
                    <span className={`font-bold ${statusColorMap[player.status] || 'text-gray-400'}`}>{player.status}</span>
                </li>
            ))}
        </ul>
    );
};

const App: React.FC = () => {
    const [opponent, setOpponent] = useState<string>('');
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [loadingMessage, setLoadingMessage] = useState<string>('');
    const [analysisResult, setAnalysisResult] = useState<MatchAnalysis | null>(null);
    const [error, setError] = useState<string | null>(null);

    const handleAnalysis = useCallback(async () => {
        if (!opponent.trim()) {
            setError('Por favor, digite o nome do adversário.');
            return;
        }
        setIsLoading(true);
        setError(null);
        setAnalysisResult(null);

        let messageIndex = 0;
        const intervalId = setInterval(() => {
            setLoadingMessage(loadingMessages[messageIndex]);
            messageIndex = (messageIndex + 1) % loadingMessages.length;
        }, 1500);

        try {
            const result = await getMatchPrediction(opponent, "Brasileirão");
            setAnalysisResult(result);
        } catch (e) {
            console.error(e);
            setError('Falha ao obter a análise da partida. A IA pode estar ocupada ou ocorreu um erro. Verifique sua chave de API e tente novamente.');
        } finally {
            clearInterval(intervalId);
            setIsLoading(false);
            setLoadingMessage('');
        }
    }, [opponent]);

    return (
        <div className="min-h-screen bg-gray-900 text-gray-100 p-4 sm:p-6 md:p-8">
            <div className="max-w-7xl mx-auto">
                <header className="text-center mb-8">
                    <div className="flex justify-center items-center gap-4">
                        <CorinthiansLogo />
                        <h1 className="font-teko text-5xl md:text-7xl font-bold uppercase tracking-wider">
                            Analisador de Partidas
                        </h1>
                    </div>
                    <p className="text-gray-400 md:text-lg">
                        Análise de Partidas do Corinthians com Inteligência Artificial
                    </p>
                </header>

                <main>
                    <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 mb-8 shadow-2xl">
                        <div className="flex flex-col md:flex-row gap-4 items-center">
                            <input
                                type="text"
                                value={opponent}
                                onChange={(e) => setOpponent(e.target.value)}
                                placeholder="Digite o nome do adversário (ex: Palmeiras)"
                                className="w-full flex-grow bg-gray-700 border border-gray-600 rounded-md px-4 py-3 text-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-white focus:outline-none transition"
                                disabled={isLoading}
                            />
                            <button
                                onClick={handleAnalysis}
                                disabled={isLoading}
                                className="w-full md:w-auto bg-white text-gray-900 font-bold text-lg px-8 py-3 rounded-md hover:bg-gray-200 transition-transform transform hover:scale-105 disabled:bg-gray-500 disabled:cursor-not-allowed disabled:scale-100"
                            >
                                {isLoading ? 'Analisando...' : 'Analisar Partida'}
                            </button>
                        </div>
                        {error && <p className="text-red-400 mt-4 text-center">{error}</p>}
                    </div>

                    {isLoading && (
                        <div className="text-center">
                            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-white mb-4"></div>
                            <p className="text-xl font-teko tracking-wider">{loadingMessage}</p>
                        </div>
                    )}

                    {analysisResult && (
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 animate-fade-in">
                            {/* Prediction Summary */}
                            <div className="md:col-span-2 lg:col-span-3 bg-gray-800 border border-gray-700 rounded-lg p-6 shadow-lg flex flex-col md:flex-row items-center justify-between gap-6">
                                <div className="text-center md:text-left">
                                    <h2 className="font-teko text-4xl uppercase">{analysisResult.match_info.corinthians} vs {analysisResult.match_info.opponent}</h2>
                                    <p className="text-gray-400 text-lg">{analysisResult.match_info.competition}</p>
                                </div>
                                <div className="flex gap-4 md:gap-8 text-center">
                                    <div>
                                        <p className="font-teko text-4xl">{(analysisResult.prediction.probability_home_win * 100).toFixed(0)}%</p>
                                        <p className="text-sm text-gray-400">Vitória Corinthians</p>
                                    </div>
                                    <div>
                                        <p className="font-teko text-4xl">{(analysisResult.prediction.probability_draw * 100).toFixed(0)}%</p>
                                        <p className="text-sm text-gray-400">Empate</p>
                                    </div>
                                    <div>
                                        <p className="font-teko text-4xl">{(analysisResult.prediction.probability_away_win * 100).toFixed(0)}%</p>
                                        <p className="text-sm text-gray-400">Vitória {analysisResult.match_info.opponent}</p>
                                    </div>
                                </div>
                                <div className="text-center bg-gray-700 p-4 rounded-lg">
                                    <p className="text-sm text-gray-400 uppercase tracking-widest">Palpite</p>
                                    <p className="font-teko text-5xl">{analysisResult.prediction.winner}</p>
                                    <p className="text-xs text-gray-400">Confiança: {(analysisResult.prediction.confidence_score * 100).toFixed(0)}%</p>
                                </div>
                            </div>
                            
                            {/* Team Stats */}
                            <AnalysisCard title="Estatísticas das Equipes" icon={<BarChartIcon />} className="md:col-span-2">
                                <div className="space-y-4">
                                  <div className="flex justify-between items-center font-bold text-lg font-teko tracking-wider">
                                    <span>CORINTHIANS</span>
                                    <span>{analysisResult.match_info.opponent.toUpperCase()}</span>
                                  </div>
                                  <StatBar label="Posse de Bola" value1={analysisResult.corinthians_stats.avg_possession} value2={analysisResult.opponent_stats.avg_possession} isPercentage />
                                  <StatBar label="Chutes / Jogo" value1={analysisResult.corinthians_stats.shots_per_game} value2={analysisResult.opponent_stats.shots_per_game} />
                                  <StatBar label="Chutes no Alvo / Jogo" value1={analysisResult.corinthians_stats.shots_on_target_per_game} value2={analysisResult.opponent_stats.shots_on_target_per_game} />
                                  <StatBar label="Gols Marcados (Últ. 10)" value1={analysisResult.corinthians_stats.goals_scored_last_10} value2={analysisResult.opponent_stats.goals_scored_last_10} />
                                   <StatBar label="Gols Sofridos (Últ. 10)" value1={analysisResult.corinthians_stats.goals_conceded_last_10} value2={analysisResult.opponent_stats.goals_conceded_last_10} />
                                  <div className="text-center pt-2 text-gray-400 text-sm">
                                      <p>Últimos 10: COR {analysisResult.corinthians_stats.last_10_games_record} vs OPP {analysisResult.opponent_stats.last_10_games_record}</p>
                                  </div>
                                </div>
                            </AnalysisCard>
                            
                            {/* Key Factors */}
                            <AnalysisCard title="Fatores Chave" icon={<CheckCircleIcon />}>
                                <ul className="list-disc list-inside space-y-2">
                                    {analysisResult.key_factors.map((factor, index) => <li key={index}>{factor}</li>)}
                                </ul>
                            </AnalysisCard>
                            
                            {/* Head to Head */}
                            <AnalysisCard title="Confronto Direto" icon={<UsersIcon />}>
                                <div className="text-center mb-4">
                                    <p>Total de Partidas: {analysisResult.head_to_head.total_matches}</p>
                                    <p>
                                        <span className="font-bold">{analysisResult.head_to_head.corinthians_wins}</span> Vitórias (COR) |
                                        <span className="font-bold"> {analysisResult.head_to_head.draws}</span> Empates |
                                        <span className="font-bold"> {analysisResult.head_to_head.opponent_wins}</span> Vitórias (OPP)
                                    </p>
                                </div>
                                <p className="font-medium text-center mb-2">Últimos 5 Jogos:</p>
                                <div className="flex flex-col items-center space-y-1">
                                    {analysisResult.head_to_head.last_5_results.map((result, index) => (
                                        <span key={index} className="bg-gray-700 px-3 py-1 rounded-full text-xs font-mono">{result}</span>
                                    ))}
                                </div>
                            </AnalysisCard>
                            
                            {/* Player Status */}
                            <div className="lg:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-6">
                                <AnalysisCard title="Situação dos Jogadores (Corinthians)" icon={<ShieldIcon className="text-white"/>}>
                                    <PlayerStatusList players={analysisResult.players_status.corinthians} />
                                </AnalysisCard>
                                <AnalysisCard title={`Situação dos Jogadores (${analysisResult.match_info.opponent})`} icon={<ShieldIcon className="text-gray-400"/>}>
                                    <PlayerStatusList players={analysisResult.players_status.opponent} />
                                </AnalysisCard>
                            </div>
                            
                            {/* Risk Factors */}
                            <AnalysisCard title="Fatores de Risco" icon={<AlertTriangleIcon />}>
                                <ul className="list-disc list-inside space-y-2">
                                    {analysisResult.risk_factors.map((factor, index) => <li key={index}>{factor}</li>)}
                                </ul>
                            </AnalysisCard>
                            
                            {/* Tactical Analysis */}
                            <AnalysisCard title="Análise Tática" icon={<TacticIcon />} className="md:col-span-2">
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-center mb-4">
                                    <div>
                                        <p className="text-gray-400">Formação Corinthians</p>
                                        <p className="font-teko text-3xl">{analysisResult.tactical_analysis.corinthians_formation}</p>
                                    </div>
                                     <div>
                                        <p className="text-gray-400">Formação {analysisResult.match_info.opponent}</p>
                                        <p className="font-teko text-3xl">{analysisResult.tactical_analysis.opponent_formation}</p>
                                    </div>
                                </div>
                                <div className="space-y-3">
                                    <div>
                                        <h4 className="font-bold mb-1">Dinâmica Prevista:</h4>
                                        <p>{analysisResult.tactical_analysis.predicted_dynamics}</p>
                                    </div>
                                    <div>
                                        <h4 className="font-bold mb-1">Duelos Chave:</h4>
                                        <ul className="list-disc list-inside">
                                            {analysisResult.tactical_analysis.key_battles.map((battle, i) => <li key={i}>{battle}</li>)}
                                        </ul>
                                    </div>
                                </div>
                            </AnalysisCard>

                        </div>
                    )}
                </main>
            </div>
        </div>
    );
};

export default App;