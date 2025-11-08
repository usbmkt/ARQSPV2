
import { GoogleGenAI, Type } from "@google/genai";
import { MatchAnalysis } from '../types';

const getGeminiService = () => {
  if (!process.env.API_KEY) {
    throw new Error("API_KEY environment variable is not set.");
  }
  return new GoogleGenAI({ apiKey: process.env.API_KEY });
};

const responseSchema = {
    type: Type.OBJECT,
    properties: {
        match_info: {
            type: Type.OBJECT,
            properties: {
                corinthians: { type: Type.STRING },
                opponent: { type: Type.STRING },
                competition: { type: Type.STRING },
            },
            required: ["corinthians", "opponent", "competition"],
        },
        prediction: {
            type: Type.OBJECT,
            properties: {
                winner: { type: Type.STRING },
                probability_home_win: { type: Type.NUMBER },
                probability_draw: { type: Type.NUMBER },
                probability_away_win: { type: Type.NUMBER },
                expected_goals_corinthians: { type: Type.NUMBER },
                expected_goals_opponent: { type: Type.NUMBER },
                confidence_score: { type: Type.NUMBER },
            },
             required: ["winner", "probability_home_win", "probability_draw", "probability_away_win", "expected_goals_corinthians", "expected_goals_opponent", "confidence_score"],
        },
        key_factors: { type: Type.ARRAY, items: { type: Type.STRING } },
        risk_factors: { type: Type.ARRAY, items: { type: Type.STRING } },
        corinthians_stats: {
            type: Type.OBJECT,
            properties: {
                last_10_games_record: { type: Type.STRING },
                avg_possession: { type: Type.NUMBER },
                shots_per_game: { type: Type.NUMBER },
                shots_on_target_per_game: { type: Type.NUMBER },
                goals_scored_last_10: { type: Type.NUMBER },
                goals_conceded_last_10: { type: Type.NUMBER },
            },
             required: ["last_10_games_record", "avg_possession", "shots_per_game", "shots_on_target_per_game", "goals_scored_last_10", "goals_conceded_last_10"],
        },
        opponent_stats: {
            type: Type.OBJECT,
            properties: {
                last_10_games_record: { type: Type.STRING },
                avg_possession: { type: Type.NUMBER },
                shots_per_game: { type: Type.NUMBER },
                shots_on_target_per_game: { type: Type.NUMBER },
                goals_scored_last_10: { type: Type.NUMBER },
                goals_conceded_last_10: { type: Type.NUMBER },
            },
             required: ["last_10_games_record", "avg_possession", "shots_per_game", "shots_on_target_per_game", "goals_scored_last_10", "goals_conceded_last_10"],
        },
        head_to_head: {
            type: Type.OBJECT,
            properties: {
                total_matches: { type: Type.NUMBER },
                corinthians_wins: { type: Type.NUMBER },
                opponent_wins: { type: Type.NUMBER },
                draws: { type: Type.NUMBER },
                last_5_results: { type: Type.ARRAY, items: { type: Type.STRING } },
            },
             required: ["total_matches", "corinthians_wins", "opponent_wins", "draws", "last_5_results"],
        },
        players_status: {
            type: Type.OBJECT,
            properties: {
                corinthians: {
                    type: Type.ARRAY,
                    items: {
                        type: Type.OBJECT,
                        properties: {
                            name: { type: Type.STRING },
                            position: { type: Type.STRING },
                            status: { type: Type.STRING },
                            importance: { type: Type.STRING },
                        },
                        required: ["name", "position", "status", "importance"],
                    },
                },
                opponent: {
                    type: Type.ARRAY,
                    items: {
                        type: Type.OBJECT,
                        properties: {
                            name: { type: Type.STRING },
                            position: { type: Type.STRING },
                            status: { type: Type.STRING },
                            importance: { type: Type.STRING },
                        },
                         required: ["name", "position", "status", "importance"],
                    },
                },
            },
             required: ["corinthians", "opponent"],
        },
        tactical_analysis: {
            type: Type.OBJECT,
            properties: {
                corinthians_formation: { type: Type.STRING },
                opponent_formation: { type: Type.STRING },
                key_battles: { type: Type.ARRAY, items: { type: Type.STRING } },
                predicted_dynamics: { type: Type.STRING },
            },
             required: ["corinthians_formation", "opponent_formation", "key_battles", "predicted_dynamics"],
        },
    },
    required: ["match_info", "prediction", "key_factors", "risk_factors", "corinthians_stats", "opponent_stats", "head_to_head", "players_status", "tactical_analysis"],
};

export const getMatchPrediction = async (opponent: string, competition: string): Promise<MatchAnalysis> => {
  const ai = getGeminiService();

  const prompt = `
    Aja como uma IA analista de dados esportivos de classe mundial, especialista em futebol brasileiro.
    Sua tarefa é gerar uma análise de previsão de partida detalhada, realista, porém fictícia, para um jogo entre Corinthians e ${opponent} no ${competition}.
    
    Gere um objeto JSON completo em português do Brasil que siga o schema fornecido. Os dados devem ser plausíveis e refletir estatísticas e narrativas típicas do futebol.
    
    - IMPORTANTE: Use nomes de jogadores que ESTÃO ATUALMENTE no elenco do Corinthians. Por exemplo, jogadores como Renato Augusto ou Cássio não fazem mais parte do time. Use jogadores atuais como Yuri Alberto, Wesley, Coronado, etc. Faça o mesmo para o time adversário, usando jogadores realistas.
    - Os status dos jogadores devem ser: 'Lesionado', 'Suspenso', 'Dúvida', 'Disponível'.
    - Todas as strings de texto (como 'key_factors', 'risk_factors', 'predicted_dynamics', etc.) DEVEM estar em português do Brasil.
    - Faça as estatísticas serem críveis. Um time de ponta pode ter 60% de posse de bola, enquanto um time defensivo pode ter 40%.
    - As probabilidades de vitória, empate e derrota devem somar aproximadamente 1.0.
    - Os fatores chave e de risco devem ser perspicazes e diretamente relacionados aos dados que você gerar.
    - Crie uma narrativa convincente através dos dados. Por exemplo, se o Corinthians tem um ataque forte, mas uma defesa fraca, isso deve ser refletido nas estatísticas e nos fatores chave.
    - O 'last_10_games_record' deve ser uma string no formato "6V-2E-2D" (V=Vitória, E=Empate, D=Derrota).
    - Os 'last_5_results' no confronto direto devem ser strings como "COR 2-1 ${opponent.substring(0,3).toUpperCase()}" ou "EMP 1-1".
  `;
  
  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents: prompt,
    config: {
      responseMimeType: "application/json",
      responseSchema: responseSchema,
    },
  });

  const jsonText = response.text.trim();
  return JSON.parse(jsonText);
};