
export interface Prediction {
  winner: 'Corinthians' | 'Draw' | string;
  probability_home_win: number;
  probability_draw: number;
  probability_away_win: number;
  expected_goals_corinthians: number;
  expected_goals_opponent: number;
  confidence_score: number;
}

export interface Player {
  name: string;
  position: string;
  status: 'Lesionado' | 'Suspenso' | 'Dúvida' | 'Disponível';
  importance: 'High' | 'Medium' | 'Low';
}

export interface TeamStats {
  last_10_games_record: string; // e.g., "6V-2E-2D"
  avg_possession: number;
  shots_per_game: number;
  shots_on_target_per_game: number;
  goals_scored_last_10: number;
  goals_conceded_last_10: number;
}

export interface HeadToHead {
  total_matches: number;
  corinthians_wins: number;
  opponent_wins: number;
  draws: number;
  last_5_results: string[]; // e.g., ["COR 2-1 OPP", "EMP 1-1"]
}

export interface TacticalAnalysis {
  corinthians_formation: string;
  opponent_formation: string;
  key_battles: string[];
  predicted_dynamics: string;
}

export interface MatchAnalysis {
  match_info: {
    corinthians: string;
    opponent: string;
    competition: string;
  };
  prediction: Prediction;
  key_factors: string[];
  risk_factors: string[];
  corinthians_stats: TeamStats;
  opponent_stats: TeamStats;
  head_to_head: HeadToHead;
  players_status: {
    corinthians: Player[];
    opponent: Player[];
  };
  tactical_analysis: TacticalAnalysis;
}