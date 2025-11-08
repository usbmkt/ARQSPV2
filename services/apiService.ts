import { MatchAnalysis } from '../types';

const API_URL = 'http://localhost:5000';

export const getMatchPrediction = async (opponent: string, competition: string): Promise<MatchAnalysis> => {
  const response = await fetch(`${API_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ opponent, competition }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || 'An unknown error occurred');
  }

  const data = await response.json();
  return data;
};
