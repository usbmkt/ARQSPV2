import asyncio
import aiohttp
import json
from typing import Dict, Any, Optional

class SofaScoreScraper:
    def __init__(self):
        self.base_url = "https://api.sofascore.com/api/v1"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Origin': 'https://www.sofascore.com',
            'Referer': 'https://www.sofascore.com/'
        }

    async def _fetch(self, session, url):
        async with session.get(url, headers=self.headers) as response:
            if response.status == 200:
                return await response.json()
            return None

    async def _get_team_id(self, session, team_name: str) -> Optional[int]:
        search_url = f"{self.base_url}/search/all?q={team_name}"
        data = await self._fetch(session, search_url)
        if data:
            for result in data.get('results', []):
                if result.get('type') == 'team' and team_name.lower() in result.get('name', '').lower():
                    return result.get('entity', {}).get('id')
        return None

    async def fetch_team_data(self, team_name: str) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            team_id = await self._get_team_id(session, team_name)
            if not team_id:
                return {"error": f"Team not found: {team_name}"}

            team_stats_url = f"{self.base_url}/team/{team_id}/statistics/overall"
            team_events_url = f"{self.base_url}/team/{team_id}/events/last/0"

            stats_data, events_data = await asyncio.gather(
                self._fetch(session, team_stats_url),
                self._fetch(session, team_events_url)
            )

            return self._process_data(team_name, stats_data, events_data)

    def _process_data(self, team_name: str, stats_data: Dict[str, Any], events_data: Dict[str, Any]) -> Dict[str, Any]:
        if not stats_data or not events_data:
            return {"error": "Could not fetch all data for {team_name}"}

        # This is a simplified data processing step. More detailed processing will be added later.
        processed_data = {
            "team_name": team_name,
            "avg_possession": stats_data.get('statistics', {}).get('avgPossession'),
            "shots_per_game": stats_data.get('statistics', {}).get('shotsPerGame'),
            "last_10_games": []
        }

        for event in events_data.get('events', [])[:10]:
            processed_data["last_10_games"].append({
                "home_team": event.get('homeTeam', {}).get('name'),
                "away_team": event.get('awayTeam', {}).get('name'),
                "home_score": event.get('homeScore', {}).get('current'),
                "away_score": event.get('awayScore', {}).get('current'),
            })

        return processed_data
