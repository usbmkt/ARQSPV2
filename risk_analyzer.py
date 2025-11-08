"""
Sistema de Análise Automatizada de Riscos
Implementa identificação, scoring e mitigação de riscos
"""

import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import re

@dataclass
class Risk:
    """Estrutura de um risco identificado"""
    id: str
    name: str
    category: str  # market, financial, operational, regulatory, external
    description: str
    probability: float  # 0.0 a 1.0
    impact: float  # 0.0 a 1.0
    risk_score: float  # probability * impact
    severity_level: str  # low, medium, high, critical
    triggers: List[str]  # Indicadores que sugerem este risco
    mitigation_strategies: List[str]
    monitoring_indicators: List[str]
    time_horizon: str  # short, medium, long
    
@dataclass
class RiskMatrix:
    """Matriz de riscos com análise completa"""
    risks: List[Risk]
    overall_risk_score: float
    risk_distribution: Dict[str, int]
    priority_risks: List[Risk]
    mitigation_plan: List[str]
    monitoring_plan: List[str]

class RiskAnalyzer:
    """Sistema principal de análise de riscos"""
    
    def __init__(self):
        self.risk_database = self._load_risk_database()
        self.mitigation_strategies = self._load_mitigation_strategies()
        self.industry_risk_profiles = self._load_industry_profiles()
        
    def _load_risk_database(self) -> Dict[str, Dict]:
        """Carrega base de dados de riscos por categoria"""
        return {
            'market': {
                'demand_decline': {
                    'name': 'Declínio da Demanda',
                    'description': 'Redução significativa na demanda por produtos/serviços',
                    'triggers': ['queda vendas', 'redução procura', 'mercado saturado', 'concorrência'],
                    'base_probability': 0.3,
                    'base_impact': 0.8,
                    'mitigation': ['diversificação produtos', 'novos mercados', 'inovação'],
                    'indicators': ['volume vendas', 'leads qualificados', 'pesquisas mercado']
                },
                'new_competitors': {
                    'name': 'Entrada de Novos Concorrentes',
                    'description': 'Chegada de competidores com vantagens significativas',
                    'triggers': ['novos players', 'concorrência', 'market share', 'preços baixos'],
                    'base_probability': 0.4,
                    'base_impact': 0.6,
                    'mitigation': ['diferenciação', 'fidelização clientes', 'barreiras entrada'],
                    'indicators': ['análise concorrencial', 'participação mercado', 'preços']
                },
                'market_saturation': {
                    'name': 'Saturação do Mercado',
                    'description': 'Mercado atingiu ponto de saturação limitando crescimento',
                    'triggers': ['crescimento lento', 'saturação', 'maturidade mercado'],
                    'base_probability': 0.25,
                    'base_impact': 0.7,
                    'mitigation': ['expansão geográfica', 'novos segmentos', 'inovação'],
                    'indicators': ['taxa crescimento mercado', 'densidade competitiva']
                }
            },
            'financial': {
                'cash_flow_problems': {
                    'name': 'Problemas de Fluxo de Caixa',
                    'description': 'Dificuldades para manter fluxo de caixa positivo',
                    'triggers': ['fluxo caixa', 'capital giro', 'inadimplência', 'pagamentos'],
                    'base_probability': 0.35,
                    'base_impact': 0.9,
                    'mitigation': ['reserva emergência', 'gestão recebíveis', 'linhas crédito'],
                    'indicators': ['DRE', 'fluxo caixa', 'índices liquidez']
                },
                'high_customer_acquisition_cost': {
                    'name': 'Alto Custo de Aquisição de Clientes',
                    'description': 'CAC elevado comprometendo rentabilidade',
                    'triggers': ['cac alto', 'custo aquisição', 'marketing caro', 'conversão baixa'],
                    'base_probability': 0.4,
                    'base_impact': 0.6,
                    'mitigation': ['otimização marketing', 'referrals', 'retenção clientes'],
                    'indicators': ['CAC', 'LTV/CAC ratio', 'ROI marketing']
                },
                'currency_fluctuation': {
                    'name': 'Flutuação Cambial',
                    'description': 'Variações cambiais afetando custos ou receitas',
                    'triggers': ['câmbio', 'dólar', 'importação', 'exportação', 'moeda'],
                    'base_probability': 0.6,
                    'base_impact': 0.5,
                    'mitigation': ['hedge cambial', 'fornecedores locais', 'pricing dinâmico'],
                    'indicators': ['taxa câmbio', 'exposição cambial', 'custos importação']
                }
            },
            'operational': {
                'supply_chain_disruption': {
                    'name': 'Interrupção da Cadeia de Suprimentos',
                    'description': 'Problemas com fornecedores ou logística',
                    'triggers': ['fornecedores', 'supply chain', 'logística', 'estoque', 'entrega'],
                    'base_probability': 0.3,
                    'base_impact': 0.8,
                    'mitigation': ['múltiplos fornecedores', 'estoque segurança', 'parcerias'],
                    'indicators': ['lead time fornecedores', 'nível estoque', 'qualidade entrega']
                },
                'talent_shortage': {
                    'name': 'Escassez de Talentos',
                    'description': 'Dificuldade para contratar e reter profissionais qualificados',
                    'triggers': ['contratação', 'talentos', 'turnover', 'recursos humanos'],
                    'base_probability': 0.5,
                    'base_impact': 0.6,
                    'mitigation': ['employer branding', 'desenvolvimento interno', 'retenção'],
                    'indicators': ['turnover rate', 'tempo contratação', 'satisfação funcionários']
                },
                'technology_obsolescence': {
                    'name': 'Obsolescência Tecnológica',
                    'description': 'Tecnologias utilizadas tornando-se obsoletas',
                    'triggers': ['tecnologia', 'obsolescência', 'inovação', 'digital', 'sistemas'],
                    'base_probability': 0.4,
                    'base_impact': 0.7,
                    'mitigation': ['atualização contínua', 'P&D', 'parcerias tecnológicas'],
                    'indicators': ['investimento tecnologia', 'ciclo vida produtos', 'inovação']
                }
            },
            'regulatory': {
                'regulatory_changes': {
                    'name': 'Mudanças Regulatórias',
                    'description': 'Alterações em leis e regulamentações do setor',
                    'triggers': ['regulamentação', 'lei', 'norma', 'compliance', 'governo'],
                    'base_probability': 0.4,
                    'base_impact': 0.6,
                    'mitigation': ['monitoramento regulatório', 'compliance proativo', 'lobby'],
                    'indicators': ['mudanças legislativas', 'consultas públicas', 'jurisprudência']
                },
                'tax_changes': {
                    'name': 'Alterações Tributárias',
                    'description': 'Mudanças na carga tributária ou estrutura fiscal',
                    'triggers': ['imposto', 'tributário', 'fiscal', 'alíquota', 'receita federal'],
                    'base_probability': 0.6,
                    'base_impact': 0.5,
                    'mitigation': ['planejamento tributário', 'estrutura fiscal', 'consultoria'],
                    'indicators': ['propostas tributárias', 'arrecadação governo', 'déficit fiscal']
                },
                'licensing_issues': {
                    'name': 'Problemas de Licenciamento',
                    'description': 'Dificuldades para obter ou manter licenças necessárias',
                    'triggers': ['licença', 'autorização', 'alvará', 'certificação', 'órgão'],
                    'base_probability': 0.25,
                    'base_impact': 0.8,
                    'mitigation': ['compliance rigoroso', 'relacionamento órgãos', 'consultoria'],
                    'indicators': ['status licenças', 'prazos renovação', 'mudanças requisitos']
                }
            },
            'external': {
                'economic_recession': {
                    'name': 'Recessão Econômica',
                    'description': 'Deterioração das condições econômicas gerais',
                    'triggers': ['recessão', 'crise econômica', 'pib', 'desemprego', 'inflação'],
                    'base_probability': 0.3,
                    'base_impact': 0.9,
                    'mitigation': ['diversificação', 'produtos essenciais', 'eficiência custos'],
                    'indicators': ['PIB', 'taxa desemprego', 'confiança consumidor']
                },
                'pandemic_impact': {
                    'name': 'Impacto de Pandemias',
                    'description': 'Efeitos de crises sanitárias nas operações',
                    'triggers': ['pandemia', 'covid', 'saúde pública', 'lockdown', 'isolamento'],
                    'base_probability': 0.2,
                    'base_impact': 0.8,
                    'mitigation': ['trabalho remoto', 'digitalização', 'plano contingência'],
                    'indicators': ['casos covid', 'restrições governo', 'vacinação']
                },
                'climate_change': {
                    'name': 'Mudanças Climáticas',
                    'description': 'Impactos de eventos climáticos extremos',
                    'triggers': ['clima', 'sustentabilidade', 'carbono', 'ambiental', 'esg'],
                    'base_probability': 0.4,
                    'base_impact': 0.5,
                    'mitigation': ['sustentabilidade', 'energia renovável', 'adaptação'],
                    'indicators': ['eventos climáticos', 'regulação ambiental', 'pressão ESG']
                }
            }
        }
    
    def _load_mitigation_strategies(self) -> Dict[str, List[str]]:
        """Carrega estratégias de mitigação por categoria"""
        return {
            'market': [
                'Diversificação de produtos e serviços',
                'Expansão para novos mercados geográficos',
                'Segmentação de clientes mais específica',
                'Inovação contínua em produtos',
                'Parcerias estratégicas',
                'Programa de fidelização de clientes'
            ],
            'financial': [
                'Criação de reserva de emergência',
                'Diversificação de fontes de receita',
                'Gestão rigorosa de fluxo de caixa',
                'Negociação de prazos com fornecedores',
                'Linhas de crédito pré-aprovadas',
                'Monitoramento de indicadores financeiros'
            ],
            'operational': [
                'Múltiplos fornecedores para itens críticos',
                'Automação de processos',
                'Treinamento e desenvolvimento de equipe',
                'Backup de sistemas críticos',
                'Planos de contingência operacional',
                'Melhoria contínua de processos'
            ],
            'regulatory': [
                'Monitoramento ativo de mudanças regulatórias',
                'Relacionamento com órgãos reguladores',
                'Consultoria jurídica especializada',
                'Compliance proativo',
                'Participação em associações setoriais',
                'Documentação rigorosa de processos'
            ],
            'external': [
                'Diversificação geográfica',
                'Produtos/serviços essenciais',
                'Flexibilidade operacional',
                'Seguros adequados',
                'Planos de continuidade de negócios',
                'Monitoramento de indicadores externos'
            ]
        }
    
    def _load_industry_profiles(self) -> Dict[str, Dict]:
        """Carrega perfis de risco por indústria"""
        return {
            'technology': {
                'high_risk_categories': ['operational', 'market'],
                'risk_multipliers': {'operational': 1.3, 'market': 1.2},
                'specific_risks': ['technology_obsolescence', 'talent_shortage']
            },
            'retail': {
                'high_risk_categories': ['market', 'external'],
                'risk_multipliers': {'market': 1.2, 'external': 1.1},
                'specific_risks': ['demand_decline', 'supply_chain_disruption']
            },
            'manufacturing': {
                'high_risk_categories': ['operational', 'regulatory'],
                'risk_multipliers': {'operational': 1.3, 'regulatory': 1.2},
                'specific_risks': ['supply_chain_disruption', 'regulatory_changes']
            },
            'services': {
                'high_risk_categories': ['financial', 'market'],
                'risk_multipliers': {'financial': 1.1, 'market': 1.1},
                'specific_risks': ['talent_shortage', 'high_customer_acquisition_cost']
            },
            'healthcare': {
                'high_risk_categories': ['regulatory', 'external'],
                'risk_multipliers': {'regulatory': 1.4, 'external': 1.2},
                'specific_risks': ['regulatory_changes', 'licensing_issues']
            }
        }
    
    def analyze_risks(self, content: str, industry: str = 'services', context: Dict = None) -> RiskMatrix:
        """Analisa riscos baseado no conteúdo e contexto"""
        content_lower = content.lower()
        identified_risks = []
        
        # Identifica riscos baseado em triggers no conteúdo
        for category, risks in self.risk_database.items():
            for risk_id, risk_data in risks.items():
                # Verifica se algum trigger está presente
                trigger_count = sum(1 for trigger in risk_data['triggers'] if trigger in content_lower)
                
                if trigger_count > 0:
                    # Calcula probabilidade e impacto ajustados
                    probability = self._calculate_adjusted_probability(
                        risk_data['base_probability'], 
                        trigger_count, 
                        len(risk_data['triggers']),
                        industry,
                        category
                    )
                    
                    impact = self._calculate_adjusted_impact(
                        risk_data['base_impact'],
                        industry,
                        category,
                        context
                    )
                    
                    risk_score = probability * impact
                    severity = self._determine_severity(risk_score)
                    
                    risk = Risk(
                        id=risk_id,
                        name=risk_data['name'],
                        category=category,
                        description=risk_data['description'],
                        probability=probability,
                        impact=impact,
                        risk_score=risk_score,
                        severity_level=severity,
                        triggers=risk_data['triggers'],
                        mitigation_strategies=risk_data['mitigation'],
                        monitoring_indicators=risk_data['indicators'],
                        time_horizon=self._determine_time_horizon(category, risk_id)
                    )
                    
                    identified_risks.append(risk)
        
        # Adiciona riscos específicos da indústria
        if industry in self.industry_risk_profiles:
            profile = self.industry_risk_profiles[industry]
            for specific_risk in profile['specific_risks']:
                if not any(r.id == specific_risk for r in identified_risks):
                    # Adiciona risco específico mesmo sem trigger
                    risk_data = self._find_risk_data(specific_risk)
                    if risk_data:
                        category = self._find_risk_category(specific_risk)
                        probability = risk_data['base_probability'] * 1.2  # Aumenta por ser específico da indústria
                        impact = risk_data['base_impact']
                        risk_score = probability * impact
                        
                        risk = Risk(
                            id=specific_risk,
                            name=risk_data['name'],
                            category=category,
                            description=risk_data['description'],
                            probability=min(probability, 1.0),
                            impact=impact,
                            risk_score=risk_score,
                            severity_level=self._determine_severity(risk_score),
                            triggers=risk_data['triggers'],
                            mitigation_strategies=risk_data['mitigation'],
                            monitoring_indicators=risk_data['indicators'],
                            time_horizon=self._determine_time_horizon(category, specific_risk)
                        )
                        
                        identified_risks.append(risk)
        
        # Ordena riscos por score
        identified_risks.sort(key=lambda x: x.risk_score, reverse=True)
        
        # Calcula métricas gerais
        overall_score = sum(r.risk_score for r in identified_risks) / len(identified_risks) if identified_risks else 0
        
        risk_distribution = {}
        for category in ['market', 'financial', 'operational', 'regulatory', 'external']:
            risk_distribution[category] = len([r for r in identified_risks if r.category == category])
        
        # Identifica riscos prioritários (top 5 ou score > 0.6)
        priority_risks = [r for r in identified_risks if r.risk_score > 0.6][:5]
        
        # Gera planos de mitigação e monitoramento
        mitigation_plan = self._generate_mitigation_plan(priority_risks)
        monitoring_plan = self._generate_monitoring_plan(priority_risks)
        
        return RiskMatrix(
            risks=identified_risks,
            overall_risk_score=overall_score,
            risk_distribution=risk_distribution,
            priority_risks=priority_risks,
            mitigation_plan=mitigation_plan,
            monitoring_plan=monitoring_plan
        )
    
    def _calculate_adjusted_probability(self, base_prob: float, trigger_count: int, 
                                      total_triggers: int, industry: str, category: str) -> float:
        """Calcula probabilidade ajustada baseada em fatores"""
        # Ajuste baseado na densidade de triggers
        trigger_density = trigger_count / total_triggers
        density_multiplier = 1 + (trigger_density * 0.5)  # Até 50% de aumento
        
        # Ajuste baseado na indústria
        industry_multiplier = 1.0
        if industry in self.industry_risk_profiles:
            profile = self.industry_risk_profiles[industry]
            if category in profile.get('risk_multipliers', {}):
                industry_multiplier = profile['risk_multipliers'][category]
        
        adjusted_prob = base_prob * density_multiplier * industry_multiplier
        return min(adjusted_prob, 1.0)  # Não pode exceder 100%
    
    def _calculate_adjusted_impact(self, base_impact: float, industry: str, 
                                 category: str, context: Dict = None) -> float:
        """Calcula impacto ajustado baseado em fatores"""
        # Ajuste baseado no contexto (tamanho da empresa, recursos, etc.)
        context_multiplier = 1.0
        if context:
            if context.get('company_size') == 'small':
                context_multiplier = 1.2  # Empresas pequenas são mais vulneráveis
            elif context.get('company_size') == 'large':
                context_multiplier = 0.9  # Empresas grandes têm mais recursos
        
        adjusted_impact = base_impact * context_multiplier
        return min(adjusted_impact, 1.0)
    
    def _determine_severity(self, risk_score: float) -> str:
        """Determina nível de severidade baseado no score"""
        if risk_score >= 0.8:
            return 'critical'
        elif risk_score >= 0.6:
            return 'high'
        elif risk_score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _determine_time_horizon(self, category: str, risk_id: str) -> str:
        """Determina horizonte temporal do risco"""
        short_term_risks = ['cash_flow_problems', 'supply_chain_disruption', 'pandemic_impact']
        long_term_risks = ['technology_obsolescence', 'climate_change', 'market_saturation']
        
        if risk_id in short_term_risks:
            return 'short'
        elif risk_id in long_term_risks:
            return 'long'
        else:
            return 'medium'
    
    def _find_risk_data(self, risk_id: str) -> Optional[Dict]:
        """Encontra dados de um risco específico"""
        for category, risks in self.risk_database.items():
            if risk_id in risks:
                return risks[risk_id]
        return None
    
    def _find_risk_category(self, risk_id: str) -> str:
        """Encontra categoria de um risco específico"""
        for category, risks in self.risk_database.items():
            if risk_id in risks:
                return category
        return 'external'
    
    def _generate_mitigation_plan(self, priority_risks: List[Risk]) -> List[str]:
        """Gera plano de mitigação para riscos prioritários"""
        mitigation_actions = []
        
        # Agrupa estratégias por categoria
        category_strategies = {}
        for risk in priority_risks:
            if risk.category not in category_strategies:
                category_strategies[risk.category] = set()
            category_strategies[risk.category].update(risk.mitigation_strategies)
        
        # Gera ações específicas
        for category, strategies in category_strategies.items():
            category_name = category.replace('_', ' ').title()
            mitigation_actions.append(f"**{category_name}:**")
            for strategy in list(strategies)[:3]:  # Top 3 estratégias por categoria
                mitigation_actions.append(f"• {strategy}")
        
        # Adiciona ações gerais
        mitigation_actions.extend([
            "**Ações Gerais:**",
            "• Revisar e atualizar plano de riscos mensalmente",
            "• Estabelecer indicadores de alerta precoce",
            "• Criar comitê de gestão de riscos",
            "• Desenvolver planos de contingência detalhados"
        ])
        
        return mitigation_actions
    
    def _generate_monitoring_plan(self, priority_risks: List[Risk]) -> List[str]:
        """Gera plano de monitoramento para riscos prioritários"""
        monitoring_actions = []
        
        # Coleta todos os indicadores únicos
        all_indicators = set()
        for risk in priority_risks:
            all_indicators.update(risk.monitoring_indicators)
        
        monitoring_actions.extend([
            "**Indicadores-Chave a Monitorar:**"
        ])
        
        for indicator in list(all_indicators)[:10]:  # Top 10 indicadores
            monitoring_actions.append(f"• {indicator}")
        
        monitoring_actions.extend([
            "",
            "**Frequência de Monitoramento:**",
            "• Riscos críticos: Monitoramento semanal",
            "• Riscos altos: Monitoramento quinzenal", 
            "• Riscos médios: Monitoramento mensal",
            "• Revisão geral: Trimestral"
        ])
        
        return monitoring_actions
    
    def generate_risk_matrix_html(self, risk_matrix: RiskMatrix) -> str:
        """Gera HTML da matriz de riscos"""
        html = ['<div class="risk-analysis">']
        html.append('<h3>⚠️ Análise de Riscos</h3>')
        
        # Resumo geral
        html.append('<div class="risk-summary">')
        html.append(f'<div class="overall-score">Score Geral de Risco: <span class="score">{risk_matrix.overall_risk_score:.2f}</span></div>')
        html.append('<div class="risk-distribution">')
        html.append('<h4>Distribuição por Categoria:</h4>')
        
        for category, count in risk_matrix.risk_distribution.items():
            if count > 0:
                category_name = category.replace('_', ' ').title()
                html.append(f'<span class="risk-category-badge category-{category}">{category_name}: {count}</span>')
        
        html.append('</div>')
        html.append('</div>')
        
        # Matriz visual de riscos
        html.append('<div class="risk-matrix-visual">')
        html.append('<h4>🎯 Matriz de Probabilidade vs Impacto</h4>')
        html.append('<div class="matrix-grid">')
        
        # Cria grid 3x3 para a matriz
        for impact_level in ['Alto', 'Médio', 'Baixo']:
            for prob_level in ['Baixa', 'Média', 'Alta']:
                risks_in_cell = self._get_risks_for_cell(risk_matrix.risks, prob_level, impact_level)
                cell_class = self._get_cell_class(prob_level, impact_level)
                
                html.append(f'<div class="matrix-cell {cell_class}">')
                html.append(f'<div class="cell-header">{prob_level} / {impact_level}</div>')
                
                if risks_in_cell:
                    html.append('<div class="cell-risks">')
                    for risk in risks_in_cell[:2]:  # Máximo 2 riscos por célula
                        html.append(f'<div class="risk-item">{risk.name}</div>')
                    if len(risks_in_cell) > 2:
                        html.append(f'<div class="more-risks">+{len(risks_in_cell)-2} mais</div>')
                    html.append('</div>')
                
                html.append('</div>')
        
        html.append('</div>')
        html.append('</div>')
        
        # Riscos prioritários
        if risk_matrix.priority_risks:
            html.append('<div class="priority-risks">')
            html.append('<h4>🚨 Riscos Prioritários</h4>')
            
            for risk in risk_matrix.priority_risks:
                severity_class = f"severity-{risk.severity_level}"
                html.append(f'<div class="priority-risk-item {severity_class}">')
                html.append(f'<h5>{risk.name} <span class="risk-score">({risk.risk_score:.2f})</span></h5>')
                html.append(f'<p>{risk.description}</p>')
                html.append(f'<div class="risk-details">')
                html.append(f'<span class="probability">Probabilidade: {risk.probability:.0%}</span>')
                html.append(f'<span class="impact">Impacto: {risk.impact:.0%}</span>')
                html.append(f'<span class="horizon">Horizonte: {risk.time_horizon}</span>')
                html.append('</div>')
                html.append('</div>')
            
            html.append('</div>')
        
        # Plano de mitigação
        html.append('<div class="mitigation-plan">')
        html.append('<h4>🛡️ Plano de Mitigação</h4>')
        html.append('<div class="mitigation-actions">')
        
        for action in risk_matrix.mitigation_plan:
            if action.startswith('**') and action.endswith(':**'):
                html.append(f'<h5>{action[2:-3]}</h5>')
            else:
                html.append(f'<div class="mitigation-item">{action}</div>')
        
        html.append('</div>')
        html.append('</div>')
        
        # Plano de monitoramento
        html.append('<div class="monitoring-plan">')
        html.append('<h4>📊 Plano de Monitoramento</h4>')
        html.append('<div class="monitoring-actions">')
        
        for action in risk_matrix.monitoring_plan:
            if action.startswith('**') and action.endswith(':**'):
                html.append(f'<h5>{action[2:-3]}</h5>')
            elif action.strip():
                html.append(f'<div class="monitoring-item">{action}</div>')
        
        html.append('</div>')
        html.append('</div>')
        
        html.append('</div>')
        
        return '\n'.join(html)
    
    def _get_risks_for_cell(self, risks: List[Risk], prob_level: str, impact_level: str) -> List[Risk]:
        """Retorna riscos que se encaixam em uma célula específica da matriz"""
        prob_ranges = {'Baixa': (0, 0.33), 'Média': (0.33, 0.67), 'Alta': (0.67, 1.0)}
        impact_ranges = {'Baixo': (0, 0.33), 'Médio': (0.33, 0.67), 'Alto': (0.67, 1.0)}
        
        prob_min, prob_max = prob_ranges[prob_level]
        impact_min, impact_max = impact_ranges[impact_level]
        
        return [r for r in risks 
                if prob_min <= r.probability < prob_max and impact_min <= r.impact < impact_max]
    
    def _get_cell_class(self, prob_level: str, impact_level: str) -> str:
        """Retorna classe CSS para célula da matriz"""
        risk_level_map = {
            ('Baixa', 'Baixo'): 'low',
            ('Baixa', 'Médio'): 'low',
            ('Baixa', 'Alto'): 'medium',
            ('Média', 'Baixo'): 'low',
            ('Média', 'Médio'): 'medium',
            ('Média', 'Alto'): 'high',
            ('Alta', 'Baixo'): 'medium',
            ('Alta', 'Médio'): 'high',
            ('Alta', 'Alto'): 'critical'
        }
        
        return f"risk-{risk_level_map.get((prob_level, impact_level), 'medium')}"
    
    def generate_risk_css(self) -> str:
        """Gera CSS para visualização da análise de riscos"""
        return """
        <style>
        .risk-analysis {
            margin: 30px 0;
            padding: 25px;
            background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
            border-radius: 12px;
            border-left: 5px solid #e53e3e;
        }
        
        .risk-summary {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 25px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .overall-score {
            text-align: center;
            font-size: 1.2em;
            margin-bottom: 15px;
        }
        
        .overall-score .score {
            font-size: 1.5em;
            font-weight: bold;
            color: #e53e3e;
        }
        
        .risk-distribution {
            text-align: center;
        }
        
        .risk-category-badge {
            display: inline-block;
            padding: 5px 12px;
            margin: 3px;
            border-radius: 15px;
            font-size: 0.9em;
            font-weight: 500;
            color: white;
        }
        
        .category-market { background-color: #3182ce; }
        .category-financial { background-color: #e53e3e; }
        .category-operational { background-color: #38a169; }
        .category-regulatory { background-color: #d69e2e; }
        .category-external { background-color: #805ad5; }
        
        .risk-matrix-visual {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 25px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .matrix-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 15px;
        }
        
        .matrix-cell {
            padding: 15px;
            border-radius: 5px;
            min-height: 100px;
            border: 2px solid #e2e8f0;
        }
        
        .matrix-cell.risk-low { background-color: #f0fff4; border-color: #38a169; }
        .matrix-cell.risk-medium { background-color: #fffbf0; border-color: #d69e2e; }
        .matrix-cell.risk-high { background-color: #fff5f5; border-color: #e53e3e; }
        .matrix-cell.risk-critical { background-color: #4a0e0e; color: white; border-color: #742a2a; }
        
        .cell-header {
            font-weight: bold;
            font-size: 0.9em;
            margin-bottom: 8px;
            text-align: center;
        }
        
        .cell-risks {
            font-size: 0.8em;
        }
        
        .risk-item {
            background-color: rgba(0,0,0,0.1);
            padding: 3px 6px;
            margin: 2px 0;
            border-radius: 3px;
        }
        
        .more-risks {
            font-style: italic;
            color: #666;
            text-align: center;
            margin-top: 5px;
        }
        
        .priority-risks {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 25px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .priority-risk-item {
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 5px;
            border-left: 5px solid #6c757d;
        }
        
        .priority-risk-item.severity-critical {
            background-color: #f8d7da;
            border-left-color: #dc3545;
        }
        
        .priority-risk-item.severity-high {
            background-color: #fff3cd;
            border-left-color: #ffc107;
        }
        
        .priority-risk-item.severity-medium {
            background-color: #d1ecf1;
            border-left-color: #17a2b8;
        }
        
        .priority-risk-item.severity-low {
            background-color: #d4edda;
            border-left-color: #28a745;
        }
        
        .risk-score {
            font-size: 0.9em;
            color: #666;
            font-weight: normal;
        }
        
        .risk-details {
            display: flex;
            gap: 15px;
            margin-top: 10px;
            font-size: 0.9em;
        }
        
        .risk-details span {
            padding: 3px 8px;
            background-color: rgba(0,0,0,0.1);
            border-radius: 3px;
        }
        
        .mitigation-plan, .monitoring-plan {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 25px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .mitigation-actions, .monitoring-actions {
            margin-top: 15px;
        }
        
        .mitigation-item, .monitoring-item {
            padding: 8px 0;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .mitigation-item:last-child, .monitoring-item:last-child {
            border-bottom: none;
        }
        
        .mitigation-plan h5, .monitoring-plan h5 {
            color: #2d3748;
            margin: 15px 0 10px 0;
            font-size: 1.1em;
        }
        </style>
        """
    
    def export_risk_analysis_json(self, risk_matrix: RiskMatrix) -> str:
        """Exporta análise de riscos em JSON"""
        data = {
            'risks': [asdict(risk) for risk in risk_matrix.risks],
            'overall_risk_score': risk_matrix.overall_risk_score,
            'risk_distribution': risk_matrix.risk_distribution,
            'priority_risks': [asdict(risk) for risk in risk_matrix.priority_risks],
            'mitigation_plan': risk_matrix.mitigation_plan,
            'monitoring_plan': risk_matrix.monitoring_plan,
            'export_date': datetime.now().isoformat()
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

# Instância global do analisador
risk_analyzer = RiskAnalyzer()