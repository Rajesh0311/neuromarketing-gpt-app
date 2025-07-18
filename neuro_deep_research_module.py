#!/usr/bin/env python3
"""
Deep Research Engine for NeuroMarketing GPT Platform
===================================================

This module provides comprehensive research capabilities including:
- Multi-source dataset integration (OpenNeuro, Zenodo, PhysioNet, IEEE DataPort)
- Literature search and analysis
- Research data synthesis
- Academic paper analysis
- Research export capabilities

Authors: NeuroMarketing GPT Team
Version: 1.0.0
License: MIT
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging
import asyncio
import aiohttp
from functools import lru_cache
import time
import re
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResearchDataset:
    """Represents a research dataset"""
    id: str
    title: str
    description: str
    source: str
    authors: List[str]
    publication_date: datetime
    subject_count: int
    data_type: str
    file_formats: List[str]
    download_url: Optional[str]
    relevance_score: float
    tags: List[str]

@dataclass
class ResearchPaper:
    """Represents a research paper"""
    id: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    publication_date: datetime
    doi: Optional[str]
    keywords: List[str]
    citation_count: int
    relevance_score: float
    url: Optional[str]

@dataclass
class ResearchSynthesis:
    """Synthesis of research findings"""
    query: str
    datasets_found: int
    papers_found: int
    key_findings: List[str]
    methodology_summary: Dict[str, int]
    sample_size_stats: Dict[str, float]
    publication_trends: Dict[str, int]
    recommendations: List[str]
    confidence_score: float
    timestamp: datetime

class NeuroResearchModule:
    """Main research module for accessing multiple data sources"""
    
    def __init__(self):
        self.data_sources = self._initialize_data_sources()
        self.search_history = []
        self.cache_timeout = 3600  # 1 hour cache
        self.cache = {}
        
    def _initialize_data_sources(self) -> Dict[str, Dict[str, Any]]:
        """Initialize data source configurations"""
        return {
            'openneuro': {
                'name': 'OpenNeuro',
                'base_url': 'https://openneuro.org/crn/graphql',
                'api_type': 'graphql',
                'requires_auth': False,
                'status': 'active',
                'description': 'Open neuroimaging data sharing platform'
            },
            'zenodo': {
                'name': 'Zenodo',
                'base_url': 'https://zenodo.org/api/records',
                'api_type': 'rest',
                'requires_auth': False,
                'status': 'active',
                'description': 'Open-access repository for research data'
            },
            'physionet': {
                'name': 'PhysioNet',
                'base_url': 'https://physionet.org/content',
                'api_type': 'rest',
                'requires_auth': False,
                'status': 'active',
                'description': 'Physiological signal databases'
            },
            'ieee_dataport': {
                'name': 'IEEE DataPort',
                'base_url': 'https://ieee-dataport.org/api',
                'api_type': 'rest',
                'requires_auth': False,
                'status': 'limited',
                'description': 'IEEE research data repository'
            },
            'pubmed': {
                'name': 'PubMed',
                'base_url': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils',
                'api_type': 'rest',
                'requires_auth': False,
                'status': 'active',
                'description': 'Biomedical literature database'
            }
        }
    
    @lru_cache(maxsize=100)
    def search_datasets(self, query: str, sources: List[str] = None, 
                       limit: int = 50) -> List[ResearchDataset]:
        """
        Search for datasets across multiple sources
        
        Args:
            query: Search query
            sources: List of data sources to search (default: all)
            limit: Maximum number of results per source
            
        Returns:
            List of ResearchDataset objects
        """
        if sources is None:
            sources = list(self.data_sources.keys())
        
        logger.info(f"Searching datasets for query: '{query}' in sources: {sources}")
        
        all_datasets = []
        
        for source in sources:
            try:
                if source == 'openneuro':
                    datasets = self._search_openneuro(query, limit)
                elif source == 'zenodo':
                    datasets = self._search_zenodo(query, limit)
                elif source == 'physionet':
                    datasets = self._search_physionet(query, limit)
                elif source == 'ieee_dataport':
                    datasets = self._search_ieee_dataport(query, limit)
                else:
                    logger.warning(f"Unknown data source: {source}")
                    continue
                
                all_datasets.extend(datasets)
                logger.info(f"Found {len(datasets)} datasets in {source}")
                
            except Exception as e:
                logger.error(f"Error searching {source}: {str(e)}")
                # Continue with other sources even if one fails
                continue
        
        # Sort by relevance score
        all_datasets.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return all_datasets
    
    def _search_openneuro(self, query: str, limit: int) -> List[ResearchDataset]:
        """Search OpenNeuro datasets using GraphQL API"""
        datasets = []
        
        # Simulate OpenNeuro search results - in production, use actual GraphQL API
        sample_datasets = [
            {
                'id': 'ds000001',
                'title': 'Consumer choice EEG study',
                'description': 'EEG recordings during consumer decision making tasks',
                'authors': ['Smith, J.', 'Johnson, M.'],
                'subject_count': 24,
                'data_type': 'EEG',
                'tags': ['consumer', 'choice', 'decision-making', 'EEG']
            },
            {
                'id': 'ds000002', 
                'title': 'Brand recognition fMRI study',
                'description': 'fMRI data from brand logo recognition experiment',
                'authors': ['Williams, K.', 'Brown, L.'],
                'subject_count': 32,
                'data_type': 'fMRI',
                'tags': ['brand', 'recognition', 'fMRI', 'marketing']
            },
            {
                'id': 'ds000003',
                'title': 'Emotional response to advertisements',
                'description': 'EEG and physiological responses to video advertisements',
                'authors': ['Davis, R.', 'Miller, S.'],
                'subject_count': 18,
                'data_type': 'EEG+Physio',
                'tags': ['emotion', 'advertisement', 'EEG', 'physiology']
            }
        ]
        
        for sample in sample_datasets:
            # Calculate relevance score based on query match
            relevance = self._calculate_relevance(query, sample['title'] + ' ' + sample['description'])
            
            if relevance > 0.1:  # Minimum relevance threshold
                dataset = ResearchDataset(
                    id=sample['id'],
                    title=sample['title'],
                    description=sample['description'],
                    source='OpenNeuro',
                    authors=sample['authors'],
                    publication_date=datetime.now() - timedelta(days=np.random.randint(30, 1000)),
                    subject_count=sample['subject_count'],
                    data_type=sample['data_type'],
                    file_formats=['BIDS', 'NIfTI', 'JSON'],
                    download_url=f"https://openneuro.org/datasets/{sample['id']}",
                    relevance_score=relevance,
                    tags=sample['tags']
                )
                datasets.append(dataset)
        
        return datasets[:limit]
    
    def _search_zenodo(self, query: str, limit: int) -> List[ResearchDataset]:
        """Search Zenodo for research datasets"""
        datasets = []
        
        # Simulate Zenodo search results
        sample_datasets = [
            {
                'id': 'zenodo.123456',
                'title': 'Neuromarketing EEG database',
                'description': 'Comprehensive EEG dataset for neuromarketing research',
                'authors': ['Anderson, P.', 'Taylor, M.'],
                'subject_count': 45,
                'data_type': 'EEG',
                'tags': ['neuromarketing', 'EEG', 'consumer', 'database']
            },
            {
                'id': 'zenodo.234567',
                'title': 'Eye-tracking consumer behavior data',
                'description': 'Eye-tracking data from shopping behavior studies',
                'authors': ['Wilson, K.', 'Garcia, L.'],
                'subject_count': 67,
                'data_type': 'Eye-tracking',
                'tags': ['eye-tracking', 'shopping', 'behavior', 'consumer']
            }
        ]
        
        for sample in sample_datasets:
            relevance = self._calculate_relevance(query, sample['title'] + ' ' + sample['description'])
            
            if relevance > 0.1:
                dataset = ResearchDataset(
                    id=sample['id'],
                    title=sample['title'],
                    description=sample['description'],
                    source='Zenodo',
                    authors=sample['authors'],
                    publication_date=datetime.now() - timedelta(days=np.random.randint(30, 800)),
                    subject_count=sample['subject_count'],
                    data_type=sample['data_type'],
                    file_formats=['CSV', 'MAT', 'HDF5'],
                    download_url=f"https://zenodo.org/record/{sample['id'].split('.')[1]}",
                    relevance_score=relevance,
                    tags=sample['tags']
                )
                datasets.append(dataset)
        
        return datasets[:limit]
    
    def _search_physionet(self, query: str, limit: int) -> List[ResearchDataset]:
        """Search PhysioNet databases"""
        datasets = []
        
        # Simulate PhysioNet search results
        sample_datasets = [
            {
                'id': 'physionet.001',
                'title': 'Consumer EEG Response Database',
                'description': 'EEG signals recorded during consumer product evaluation',
                'authors': ['Martinez, A.', 'Rodriguez, C.'],
                'subject_count': 30,
                'data_type': 'EEG',
                'tags': ['EEG', 'consumer', 'evaluation', 'response']
            },
            {
                'id': 'physionet.002',
                'title': 'Stress Response in Marketing Scenarios',
                'description': 'Physiological stress markers during marketing exposure',
                'authors': ['Thompson, J.', 'Lee, S.'],
                'subject_count': 22,
                'data_type': 'Physio',
                'tags': ['stress', 'marketing', 'physiology', 'response']
            },
            {
                'id': 'physionet.003',
                'title': 'Heart Rate Variability in Shopping',
                'description': 'HRV measurements during shopping decision tasks',
                'authors': ['Clark, M.', 'White, D.'],
                'subject_count': 28,
                'data_type': 'ECG',
                'tags': ['HRV', 'shopping', 'decision', 'ECG']
            }
        ]
        
        for sample in sample_datasets:
            relevance = self._calculate_relevance(query, sample['title'] + ' ' + sample['description'])
            
            if relevance > 0.1:
                dataset = ResearchDataset(
                    id=sample['id'],
                    title=sample['title'],
                    description=sample['description'],
                    source='PhysioNet',
                    authors=sample['authors'],
                    publication_date=datetime.now() - timedelta(days=np.random.randint(60, 1200)),
                    subject_count=sample['subject_count'],
                    data_type=sample['data_type'],
                    file_formats=['WFDB', 'EDF', 'CSV'],
                    download_url=f"https://physionet.org/content/{sample['id']}",
                    relevance_score=relevance,
                    tags=sample['tags']
                )
                datasets.append(dataset)
        
        return datasets[:limit]
    
    def _search_ieee_dataport(self, query: str, limit: int) -> List[ResearchDataset]:
        """Search IEEE DataPort"""
        datasets = []
        
        # Simulate IEEE DataPort search results
        sample_datasets = [
            {
                'id': 'ieee.001',
                'title': 'Neural Networks for Consumer Prediction',
                'description': 'Dataset for training neural networks in consumer behavior prediction',
                'authors': ['Kumar, R.', 'Patel, N.'],
                'subject_count': 1000,
                'data_type': 'Behavioral',
                'tags': ['neural-networks', 'prediction', 'behavior', 'consumer']
            },
            {
                'id': 'ieee.002',
                'title': 'Machine Learning Marketing Dataset',
                'description': 'Curated dataset for machine learning in marketing applications',
                'authors': ['Zhang, L.', 'Wang, Y.'],
                'subject_count': 500,
                'data_type': 'ML-Features',
                'tags': ['machine-learning', 'marketing', 'features', 'dataset']
            },
            {
                'id': 'ieee.003',
                'title': 'Sentiment Analysis Training Data',
                'description': 'Labeled dataset for sentiment analysis in marketing context',
                'authors': ['Singh, A.', 'Jones, B.'],
                'subject_count': 2000,
                'data_type': 'Text',
                'tags': ['sentiment', 'analysis', 'text', 'marketing']
            }
        ]
        
        for sample in sample_datasets:
            relevance = self._calculate_relevance(query, sample['title'] + ' ' + sample['description'])
            
            if relevance > 0.1:
                dataset = ResearchDataset(
                    id=sample['id'],
                    title=sample['title'],
                    description=sample['description'],
                    source='IEEE DataPort',
                    authors=sample['authors'],
                    publication_date=datetime.now() - timedelta(days=np.random.randint(30, 600)),
                    subject_count=sample['subject_count'],
                    data_type=sample['data_type'],
                    file_formats=['CSV', 'JSON', 'Excel'],
                    download_url=f"https://ieee-dataport.org/data/{sample['id']}",
                    relevance_score=relevance,
                    tags=sample['tags']
                )
                datasets.append(dataset)
        
        return datasets[:limit]
    
    def search_literature(self, query: str, limit: int = 20) -> List[ResearchPaper]:
        """
        Search academic literature using PubMed and other sources
        
        Args:
            query: Search query
            limit: Maximum number of papers to return
            
        Returns:
            List of ResearchPaper objects
        """
        logger.info(f"Searching literature for: '{query}'")
        
        papers = []
        
        # Simulate PubMed search results
        sample_papers = [
            {
                'id': 'pmid_001',
                'title': 'Neural correlates of consumer decision making in neuromarketing',
                'abstract': 'This study investigates the neural mechanisms underlying consumer decision-making processes using fMRI...',
                'authors': ['Smith, J.A.', 'Johnson, B.C.', 'Williams, D.E.'],
                'journal': 'NeuroImage',
                'doi': '10.1016/j.neuroimage.2023.001',
                'keywords': ['neuromarketing', 'fMRI', 'decision-making', 'consumer behavior'],
                'citation_count': 45
            },
            {
                'id': 'pmid_002',
                'title': 'EEG-based analysis of emotional responses to advertising stimuli',
                'abstract': 'We examined EEG responses to different types of advertising content to understand emotional engagement...',
                'authors': ['Brown, K.L.', 'Davis, R.M.'],
                'journal': 'Journal of Consumer Psychology',
                'doi': '10.1016/j.jcps.2023.002',
                'keywords': ['EEG', 'emotion', 'advertising', 'engagement'],
                'citation_count': 32
            },
            {
                'id': 'pmid_003',
                'title': 'Machine learning approaches in neuromarketing research',
                'abstract': 'This review examines current machine learning techniques applied to neuromarketing data analysis...',
                'authors': ['Garcia, M.P.', 'Anderson, L.T.', 'Wilson, S.K.'],
                'journal': 'Nature Reviews Neuroscience',
                'doi': '10.1038/nrn.2023.003',
                'keywords': ['machine learning', 'neuromarketing', 'data analysis', 'review'],
                'citation_count': 78
            },
            {
                'id': 'pmid_004',
                'title': 'Eye-tracking studies in consumer behavior: A meta-analysis',
                'abstract': 'Meta-analysis of eye-tracking studies examining visual attention patterns in consumer contexts...',
                'authors': ['Taylor, R.J.', 'Martinez, C.A.'],
                'journal': 'Psychological Science',
                'doi': '10.1177/psci.2023.004',
                'keywords': ['eye-tracking', 'consumer behavior', 'meta-analysis', 'attention'],
                'citation_count': 56
            },
            {
                'id': 'pmid_005',
                'title': 'Cultural differences in neural responses to marketing stimuli',
                'abstract': 'Cross-cultural neuroimaging study examining cultural variations in responses to marketing content...',
                'authors': ['Lee, H.S.', 'Zhang, W.Q.', 'Patel, N.R.'],
                'journal': 'Cultural Cognitive Science',
                'doi': '10.1007/ccs.2023.005',
                'keywords': ['culture', 'neuroimaging', 'marketing', 'cross-cultural'],
                'citation_count': 23
            }
        ]
        
        for sample in sample_papers:
            relevance = self._calculate_relevance(query, sample['title'] + ' ' + sample['abstract'])
            
            if relevance > 0.1:
                paper = ResearchPaper(
                    id=sample['id'],
                    title=sample['title'],
                    abstract=sample['abstract'],
                    authors=sample['authors'],
                    journal=sample['journal'],
                    publication_date=datetime.now() - timedelta(days=np.random.randint(30, 1800)),
                    doi=sample['doi'],
                    keywords=sample['keywords'],
                    citation_count=sample['citation_count'],
                    relevance_score=relevance,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{sample['id'].split('_')[1]}"
                )
                papers.append(paper)
        
        # Sort by relevance and citation count
        papers.sort(key=lambda x: (x.relevance_score * 0.7 + (x.citation_count / 100) * 0.3), reverse=True)
        
        return papers[:limit]
    
    def _calculate_relevance(self, query: str, text: str) -> float:
        """Calculate relevance score between query and text"""
        query_terms = set(query.lower().split())
        text_terms = set(re.findall(r'\w+', text.lower()))
        
        if not query_terms:
            return 0.0
        
        # Calculate overlap
        overlap = len(query_terms.intersection(text_terms))
        relevance = overlap / len(query_terms)
        
        # Boost score for exact phrase matches
        if query.lower() in text.lower():
            relevance += 0.3
        
        return min(relevance, 1.0)
    
    def synthesize_research(self, query: str, datasets: List[ResearchDataset], 
                          papers: List[ResearchPaper]) -> ResearchSynthesis:
        """
        Synthesize research findings from datasets and papers
        
        Args:
            query: Original research query
            datasets: List of relevant datasets
            papers: List of relevant papers
            
        Returns:
            ResearchSynthesis object with consolidated findings
        """
        logger.info("Synthesizing research findings...")
        
        # Analyze methodologies
        methodology_summary = {}
        for dataset in datasets:
            data_type = dataset.data_type
            methodology_summary[data_type] = methodology_summary.get(data_type, 0) + 1
        
        # Calculate sample size statistics
        sample_sizes = [d.subject_count for d in datasets if d.subject_count > 0]
        sample_size_stats = {}
        if sample_sizes:
            sample_size_stats = {
                'total_participants': sum(sample_sizes),
                'average_sample_size': np.mean(sample_sizes),
                'median_sample_size': np.median(sample_sizes),
                'largest_study': max(sample_sizes),
                'smallest_study': min(sample_sizes)
            }
        
        # Analyze publication trends
        publication_trends = {}
        for paper in papers:
            year = paper.publication_date.year
            publication_trends[str(year)] = publication_trends.get(str(year), 0) + 1
        
        # Generate key findings (simplified)
        key_findings = self._generate_key_findings(datasets, papers, query)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(datasets, papers, query)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(datasets, papers)
        
        return ResearchSynthesis(
            query=query,
            datasets_found=len(datasets),
            papers_found=len(papers),
            key_findings=key_findings,
            methodology_summary=methodology_summary,
            sample_size_stats=sample_size_stats,
            publication_trends=publication_trends,
            recommendations=recommendations,
            confidence_score=confidence_score,
            timestamp=datetime.now()
        )
    
    def _generate_key_findings(self, datasets: List[ResearchDataset], 
                              papers: List[ResearchPaper], query: str) -> List[str]:
        """Generate key findings from research analysis"""
        findings = []
        
        # Dataset-based findings
        if datasets:
            total_subjects = sum(d.subject_count for d in datasets)
            data_types = set(d.data_type for d in datasets)
            findings.append(f"Found {len(datasets)} relevant datasets with {total_subjects} total participants")
            findings.append(f"Primary data collection methods include: {', '.join(data_types)}")
        
        # Paper-based findings
        if papers:
            avg_citations = np.mean([p.citation_count for p in papers])
            top_journals = list(set(p.journal for p in papers[:5]))
            findings.append(f"Identified {len(papers)} relevant publications with average {avg_citations:.1f} citations")
            findings.append(f"Top publishing venues: {', '.join(top_journals[:3])}")
        
        # Query-specific insights
        if 'EEG' in query or 'eeg' in query:
            findings.append("EEG appears to be a primary methodology for this research area")
        if 'fMRI' in query or 'fmri' in query:
            findings.append("fMRI neuroimaging is commonly used in this research domain")
        if 'consumer' in query.lower():
            findings.append("Consumer behavior research shows strong integration with neuroscience methods")
        
        return findings
    
    def _generate_recommendations(self, datasets: List[ResearchDataset], 
                                 papers: List[ResearchPaper], query: str) -> List[str]:
        """Generate research recommendations"""
        recommendations = []
        
        if datasets:
            # Data source recommendations
            sources = set(d.source for d in datasets)
            recommendations.append(f"Consider accessing data from: {', '.join(sources)}")
            
            # Sample size recommendations
            sample_sizes = [d.subject_count for d in datasets if d.subject_count > 0]
            if sample_sizes:
                avg_sample = np.mean(sample_sizes)
                recommendations.append(f"Typical sample sizes range from {min(sample_sizes)} to {max(sample_sizes)} participants")
        
        if papers:
            # Methodology recommendations
            recent_papers = [p for p in papers if (datetime.now() - p.publication_date).days < 730]
            if recent_papers:
                recommendations.append("Recent publications suggest emerging methodological approaches")
            
            # High-impact work
            high_impact = [p for p in papers if p.citation_count > 50]
            if high_impact:
                recommendations.append("Consider reviewing highly-cited foundational papers")
        
        # General recommendations
        recommendations.append("Cross-reference findings across multiple data sources for robustness")
        recommendations.append("Consider cultural and demographic factors in study design")
        
        return recommendations
    
    def _calculate_confidence_score(self, datasets: List[ResearchDataset], 
                                   papers: List[ResearchPaper]) -> float:
        """Calculate confidence score for research synthesis"""
        score = 0.0
        
        # Dataset quality factors
        if datasets:
            avg_dataset_relevance = np.mean([d.relevance_score for d in datasets])
            dataset_diversity = len(set(d.source for d in datasets))
            total_subjects = sum(d.subject_count for d in datasets)
            
            score += avg_dataset_relevance * 0.3
            score += min(dataset_diversity / 4, 1.0) * 0.2
            score += min(total_subjects / 1000, 1.0) * 0.2
        
        # Paper quality factors
        if papers:
            avg_paper_relevance = np.mean([p.relevance_score for p in papers])
            avg_citations = np.mean([p.citation_count for p in papers])
            journal_diversity = len(set(p.journal for p in papers))
            
            score += avg_paper_relevance * 0.2
            score += min(avg_citations / 100, 1.0) * 0.1
            score += min(journal_diversity / 5, 1.0) * 0.1
        
        return min(score, 1.0)
    
    def get_source_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all data sources"""
        status = {}
        
        for source_key, source_info in self.data_sources.items():
            # Simulate status check
            response_time = np.random.uniform(50, 200)  # ms
            availability = np.random.choice([True, False], p=[0.95, 0.05])
            
            status[source_key] = {
                'name': source_info['name'],
                'status': 'online' if availability else 'offline',
                'response_time': f"{response_time:.0f}ms",
                'description': source_info['description'],
                'last_checked': datetime.now().isoformat()
            }
        
        return status
    
    def export_research_results(self, synthesis: ResearchSynthesis, 
                               datasets: List[ResearchDataset],
                               papers: List[ResearchPaper], 
                               format_type: str = 'json') -> str:
        """Export research results in various formats"""
        try:
            if format_type.lower() == 'json':
                export_data = {
                    'synthesis': {
                        'query': synthesis.query,
                        'datasets_found': synthesis.datasets_found,
                        'papers_found': synthesis.papers_found,
                        'key_findings': synthesis.key_findings,
                        'methodology_summary': synthesis.methodology_summary,
                        'sample_size_stats': synthesis.sample_size_stats,
                        'publication_trends': synthesis.publication_trends,
                        'recommendations': synthesis.recommendations,
                        'confidence_score': synthesis.confidence_score,
                        'timestamp': synthesis.timestamp.isoformat()
                    },
                    'datasets': [
                        {
                            'id': d.id,
                            'title': d.title,
                            'source': d.source,
                            'authors': d.authors,
                            'subject_count': d.subject_count,
                            'data_type': d.data_type,
                            'relevance_score': d.relevance_score,
                            'download_url': d.download_url
                        } for d in datasets
                    ],
                    'papers': [
                        {
                            'id': p.id,
                            'title': p.title,
                            'authors': p.authors,
                            'journal': p.journal,
                            'citation_count': p.citation_count,
                            'relevance_score': p.relevance_score,
                            'doi': p.doi,
                            'url': p.url
                        } for p in papers
                    ]
                }
                
                return json.dumps(export_data, indent=2)
            
            elif format_type.lower() == 'markdown':
                # Generate markdown report
                md_content = f"""# Research Report: {synthesis.query}

## Summary
- **Datasets Found**: {synthesis.datasets_found}
- **Papers Found**: {synthesis.papers_found}
- **Confidence Score**: {synthesis.confidence_score:.2f}
- **Generated**: {synthesis.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Key Findings
"""
                for finding in synthesis.key_findings:
                    md_content += f"- {finding}\n"
                
                md_content += f"""
## Methodology Summary
"""
                for method, count in synthesis.methodology_summary.items():
                    md_content += f"- **{method}**: {count} studies\n"
                
                if synthesis.sample_size_stats:
                    md_content += f"""
## Sample Size Statistics
- **Total Participants**: {synthesis.sample_size_stats.get('total_participants', 'N/A')}
- **Average Sample Size**: {synthesis.sample_size_stats.get('average_sample_size', 'N/A'):.1f}
- **Largest Study**: {synthesis.sample_size_stats.get('largest_study', 'N/A')}
"""
                
                md_content += f"""
## Recommendations
"""
                for rec in synthesis.recommendations:
                    md_content += f"- {rec}\n"
                
                # Add top datasets
                md_content += f"""
## Top Datasets
"""
                for i, dataset in enumerate(datasets[:5], 1):
                    md_content += f"{i}. **{dataset.title}** ({dataset.source})\n"
                    md_content += f"   - Subjects: {dataset.subject_count}\n"
                    md_content += f"   - Type: {dataset.data_type}\n"
                    md_content += f"   - Relevance: {dataset.relevance_score:.2f}\n\n"
                
                # Add top papers
                md_content += f"""
## Top Papers
"""
                for i, paper in enumerate(papers[:5], 1):
                    md_content += f"{i}. **{paper.title}**\n"
                    md_content += f"   - Journal: {paper.journal}\n"
                    md_content += f"   - Citations: {paper.citation_count}\n"
                    md_content += f"   - Relevance: {paper.relevance_score:.2f}\n\n"
                
                return md_content
            
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            logger.error(f"Error exporting research results: {str(e)}")
            return f"Error: {str(e)}"

# Convenience functions for integration
def search_neuro_research(query: str, sources: List[str] = None) -> Dict[str, Any]:
    """Quick research search function"""
    research_module = NeuroResearchModule()
    
    datasets = research_module.search_datasets(query, sources)
    papers = research_module.search_literature(query)
    synthesis = research_module.synthesize_research(query, datasets, papers)
    
    return {
        'synthesis': synthesis,
        'datasets': datasets,
        'papers': papers
    }

def get_research_insights(query: str) -> Dict[str, Any]:
    """Get research insights summary"""
    results = search_neuro_research(query)
    synthesis = results['synthesis']
    
    return {
        'datasets_found': synthesis.datasets_found,
        'papers_found': synthesis.papers_found,
        'confidence_score': synthesis.confidence_score,
        'key_findings': synthesis.key_findings[:3],  # Top 3 findings
        'recommendations': synthesis.recommendations[:3]  # Top 3 recommendations
    }

# Example usage and testing
if __name__ == "__main__":
    # Test the deep research engine
    research_module = NeuroResearchModule()
    
    test_queries = [
        "EEG consumer behavior",
        "fMRI brand recognition",
        "neuromarketing effectiveness",
        "eye tracking shopping",
        "emotional response advertising"
    ]
    
    print("Deep Research Engine Results:")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        
        # Search datasets and papers
        datasets = research_module.search_datasets(query, limit=5)
        papers = research_module.search_literature(query, limit=5)
        
        print(f"Datasets found: {len(datasets)}")
        for dataset in datasets[:3]:
            print(f"  - {dataset.title} ({dataset.source})")
            print(f"    Relevance: {dataset.relevance_score:.2f}, Subjects: {dataset.subject_count}")
        
        print(f"\nPapers found: {len(papers)}")
        for paper in papers[:3]:
            print(f"  - {paper.title}")
            print(f"    Journal: {paper.journal}, Citations: {paper.citation_count}")
        
        # Generate synthesis
        synthesis = research_module.synthesize_research(query, datasets, papers)
        print(f"\nConfidence Score: {synthesis.confidence_score:.2f}")
        print("Key Findings:")
        for finding in synthesis.key_findings[:2]:
            print(f"  - {finding}")
    
    # Test source status
    print(f"\n\nData Source Status:")
    print("=" * 30)
    
    status = research_module.get_source_status()
    for source, info in status.items():
        print(f"{info['name']}: {info['status']} ({info['response_time']})")