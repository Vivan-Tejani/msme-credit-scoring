import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict


class FraudDetector:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_directors: Dict[str, List[str]] = {}
        
    def build_graph(self, transactions_data: List[Dict], director_links: Optional[Dict[str, List[str]]] = None):
        self.graph.clear()
        self.entity_directors = director_links or {}
        
        for txn in transactions_data:
            sender = txn.get("sender_gstin", "")
            receiver = txn.get("receiver_gstin", "")
            amount = txn.get("amount", 0.0)
            date = txn.get("date", "")
            
            if sender and receiver and sender != receiver:
                if self.graph.has_edge(sender, receiver):
                    self.graph[sender][receiver]["weight"] += amount
                    self.graph[sender][receiver]["count"] += 1
                else:
                    self.graph.add_edge(sender, receiver, weight=amount, count=1, date=date)
        
        if director_links:
            for entity, directors in director_links.items():
                for other_entity, other_directors in director_links.items():
                    if entity != other_entity:
                        shared = set(directors) & set(other_directors)
                        if shared:
                            if not self.graph.has_edge(entity, other_entity):
                                self.graph.add_edge(entity, other_entity, weight=0, count=0, link_type="director")
                            if not self.graph.has_edge(other_entity, entity):
                                self.graph.add_edge(other_entity, entity, weight=0, count=0, link_type="director")
    
    def detect_cycles(self, target_gstin: str, max_length: int = 8) -> List[List[str]]:
        if target_gstin not in self.graph:
            return []
        
        ego_nodes = set([target_gstin])
        for neighbor in nx.single_source_shortest_path_length(self.graph, target_gstin, cutoff=2):
            ego_nodes.add(neighbor)
        
        ego_graph = self.graph.subgraph(ego_nodes)
        cycles = []
        
        try:
            for cycle in nx.simple_cycles(ego_graph):
                if target_gstin in cycle and 3 <= len(cycle) <= max_length:
                    cycles.append(cycle)
                    if len(cycles) >= 10:
                        break
        except nx.NetworkXNoCycle:
            pass
        
        return cycles
    
    def detect_communities(self, target_gstin: str) -> Dict:
        if target_gstin not in self.graph:
            return {"community_id": None, "community_size": 0, "density": 0.0}
        
        ego_nodes = set([target_gstin])
        for neighbor in nx.single_source_shortest_path_length(self.graph, target_gstin, cutoff=2):
            ego_nodes.add(neighbor)
        
        if len(ego_nodes) < 3:
            return {"community_id": target_gstin, "community_size": 1, "density": 0.0}
        
        ego_graph = self.graph.subgraph(ego_nodes).to_undirected()
        
        try:
            communities = nx.community.louvain_communities(ego_graph, seed=42)
            for idx, comm in enumerate(communities):
                if target_gstin in comm:
                    subgraph = ego_graph.subgraph(comm)
                    n_nodes = len(comm)
                    n_edges = subgraph.number_of_edges()
                    max_edges = n_nodes * (n_nodes - 1) / 2 if n_nodes > 1 else 1
                    density = n_edges / max_edges if max_edges > 0 else 0.0
                    return {
                        "community_id": f"comm_{idx}",
                        "community_size": n_nodes,
                        "density": round(density, 4),
                    }
        except Exception:
            pass
        
        return {"community_id": target_gstin, "community_size": 1, "density": 0.0}
    
    def compute_pagerank(self, target_gstin: str) -> float:
        if target_gstin not in self.graph or self.graph.number_of_nodes() < 3:
            return 0.0
        
        try:
            pr = nx.pagerank(self.graph, weight="weight")
            return round(pr.get(target_gstin, 0.0), 6)
        except Exception:
            return 0.0
    
    def detect_velocity_spike(self, transactions: pd.DataFrame, window_days: int = 14) -> bool:
        if transactions.empty or len(transactions) < 30:
            return False
        
        transactions["date"] = pd.to_datetime(transactions["date"])
        transactions = transactions.sort_values("date")
        
        daily = transactions.groupby("date")["amount"].sum().reset_index()
        if len(daily) < window_days * 2:
            return False
        
        recent = daily["amount"].iloc[-window_days:].mean()
        previous = daily["amount"].iloc[-(window_days*2):-window_days].mean()
        
        if previous > 0 and recent > previous * 3:
            return True
        return False
    
    def analyze(self, gstin: str, transactions_data: List[Dict], upi_df: Optional[pd.DataFrame] = None, mca_data: Optional[Dict] = None) -> Dict:
        self.build_graph(transactions_data)
        
        cycles = self.detect_cycles(gstin)
        community = self.detect_communities(gstin)
        pagerank = self.compute_pagerank(gstin)
        
        velocity_spike = False
        if upi_df is not None and not upi_df.empty:
            velocity_spike = self.detect_velocity_spike(upi_df)
        
        director_density = 0.0
        if mca_data and "directors" in mca_data:
            directors = mca_data["directors"]
            total_other = sum(d.get("other_companies", 0) for d in directors)
            director_density = min(1.0, total_other / max(len(directors) * 5, 1))
        
        fraud_score = 0.0
        reasons = []
        circular_entities = []
        
        if len(cycles) > 0:
            fraud_score += 0.4
            reasons.append(f"Circular transaction detected: {len(cycles)} cycle(s) involving {gstin}")
            for cycle in cycles[:3]:
                for entity in cycle:
                    if entity != gstin and entity not in circular_entities:
                        circular_entities.append(entity)
        
        if community.get("density", 0.0) > 0.7 and community.get("community_size", 0) > 3:
            fraud_score += 0.25
            reasons.append(f"High-density community detected: {community['community_size']} entities, density {community['density']}")
        
        if pagerank > 0.15:
            fraud_score += 0.15
            reasons.append(f"Unusual network centrality: PageRank {pagerank}")
        
        if velocity_spike:
            fraud_score += 0.15
            reasons.append("Sudden transaction velocity spike detected (3x increase in 14 days)")
        
        if director_density > 0.5:
            fraud_score += 0.05
            reasons.append("Dense director network suggesting shell company links")
        
        fraud_score = min(1.0, fraud_score)
        
        if fraud_score >= 0.7:
            action = "DECLINE"
        elif fraud_score >= 0.4:
            action = "ESCALATE"
        elif fraud_score >= 0.2:
            action = "ENHANCED_REVIEW"
        else:
            action = "APPROVE"
        
        return {
            "fraud_flag": fraud_score >= 0.4,
            "fraud_score": round(fraud_score, 4),
            "fraud_reasons": reasons,
            "circular_entities": circular_entities,
            "recommended_action": action,
            "community": community,
            "pagerank": pagerank,
            "cycle_count": len(cycles),
        }
    
    def get_network(self, gstin: str, radius: int = 2) -> Tuple[List[Dict], List[Dict]]:
        if gstin not in self.graph:
            return [], []
        
        nodes = set([gstin])
        
        for node, dist in nx.single_source_shortest_path_length(self.graph, gstin, cutoff=radius).items():
            nodes.add(node)
        
        subgraph = self.graph.subgraph(nodes)
        
        node_list = []
        for node in subgraph.nodes():
            in_weight = sum(
                data.get("weight", 0) 
                for u, v, data in subgraph.in_edges(node, data=True) #type:ignore
            )
            out_weight = sum(
                data.get("weight", 0) 
                for u, v, data in subgraph.out_edges(node, data=True) #type:ignore
            )
            node_list.append({
                "id": node,
                "type": "target" if node == gstin else "related",
                "volume": round(in_weight + out_weight, 2),
                "is_fraudulent": False,
            })
        
        edge_list = []
        for u, v, data in subgraph.edges(data=True):
            edge_list.append({
                "source": u,
                "target": v,
                "weight": round(data.get("weight", 0), 2),
            })
        
        return node_list, edge_list