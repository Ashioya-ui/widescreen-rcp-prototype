"""
WideScreen RCP Core
Recursive Consensus Protocol implementation
"""

import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class RCPGenome:
    """The evolvable protocol genome"""
    agent_roles: List[str]
    interaction_rules: List[str]
    verification_threshold: float
    novelty_threshold: float
    fitness_score: float = 0.0
    generation: int = 0
    
    def __post_init__(self):
        self.created_at = datetime.now().isoformat()
        
    def mutate(self, rate: float = 0.15) -> 'RCPGenome':
        """Create a mutated variant"""
        new_genome = RCPGenome(
            agent_roles=self.agent_roles.copy(),
            interaction_rules=self.interaction_rules.copy(),
            verification_threshold=self.verification_threshold,
            novelty_threshold=self.novelty_threshold,
            generation=self.generation + 1
        )
        
        # Mutate verification threshold
        if np.random.random() < rate:
            delta = np.random.uniform(-0.02, 0.02)
            new_genome.verification_threshold = np.clip(
                self.verification_threshold + delta, 0.001, 0.1
            )
        
        # Mutate novelty threshold
        if np.random.random() < rate:
            delta = np.random.uniform(-0.1, 0.1)
            new_genome.novelty_threshold = np.clip(
                self.novelty_threshold + delta, 0.5, 0.95
            )
        
        # Mutate interaction rules
        if np.random.random() < rate:
            available_rules = ['debate', 'voting', 'replication', 'mutation', 'fusion']
            new_genome.interaction_rules = list(np.random.choice(
                available_rules, size=3, replace=False
            ))
        
        return new_genome
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RCPGenome':
        """Deserialize from dictionary"""
        return cls(**{k: v for k, v in data.items() if k != 'created_at'})


class RCPEvolutionEngine:
    """Handles genome evolution and selection"""
    
    def __init__(self, log_file: str = 'logs/rcp_evolution.log'):
        self.history: List[RCPGenome] = []
        self.log_file = log_file
        self._setup_logging()
    
    def _setup_logging(self):
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def evolve(self, 
               current_genome: RCPGenome,
               fitness_function: callable,
               variants_count: int = 5) -> Tuple[RCPGenome, float]:
        """
        Evolve genome through mutation and selection
        
        Args:
            current_genome: Current protocol genome
            fitness_function: Function to evaluate fitness
            variants_count: Number of variants to generate
            
        Returns:
            Tuple of (best_genome, best_fitness)
        """
        # Generate variants
        variants = [current_genome.mutate() for _ in range(variants_count)]
        
        # Evaluate fitness for each variant
        fitness_scores = []
        for variant in variants:
            try:
                score = fitness_function(variant)
                variant.fitness_score = score
                fitness_scores.append(score)
            except Exception as e:
                logging.error(f"Fitness evaluation failed: {e}")
                fitness_scores.append(-np.inf)
        
        # Select best variant
        best_idx = np.argmax(fitness_scores)
        best_genome = variants[best_idx]
        best_fitness = fitness_scores[best_idx]
        
        # Log evolution
        self.history.append(best_genome)
        logging.info(
            f"Evolution Gen {best_genome.generation}: "
            f"Fitness={best_fitness:.4f}, "
            f"Rules={best_genome.interaction_rules}, "
            f"Threshold={best_genome.verification_threshold:.4f}"
        )
        
        return best_genome, best_fitness
    
    def save_history(self, filepath: str = 'data/evolution_history.json'):
        """Save evolution history"""
        with open(filepath, 'w') as f:
            json.dump([g.to_dict() for g in self.history], f, indent=2)
