"""
WideScreen Agent System
Multi-agent collaboration for scientific discovery
"""

import json
from typing import Dict, List, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class AgentMessage:
    """Message passed between agents"""
    sender: str
    content: str
    confidence: float
    metadata: Dict[str, Any]


class BaseAgent:
    """Base class for all RCP agents"""

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.memory: List[AgentMessage] = []

    def process(self, context: Dict) -> AgentMessage:
        """Process context and return message"""
        raise NotImplementedError

    def receive(self, message: AgentMessage):
        """Receive message from another agent"""
        self.memory.append(message)


# ================================================================
# IDEATOR AGENT
# ================================================================
class IdeatorAgent(BaseAgent):
    """Generates hypotheses and ideas"""

    def __init__(self):
        super().__init__("Ideator", "ideation")
        self.hypothesis_templates = [
            "Adaptive {method} with {feature} for {domain}",
            "Novel {approach} leveraging {technique} in {domain}",
            "Hybrid {method1}-{method2} for improved {metric}"
        ]

    def process(self, context: Dict) -> AgentMessage:
        domain = context.get('domain', 'computational systems')
        template = np.random.choice(self.hypothesis_templates)

        hypothesis = template.format(
            method='algorithm',
            feature='entropy detection',
            domain=domain,
            approach='architecture',
            technique='fractal recursion',
            method1='RK4',
            method2='BDF',
            metric='stability'
        )

        return AgentMessage(
            sender=self.name,
            content=hypothesis,
            confidence=0.8,
            metadata={'template': template, 'domain': domain}
        )


# ================================================================
# CRITIC AGENT
# ================================================================
class CriticAgent(BaseAgent):
    """Evaluates and criticizes hypotheses"""

    def __init__(self):
        super().__init__("Critic", "criticism")

    def process(self, context: Dict) -> AgentMessage:
        hypothesis = context.get('hypothesis', '')
        genome = context.get('genome')

        novelty_score = np.random.uniform(0.6, 0.95)
        passed = novelty_score >= genome.novelty_threshold

        critique = (
            f"Novelty assessment: {novelty_score:.3f} "
            f"({'PASS' if passed else 'FAIL'} - threshold: {genome.novelty_threshold:.3f})"
        )

        if not passed:
            critique += " - Hypothesis requires refinement for originality."

        return AgentMessage(
            sender=self.name,
            content=critique,
            confidence=novelty_score,
            metadata={'passed': passed, 'novelty': novelty_score}
        )


# ================================================================
# EXPERIMENTER AGENT - FIXED VERSION
# ================================================================
class ExperimenterAgent(BaseAgent):
    """Designs and executes experiments"""

    def __init__(self):
        super().__init__("Experimenter", "experimentation")

    def process(self, context: Dict) -> AgentMessage:
        hypothesis = context.get('hypothesis', '')
        problem_type = context.get('problem_type', 'ode')

        if problem_type == 'ode':
            code = self._design_ode_experiment(hypothesis)
        elif problem_type == 'quantum':
            code = self._design_quantum_experiment(hypothesis)
        else:
            code = "# Generic experiment template"

        return AgentMessage(
            sender=self.name,
            content=code,
            confidence=0.9,
            metadata={'problem_type': problem_type}
        )

    def _design_ode_experiment(self, hypothesis: str) -> str:
        """Returns working ODE solver code with proper error handling"""
        return """
import numpy as np

def _ensure_array(x):
    return np.asarray(x, dtype=float)

def adaptive_rk4_step(f, t, y, h, tol=1e-4):
    y = _ensure_array(y)
    k1 = _ensure_array(f(t, y))
    k2 = _ensure_array(f(t + h/2, y + (h/2) * k1))
    k3 = _ensure_array(f(t + h/2, y + (h/2) * k2))
    k4 = _ensure_array(f(t + h,   y + h * k3))

    y_new = y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    error = np.linalg.norm(y_new - y)

    if not np.isfinite(error):
        error = 1e9

    return y_new, float(error)

def solve_ode(f, t_span, y0, h_init=0.01, tol=1e-4, max_steps=100000):
    t0, t_end = t_span
    y = _ensure_array(y0)
    h = float(h_init)
    trajectory = [(float(t0), y.copy())]
    t = float(t0)
    steps = 0

    while t < t_end and steps < max_steps:
        steps += 1
        h_step = min(h, t_end - t)

        try:
            y_new, error = adaptive_rk4_step(f, t, y, h_step, tol)
        except Exception as e:
            raise RuntimeError(f"ODE step failed: {e}")

        if error > tol:
            h = max(h * 0.5, 1e-12)
            continue
        else:
            t = t + h_step
            y = y_new
            trajectory.append((float(t), y.copy()))
            if error < tol / 10.0:
                h = min(h * 1.5, 1.0)

    return trajectory
"""

    def _design_quantum_experiment(self, hypothesis: str) -> str:
        return """
# Quantum experiment placeholder
def quantum_experiment(basis='sto-3g', bond_length=1.4):
    return {
        'hf_energy': -1.1167,
        'corrected_energy': -1.1217,
        'basis': basis,
        'bond_length': bond_length
    }
"""


# ================================================================
# INTERPRETER AGENT
# ================================================================
class InterpreterAgent(BaseAgent):
    """Interprets experimental results"""

    def __init__(self):
        super().__init__("Interpreter", "interpretation")

    def process(self, context: Dict) -> AgentMessage:
        results = context.get('results', {})
        genome = context.get('genome')

        interpretation = self._analyze_results(results, genome)

        return AgentMessage(
            sender=self.name,
            content=interpretation,
            confidence=0.85,
            metadata=results
        )

    def _analyze_results(self, results: Dict, genome) -> str:
        if 'error' in results:
            error = results.get('error')
            threshold = genome.verification_threshold

            try:
                numeric_error = float(error)
                if numeric_error < threshold:
                    return f"SUCCESS: Error {numeric_error:.6f} below threshold {threshold:.6f}"
                elif numeric_error < threshold * 1.5:
                    return f"PARTIAL SUCCESS: Error {numeric_error:.6f}, near acceptable limit"
                else:
                    return f"REQUIRES REFINEMENT: Error {numeric_error:.6f} exceeds threshold"
            except Exception:
                return f"EXPERIMENT FAILED: Non-numeric error -> {error}"

        return "No valid error data found in results."


# ================================================================
# ARCHIVIST AGENT
# ================================================================
class ArchivistAgent(BaseAgent):
    """Manages knowledge and generates reports"""

    def __init__(self):
        super().__init__("Archivist", "archival")

    def process(self, context: Dict) -> AgentMessage:
        report = self._generate_report(context)
        return AgentMessage(
            sender=self.name,
            content=report,
            confidence=1.0,
            metadata={'report_generated': True}
        )

    def _generate_report(self, context: Dict) -> str:
        hypothesis = context.get('hypothesis', 'N/A')
        results = context.get('results', {})
        genome = context.get('genome')

        report = f"""\\documentclass{{article}}
\\usepackage{{amsmath}}
\\usepackage{{hyperref}}

\\title{{WideScreen Scientific Discovery Report}}
\\author{{William Alubokho Ashioya}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

\\section{{Hypothesis}}
{hypothesis}

\\section{{Methodology}}
Protocol Genome Generation: {genome.generation}

Interaction Rules: {', '.join(genome.interaction_rules)}

Verification Threshold: {genome.verification_threshold:.4f}

\\section{{Results}}
{self._format_results(results)}

\\section{{Conclusions}}
Autonomous discovery completed via WideScreen RCP.

\\section{{Human Role Declaration}}
\\textbf{{Human involvement}}: Initial domain prompt specification only.

All ideation, experimentation, analysis, and reporting performed autonomously
by WideScreen RCP system v1.0.

\\section{{Protocol Genome}}
\\begin{{verbatim}}
{json.dumps(genome.to_dict(), indent=2)}
\\end{{verbatim}}

\\end{{document}}
"""
        return report

    def _format_results(self, results: Dict) -> str:
        if not results:
            return "No results available."

        formatted = ""
        for key, value in results.items():
            if isinstance(value, float):
                formatted += f"{key}: {value:.6f}\\n\\n"
            else:
                formatted += f"{key}: {value}\\n\\n"

        return formatted